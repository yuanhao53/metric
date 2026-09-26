import csv
import time
import numpy as np
from .data import prepare
from .model import MatrixFactorization,Adam
from .metrics import score_state,transition
from .common import write_csv,write_json,figure_module,savefig
from .reporting import trajectory_report

def bpr_epoch(data,rng,samples):
    groups=data['train_groups']
    positives=[(u,int(i)) for u,(p,neg) in groups.items() for i in p]
    arr=np.array(positives,dtype=int)
    selected=arr[rng.integers(len(arr),size=samples)]
    u,i=selected[:,0],selected[:,1];j=np.empty(samples,dtype=int)
    for user in np.unique(u):
        mask=u==user;neg=groups[int(user)][1]
        j[mask]=rng.choice(neg,size=mask.sum())
    return u,i,j

def evaluate(model,data,split,cutoffs,previous=None):
    metrics=[];states={};diagnostics=[];packed_scores=[];packed_items=[];packed_users=[];packed_labels=[]
    for u,ix in data['groups'][split].items():
        items=data['items'][ix];y=data['labels'][ix];s=model.score(np.full(len(ix),u),items)
        state=score_state(y,s,cutoffs,data['item_ids'][items]);states[u]=state
        metrics.append({'user':int(data['user_ids'][u]),'n':len(ix),'m':int(y.sum()),**state.values,
                        **{f'regret_{k}':v for k,v in state.regrets.items()}})
        packed_scores.extend(s);packed_items.extend(data['item_ids'][items]);packed_users.extend([data['user_ids'][u]]*len(ix));packed_labels.extend(y)
        if previous is not None:
            row={'user':int(data['user_ids'][u]),**transition(previous[u],state,cutoffs)}
            if row['identity_error']>1e-10:raise AssertionError('Metric identity mismatch')
            # Endpoint order reversals only: this does not count all intermediate crossings.
            before_order=previous[u].order;after_order=state.order
            before_rank=np.empty(len(ix),int);before_rank[before_order]=np.arange(len(ix))
            after_rank=np.empty(len(ix),int);after_rank[after_order]=np.arange(len(ix))
            pos=np.flatnonzero(y);neg=np.flatnonzero(1-y)
            old=before_rank[pos,None]<before_rank[neg];new=after_rank[pos,None]<after_rank[neg]
            row['endpoint_pairs_corrected']=int(np.sum(~old&new));row['endpoint_pairs_broken']=int(np.sum(old&~new))
            sorted_s=s[state.order];margins=sorted_s[:-1]-sorted_s[1:]
            row['min_adjacent_margin']=float(margins.min());row['head_adjacent_margin']=float(margins[:min(10,len(margins))].mean())
            diagnostics.append(row)
    arrays={'scores':np.asarray(packed_scores),'item_ids':np.asarray(packed_items),'user_ids':np.asarray(packed_users),'labels':np.asarray(packed_labels)}
    summary={key:float(np.mean([r[key] for r in metrics])) for key in states[next(iter(states))].values}
    if diagnostics:
        eps=1e-12;moving=[r for r in diagnostics if r['P']+r['Q']>0];multi=[r for r in moving if r['m']>=2];single=[r for r in moving if r['m']==1]
        summary.update({'moving_queries':len(moving),'unchanged_queries':len(diagnostics)-len(moving),
                        'conflict_rate_moving':float(np.mean([r['opposite_sign'] for r in moving])) if moving else 0.,
                        'multi_positive_moving_queries':len(multi),
                        'conflict_rate_multi_positive':float(np.mean([r['opposite_sign'] for r in multi])) if multi else 0.,
                        'single_positive_moving_queries':len(single),
                        'single_positive_conflicts':sum(r['opposite_sign'] for r in single),
                        'mixed_prefix_fraction':float(np.mean([r['P']>0 and r['Q']>0 for r in moving])) if moving else 0.,
                        'max_identity_error':max(r['identity_error'] for r in diagnostics)})
        if summary['single_positive_conflicts']:raise AssertionError('Single-positive list cannot have AUC/NDCG sign conflict')
    return states,metrics,diagnostics,summary,arrays

def bootstrap_mean(values,rng,repeats):
    v=np.asarray(values);means=np.empty(repeats)
    for i in range(repeats):means[i]=rng.choice(v,size=len(v),replace=True).mean()
    return float(v.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))

def train_all(config,out):
    cfg=config['training'];data=prepare(config,out);histories=[];selections=[];runs=[]
    primary=f'NDCG@{max(cfg["cutoffs"])}'
    for seed in cfg['seeds']:
        for loss in cfg['losses']:
            folder=out/f'{loss}-seed{seed}';folder.mkdir();(folder/'scores').mkdir()
            model=MatrixFactorization(data['num_users'],data['num_items'],cfg['dimension'],seed)
            opt=Adam(model.params,cfg['learning_rate']);rng=np.random.default_rng(seed+10000)
            previous={};snapshots={};local_history=[];diag_writer=None
            started=time.perf_counter()
            with open(folder/'query_transitions.csv','w',encoding='utf-8',newline='') as df:
                for epoch in range(cfg['epochs']+1):
                    losses=[]
                    if epoch:
                        order=rng.permutation(data['train']);total=len(order)
                        if loss=='bpr':bu,bi,bj=bpr_epoch(data,rng,total)
                        for lo in range(0,total,cfg['batch_size']):
                            hi=min(total,lo+cfg['batch_size'])
                            if loss=='bce':
                                ix=order[lo:hi];value,grads=model.gradients(loss,data['users'][ix],data['items'][ix],y=data['labels'][ix],l2=cfg['l2'])
                            else:value,grads=model.gradients(loss,bu[lo:hi],bi[lo:hi],j=bj[lo:hi],l2=cfg['l2'])
                            opt.step(model.params,grads);losses.append(value)
                    if epoch%cfg['checkpoint_every'] and epoch!=cfg['epochs']:continue
                    snapshots[epoch]={}
                    for split in ['validation','test']:
                        states,metrics,diag,summary,arrays=evaluate(model,data,split,cfg['cutoffs'],previous.get(split))
                        previous[split]=states;snapshots[epoch][split]=metrics
                        row={'seed':seed,'loss':loss,'epoch':epoch,'split':split,
                             'training_objective':float(np.mean(losses)) if losses else None,**summary}
                        histories.append(row);local_history.append(row)
                        if cfg['save_scores']:np.savez_compressed(folder/'scores'/f'{split}-epoch{epoch:03d}.npz',**arrays)
                        for d in diag:
                            full={'seed':seed,'loss':loss,'epoch':epoch,'split':split,**d}
                            if diag_writer is None:diag_writer=csv.DictWriter(df,fieldnames=list(full));diag_writer.writeheader()
                            diag_writer.writerow(full)
                    print(f'{loss} seed={seed} epoch={epoch}: val {primary}={local_history[-2][primary]:.4f}',flush=True)
                # Only validation metrics select a checkpoint; test is never a selection input.
                valrows=[r for r in local_history if r['split']=='validation']
                brng=np.random.default_rng(seed+20000)
                for criterion in ['AUC',primary]:
                    best=max(valrows,key=lambda r:(r[criterion],-r['epoch']))
                    test=snapshots[best['epoch']]['test']
                    for metric in ['AUC','NDCG']+[f'{name}@{k}' for k in cfg['cutoffs'] for name in ['Precision','Recall','NDCG']]:
                        mean,low,high=bootstrap_mean([r[metric] for r in test],brng,cfg['bootstrap_repeats'])
                        selections.append({'seed':seed,'loss':loss,'selected_by':criterion,'epoch':best['epoch'],'test_metric':metric,
                                           'test_mean':mean,'query_bootstrap_low':low,'query_bootstrap_high':high})
                a_epoch=max(valrows,key=lambda r:(r['AUC'],-r['epoch']))['epoch']
                n_epoch=max(valrows,key=lambda r:(r[primary],-r['epoch']))['epoch']
                ta=snapshots[a_epoch]['test'];tn=snapshots[n_epoch]['test']
                diffs=[rn[primary]-ra[primary] for ra,rn in zip(ta,tn)]
                mean,low,high=bootstrap_mean(diffs,brng,cfg['bootstrap_repeats'])
                runs.append({'seed':seed,'loss':loss,'seconds':time.perf_counter()-started,
                             'auc_selected_epoch':a_epoch,'ndcg_selected_epoch':n_epoch,
                             'paired_target_gain_ndcg_selection':mean,'paired_bootstrap_low':low,'paired_bootstrap_high':high})
            write_csv(folder/'history.csv',local_history);model.save(folder/'final_model.npz')
    write_csv(out/'history.csv',histories);write_csv(out/'checkpoint_selection.csv',selections);write_csv(out/'runs.csv',runs)
    # Report seed variation separately: many correlated checkpoint rows are not independent replicates.
    seed_summary=[]
    for loss in cfg['losses']:
        for criterion in ['AUC',primary]:
            for metric in ['AUC','NDCG',primary]:
                vals=[r['test_mean'] for r in selections if r['loss']==loss and r['selected_by']==criterion and r['test_metric']==metric]
                seed_summary.append({'loss':loss,'selected_by':criterion,'metric':metric,'mean':float(np.mean(vals)),
                    'seed_std':float(np.std(vals,ddof=1)) if len(vals)>1 else None,'seeds':len(vals)})
    write_csv(out/'seed_summary.csv',seed_summary)
    plt=figure_module();fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for loss in cfg['losses']:
        for seed in cfg['seeds']:
            sub=[r for r in histories if r['split']=='test' and r['loss']==loss and r['seed']==seed]
            for ax,metric in zip(axes,['AUC',primary,'conflict_rate_multi_positive']):
                ax.plot([r['epoch'] for r in sub],[r.get(metric,0) for r in sub],label=f'{loss}/{seed}')
    for ax,label in zip(axes,['Test AUC',f'Test {primary}','Conflict rate among moving multi-positive queries']):
        ax.set(xlabel='Epoch',ylabel=label);ax.legend(fontsize=7)
    savefig(fig,out,'training_trajectories');plt.close(fig)
    trajectory_report(out)
    return {'runs':runs,'data':data['metadata'],'primary_selection_metric':primary,
            'scope':'observed-label diagnostics; no population Bayes regrets or prospective prediction claims',
            'optimizer':'Adam for real training; gradient-flow ordering theorem not asserted here'}
