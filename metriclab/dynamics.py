import numpy as np
from .metrics import ranked_state,score_state,transition
from .common import write_csv,write_json,figure_module,savefig

def sigmoid(s):
    return np.exp(-np.logaddexp(0,-s))

def population_loss(s,eta):
    return float(np.sum(np.logaddexp(0,s)-eta*s))

def inversion_mask(s,eta,tol=1e-10):
    return (eta[:,None]>eta[None,:]+tol)&(s[:,None]<s[None,:]-tol)

def controlled(config,out):
    cfg=config['dynamics'];rng=np.random.default_rng(config['seed']);n=cfg['n'];m=cfg['m']
    cutoffs=sorted(set([1,min(5,n),min(10,n)]));rows=[]
    for trial in range(cfg['trials']):
        for boundary in [0,n//2,n-2]:
            y=np.zeros(n,dtype=int);y[boundary+1]=1
            avail=np.setdiff1d(np.arange(n),[boundary,boundary+1]);rng.shuffle(avail)
            y[avail[:m-1]]=1;a=ranked_state(y,cutoffs)
            y[boundary],y[boundary+1]=1,0;b=ranked_state(y,cutoffs)
            rows.append({'trial':trial,'scenario':'favorable_single_swap','boundary':boundary+1,**transition(a,b,cutoffs)})
    # Feasible mixed update, not an arbitrary synthetic vector h.
    a=ranked_state([1,0,0,0,0,1,0,1],[1,3,5]);b=ranked_state([0,1,0,0,1,0,1,0],[1,3,5])
    rows.append({'trial':0,'scenario':'mixed_head_loss_tail_gain','boundary':None,**transition(a,b,[1,3,5])})
    assert rows[-1]['auc_up_ndcg_down']
    write_csv(out/'structural_updates.csv',rows)
    # Optimization uses 16 scores so that every pair is checked at every step.
    size=min(16,n);traces=[];summaries=[]
    for trial in range(cfg['trials']):
        eta=rng.uniform(.05,.95,size);labels=(eta>=.5).astype(int)
        labels[0],labels[-1]=0,1
        s0=rng.normal(0,1.5,size)
        b=np.eye(size)+rng.normal(0,.25,(size,size))
        # Full-rank parameterization, same score expressivity; normalize largest curvature.
        if np.linalg.matrix_rank(b)!=size:raise AssertionError('Unexpected singular parameterization')
        k=b@b.T;k/=np.linalg.eigvalsh(k)[-1]
        for eta_type,target in [('deterministic',labels.astype(float)),('population',eta)]:
            for model,gram in [('independent',np.eye(size)),('coupled',k)]:
                s=s0.copy();old=score_state(labels,s,[1,5,10] if size>=10 else [1])
                inv=inversion_mask(s,target);new_inversions=0;conflicts=0;prefix_reversals=0;loss_increases=0
                initial_loss=population_loss(s,target)
                for step in range(cfg['steps']):
                    loss=population_loss(s,target)
                    nxt=s-cfg['step_size']*(gram@(sigmoid(s)-target))
                    newloss=population_loss(nxt,target)
                    newinv=inversion_mask(nxt,target)
                    added=int(np.sum(newinv&~inv));new_inversions+=added
                    new=score_state(labels,nxt,[1,5,10] if size>=10 else [1])
                    delta=transition(old,new,[]) # report full metrics; exact identity independent of objective
                    conflicts+=int(delta['opposite_sign']);prefix_reversals+=int(delta['Q']>0)
                    loss_increases+=int(newloss>loss+1e-10)
                    if trial==0 and (step%10==0 or added or delta['P'] or delta['Q']):
                        traces.append({'trial':trial,'eta_type':eta_type,'model':model,'step':step+1,
                            'loss':newloss,'eta_inversions':int(newinv.sum()),'new_inversions':added,
                            'AUC':new.values['AUC'],'NDCG':new.values['NDCG'],**delta})
                    s=nxt;old=new;inv=newinv
                if model=='independent':
                    assert new_inversions==0
                    if eta_type=='deterministic':assert prefix_reversals==0
                assert loss_increases==0,'Reduce step_size; loss increased in controlled convex objective'
                summaries.append({'trial':trial,'eta_type':eta_type,'model':model,'new_eta_inversions':new_inversions,
                    'realized_label_conflicts':conflicts,'prefix_reversal_steps':prefix_reversals,
                    'initial_loss':initial_loss,'final_loss':population_loss(s,target),'loss_increases':loss_increases})
    write_csv(out/'optimizer_trials.csv',summaries);write_csv(out/'optimizer_trace.csv',traces)
    # Explicit coupled counterexample near the correct side of a tie.
    eta=np.array([.9,.8]);b=np.array([[1.,0.],[2.,1.]]);s=np.array([1e-4,0.])
    v=-(b@b.T)@(sigmoid(s)-eta);step=1e-3;sn=s+step*v
    witness={'eta':eta,'B':b,'initial_scores':s,'velocity':v,'updated_scores':sn,
             'loss_before':population_loss(s,eta),'loss_after':population_loss(sn,eta)}
    assert s[0]>s[1] and sn[0]<sn[1] and witness['loss_after']<witness['loss_before']
    write_json(out/'shared_parameter_witness.json',witness)
    plt=figure_module();fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for typ in ['deterministic','population']:
        for model in ['independent','coupled']:
            sub=[r for r in traces if r['eta_type']==typ and r['model']==model]
            for ax,key in zip(axes,['loss','eta_inversions','NDCG']):
                ax.plot([r['step'] for r in sub],[r[key] for r in sub],label=f'{typ}/{model}')
    for ax,key in zip(axes,['Population BCE','Eta-order inversions','Realized-label NDCG']):
        ax.set(xlabel='Step',ylabel=key);ax.legend(fontsize=6)
    savefig(fig,out,'controlled_dynamics');plt.close(fig)
    return {'trials':len(summaries),'independent_ordering_verified':True,'shared_counterexample_verified':True,
            'max_identity_error':max(r['identity_error'] for r in rows),
            'note':'Population eta ordering is not monotonicity for each realized label vector; P,Q are net prefix masses, not swap counts.'}
