"""Generate descriptive paper tables from an already completed training run."""
import argparse
import csv
import json
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from metriclab.common import write_csv,write_json,figure_module,savefig
from metriclab.metrics import score_state,transition

def read(path):
    with open(path,encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
    run=Path(args.run).resolve();root=run/'train';out=run/'analysis';out.mkdir(exist_ok=True)
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete':raise ValueError('Training run is not complete')
    cfg=manifest['config']['training'];primary=f'NDCG@{max(cfg["cutoffs"])}';last=cfg['epochs']
    history=read(root/'history.csv');selected=read(root/'checkpoint_selection.csv');runs=read(root/'runs.csv')
    meta=json.loads((root/'data_protocol.json').read_text(encoding='utf-8'))
    cohort=[r for r in read(root/'cohort.csv') if r['included']=='True']
    cohort_summary={}
    for split in ['validation','test']:
        ns=np.array([int(r[f'{split}_n']) for r in cohort]);ms=np.array([int(r[f'{split}_m']) for r in cohort])
        cohort_summary[split]={'users':len(ns),'n_min':int(ns.min()),'n_median':float(np.median(ns)),'n_max':int(ns.max()),
                               'm_median':float(np.median(ms)),'single_positive_users':int(np.sum(ms==1)),
                               'mean_positive_fraction':float(np.mean(ms/ns))}
    write_json(out/'cohort_summary.json',cohort_summary)
    aggregates={};cases=[];max_error=0.;single_conflicts=0
    for path in sorted(root.glob('*-seed*/query_transitions.csv')):
        with open(path,encoding='utf-8') as f:
            for row in csv.DictReader(f):
                max_error=max(max_error,float(row['identity_error']))
                if row['split']!='test':continue
                epoch=int(row['epoch']);moving=float(row['P'])+float(row['Q'])>0
                da=float(row['delta_AUC']);dn=float(row['delta_NDCG']);dk=float(row[f'delta_{primary}'])
                conflict=da*dn < -1e-24;trunc_conflict=da*dk < -1e-24
                single_conflicts+=int(int(row['m'])==1 and conflict)
                for period,ok in [('all',True),('epochs_2_onward',epoch>=2),('last_10_epochs',epoch>max(0,last-10))]:
                    if not ok:continue
                    key=(row['loss'],row['seed'],period)
                    a=aggregates.setdefault(key,defaultdict(float))
                    a['query_windows']+=1;a['moving_windows']+=moving
                    a['multi_positive_moving']+=moving and int(row['m'])>=2
                    a['full_conflicts']+=conflict;a['truncated_conflicts']+=trunc_conflict
                    a['auc_up_ndcg_down']+=da>1e-12 and dn< -1e-12
                    a['auc_down_ndcg_up']+=da< -1e-12 and dn>1e-12
                    a['auc_flat_ndcg_changes']+=abs(da)<=1e-12 and abs(dn)>1e-12
                    a['mixed_prefix_windows']+=float(row['P'])>0 and float(row['Q'])>0
                if epoch>=2 and da>.001 and dn<-.001 and int(row['m'])>=2 and int(row['n'])<=150:
                    cases.append(row)
    rates=[]
    for (loss,seed,period),a in aggregates.items():
        rates.append({'loss':loss,'seed':seed,'period':period,**dict(a),
            'full_conflict_rate_moving':a['full_conflicts']/max(1,a['moving_windows']),
            'full_conflict_rate_multi_positive_moving':a['full_conflicts']/max(1,a['multi_positive_moving']),
            'truncated_conflict_rate_moving':a['truncated_conflicts']/max(1,a['moving_windows']),
            'mixed_prefix_rate_moving':a['mixed_prefix_windows']/max(1,a['moving_windows'])})
    write_csv(out/'conflict_rates_by_seed.csv',rates)
    rate_summary=[]
    for loss in cfg['losses']:
        for period in ['all','epochs_2_onward','last_10_epochs']:
            rs=[r for r in rates if r['loss']==loss and r['period']==period]
            row={'loss':loss,'period':period,'seeds':len(rs)}
            for metric in ['full_conflict_rate_moving','full_conflict_rate_multi_positive_moving','truncated_conflict_rate_moving','mixed_prefix_rate_moving']:
                vals=[r[metric] for r in rs];row[metric+'_mean']=float(np.mean(vals));row[metric+'_std']=float(np.std(vals,ddof=1))
            rate_summary.append(row)
    write_csv(out/'conflict_rates_summary.csv',rate_summary)

    # Separately quantify sign conflicts in macro-averaged trajectories.
    macro=[]
    for loss in cfg['losses']:
        for seed in cfg['seeds']:
            rows=sorted([r for r in history if r['loss']==loss and int(r['seed'])==seed and r['split']=='test'],key=lambda r:int(r['epoch']))
            for prev,cur in zip(rows,rows[1:]):
                da=float(cur['AUC'])-float(prev['AUC']);dn=float(cur['NDCG'])-float(prev['NDCG']);dk=float(cur[primary])-float(prev[primary])
                macro.append({'loss':loss,'seed':seed,'epoch':int(cur['epoch']),'delta_AUC':da,'delta_NDCG':dn,
                    f'delta_{primary}':dk,'full_conflict':da*dn< -1e-24,'truncated_conflict':da*dk< -1e-24})
    write_csv(out/'macro_transitions.csv',macro)
    plt=figure_module();fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for loss,color in zip(cfg['losses'],['tab:blue','tab:orange']):
        for ax,metric in zip(axes,['AUC',primary,'conflict_rate_multi_positive']):
            epochs=sorted(set(int(r['epoch']) for r in history if r['split']=='test' and r['loss']==loss and (metric!='conflict_rate_multi_positive' or int(r['epoch'])>0)))
            means=[];sds=[]
            for epoch in epochs:
                vals=[float(r[metric]) for r in history if r['split']=='test' and r['loss']==loss and int(r['epoch'])==epoch]
                means.append(np.mean(vals));sds.append(np.std(vals,ddof=1))
            means=np.asarray(means);sds=np.asarray(sds)
            ax.plot(epochs,means,label=loss.upper(),color=color);ax.fill_between(epochs,means-sds,means+sds,color=color,alpha=.18)
    for ax,label in zip(axes,['Test AUC',f'Test {primary}','AUC / full NDCG conflict rate']):
        ax.set(xlabel='Epoch',ylabel=label);ax.legend()
    axes[2].set_title('Moving multi-positive queries',fontsize=9)
    savefig(fig,out,'paper_training_summary');plt.close(fig)

    # Independently recompute initial/final checkpoints and verify fixed candidate identities.
    checked=0;max_metric_error=0.;fingerprints={};initial_score_hashes={}
    for loss in cfg['losses']:
        for seed in cfg['seeds']:
            for split in ['validation','test']:
                for epoch in [0,last]:
                    path=root/f'{loss}-seed{seed}'/'scores'/f'{split}-epoch{epoch:03d}.npz'
                    with np.load(path) as f:
                        users=f['user_ids'];items=f['item_ids'];labels=f['labels'];scores=f['scores']
                        fingerprint=hashlib.sha256(users.tobytes()+items.tobytes()+labels.tobytes()).hexdigest()
                        if split in fingerprints:assert fingerprint==fingerprints[split],'Candidate lists changed'
                        fingerprints[split]=fingerprint
                        if epoch==0:
                            init_key=f'{split}-seed{seed}';score_hash=hashlib.sha256(scores.tobytes()).hexdigest()
                            if init_key in initial_score_hashes:assert score_hash==initial_score_hashes[init_key],'Initialization differs across losses'
                            initial_score_hashes[init_key]=score_hash
                        boundaries=np.r_[0,np.flatnonzero(np.diff(users))+1,len(users)];values=[]
                        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
                            order=np.lexsort((items[lo:hi],-scores[lo:hi]));y=labels[lo:hi][order];m=int(y.sum());n=len(y)
                            # Direct pair counting, independent of the production inversion formula.
                            positives=np.flatnonzero(y);negatives=np.flatnonzero(1-y)
                            auc=float(np.mean(positives[:,None]<negatives))
                            w=1/np.log2(np.arange(n)+2);nd=float(w@y/w[:m].sum());k=max(cfg['cutoffs'])
                            nk=float(w[:k]@y[:k]/w[:min(k,m)].sum());values.append([auc,nd,nk])
                        means=np.mean(values,axis=0)
                        row=next(r for r in history if r['loss']==loss and int(r['seed'])==seed and r['split']==split and int(r['epoch'])==epoch)
                        err=max(abs(means[i]-float(row[metric])) for i,metric in enumerate(['AUC','NDCG',primary]))
                        max_metric_error=max(max_metric_error,err);checked+=1
    assert max_error<1e-10 and max_metric_error<1e-10 and single_conflicts==0
    integrity={'saved_checkpoints_recomputed':checked,'fixed_candidates_verified':True,'paired_initialization_verified':True,'max_saved_score_metric_error':max_metric_error,
               'max_transition_identity_error':max_error,'single_positive_conflicts':single_conflicts,'candidate_hashes':fingerprints}
    write_json(out/'integrity_checks.json',integrity)

    # One explicitly labeled illustrative endpoint conflict, with complete positive ranks.
    cases=sorted(cases,key=lambda r:float(r['delta_NDCG']))
    write_csv(out/'illustrative_conflicts.csv',cases[:20])
    if cases:
        case=cases[0];states=[];ranksets=[];epoch=int(case['epoch']);user=int(case['user'])
        for t in [epoch-1,epoch]:
            with np.load(root/f'{case["loss"]}-seed{case["seed"]}'/'scores'/f'test-epoch{t:03d}.npz') as f:
                mask=f['user_ids']==user;state=score_state(f['labels'][mask],f['scores'][mask],cfg['cutoffs'],f['item_ids'][mask])
                states.append(state);ranksets.append((np.flatnonzero(state.y)+1).tolist())
        h=(states[1].prefix-states[0].prefix)[:-1];z=transition(*states,cfg['cutoffs'])
        write_json(out/'illustrative_conflict.json',{'selection_rule':'largest full-NDCG drop among AUC-improving windows with n<=150, epoch>=2; illustrative, not representative',
            'loss':case['loss'],'seed':int(case['seed']),'user':user,'epoch':epoch,'positive_ranks_before':ranksets[0],'positive_ranks_after':ranksets[1],**z})
        plt=figure_module();fig,ax=plt.subplots(figsize=(8,3));ax.bar(np.arange(1,len(h)+1),h,color=np.where(h>=0,'tab:blue','tab:red'))
        ax.axhline(0,color='black',lw=.5);ax.set(xlabel='Prefix boundary',ylabel='Net prefix change',title=f'Illustrative update: AUC {z["delta_AUC"]:+.4f}, NDCG {z["delta_NDCG"]:+.4f}')
        savefig(fig,out,'illustrative_conflict');plt.close(fig)

    # Paired initialization comparison: means and sample SD across independent training seeds.
    table=[]
    for loss in cfg['losses']:
        for criterion in ['AUC',primary]:
            row={'loss':loss,'selected_by':criterion}
            for metric in ['AUC','NDCG',primary,'Precision@10','Recall@10']:
                vals=[float(r['test_mean']) for r in selected if r['loss']==loss and r['selected_by']==criterion and r['test_metric']==metric]
                row[metric+'_mean']=float(np.mean(vals));row[metric+'_std']=float(np.std(vals,ddof=1))
            table.append(row)
    write_csv(out/'test_performance_summary.csv',table)
    lines=['# 正式实验结果','',f'运行：`{run.name}`。完整数据 {meta["users"]:,} 位用户、{meta["ratings"]:,} 条评分；固定评估用户 {meta["eligible_eval_users"]:,} 位。BCE/BPR 各 3 个种子、30 epoch。','',
        '## 测试表现','', '以下为三个训练种子的均值 ± 样本标准差。checkpoint 仅由验证集选择。', '', '| Loss | 验证选择指标 | AUC | NDCG | NDCG@10 |','|---|---|---|---|---|']
    for row in table:lines.append('| '+row['loss'].upper()+' | '+row['selected_by']+' | '+' | '.join(f'{row[k+"_mean"]:.4f} ± {row[k+"_std"]:.4f}' for k in ['AUC','NDCG',primary])+' |')
    lines+=['','## 逐用户动态冲突','','分母是发生非零前缀变化的用户—checkpoint 窗口，以下均排除第一个训练窗口；统计为三种子的均值 ± 标准差。它们是描述性频率，不把相关窗口视为独立样本。','','| Loss | AUC/全列表 NDCG 异向 | AUC/NDCG@10 异向 | 同时存在正负前缀变化 |','|---|---|---|---|']
    for row in rate_summary:
        if row['period']!='epochs_2_onward':continue
        lines.append('| '+row['loss'].upper()+' | '+' | '.join(f'{100*row[k+"_mean"]:.2f}% ± {100*row[k+"_std"]:.2f} pp' for k in ['full_conflict_rate_moving','truncated_conflict_rate_moving','mixed_prefix_rate_moving'])+' |')
    lines+=['','## 验证指标选择的影响','','各种子上，按验证 NDCG@10 选择相对于按验证 AUC 选择的测试 NDCG@10 差值如下。区间是该种子内按用户进行的配对 bootstrap，不能作为三种子整体置信区间。','','| Loss | Seed | AUC-selected epoch | NDCG-selected epoch | 测试差值 | 配对用户 bootstrap 95% 区间 |','|---|---|---|---|---|---|']
    for r in runs:lines.append(f'| {r["loss"].upper()} | {r["seed"]} | {r["auc_selected_epoch"]} | {r["ndcg_selected_epoch"]} | {float(r["paired_target_gain_ndcg_selection"]):+.5f} | [{float(r["paired_bootstrap_low"]):+.5f}, {float(r["paired_bootstrap_high"]):+.5f}] |')
    lines+=['','## 完整性检查','',f'- 独立复算 {checked} 份初始/最终 checkpoint，最大指标误差 {max_metric_error:.3g}。',f'- 所有记录窗口的最大前缀恒等式残差 {max_error:.3g}。','- 初始/最终分数文件在各损失、种子间的候选与标签一致。',f'- 单正例用户的全列表 AUC/NDCG 方向冲突数：{single_conflicts}。','',
        '## 论文解读边界','','这些结果检验固定已评分候选列表上的指标变化。它们不检验全物品库召回、线上收益或因果迁移，也不能建立 BCE/BPR 类别的普遍优劣。前缀分解重构是恒等式核验，不是提前预测。','',
        '用户按时间切分，但不同用户之间并非全局时间切分；未评分项未被视为负例。m 和 n 随用户变化，全列表与截断指标分开报告。末尾十轮、包含初始窗口和宏平均方向冲突的补充结果分别保存在 CSV 中。','',
        '## 文件','', '- `test_performance_summary.csv`：正式性能表。','- `conflict_rates_summary.csv` / `conflict_rates_by_seed.csv`：分时段、分种子的冲突统计。','- `macro_transitions.csv`：宏平均变化，不能与逐用户频率混用。','- `integrity_checks.json`：复算与候选一致性核验。','- `illustrative_conflict.json` / `.png`：一个明确标注筛选规则的真实轨迹实例。','- 上级 `train/` 保存全量逐用户诊断、分数、原始表与图。']
    core_candidates=[p for p in sorted(run.parent.glob('*-core')) if (p/'summary.json').exists()]
    if core_candidates:
        core=core_candidates[-1];cs=json.loads((core/'summary.json').read_text(encoding='utf-8'))
        write_json(out/'theory_results.json',{'source_run':str(core),**cs})
        lines+=['','## 已完成的正式理论实验','',f'复用已完成并归档的 `{core.name}`，本次没有重复计算：',
            f'- 枚举 {cs["enumerate"]["binary_patterns"]:,} 个二元排列，检查 {cs["enumerate"]["constant_checks"]:,} 项常数和 {cs["enumerate"]["cross_cutoff_checks"]:,} 项跨 cutoff 关系，全部通过。',
            f'- {cs["scaling"]["settings"]} 组尺度设置通过达到边界的排列核验。',f'- {cs["dynamics"]["trials"]} 条受控优化轨迹完成，独立 logits 排序保持与共享参数反例均通过。']
        for sub,name in [('scaling','scaling'),('dynamics','controlled_dynamics')]:
            for ext in ['png','pdf']:shutil.copy2(core/sub/f'{name}.{ext}',out/f'{name}.{ext}')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    tex=['% Mean and sample SD over three seeds. Checkpoints selected on validation only.',r'\begin{tabular}{llccc}',r'\hline',r'Loss & Selection & AUC & NDCG & NDCG@10 \\',r'\hline']
    for row in table:tex.append(row['loss'].upper()+' & '+row['selected_by']+' & '+' & '.join(f'${row[k+"_mean"]:.4f} \\pm {row[k+"_std"]:.4f}$' for k in ['AUC','NDCG',primary])+r' \\')
    tex += [r'\hline',r'\end{tabular}']
    (out/'performance_table.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    write_json(out/'analysis_manifest.json',{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training_run':str(run),'phase':'post-run descriptive analysis; no retuning or seed filtering'})
    print(json.dumps({'output':str(out),'integrity':integrity,'conflicts':rate_summary,'performance':table},indent=2))

if __name__=='__main__':main()
