"""Descriptive diagnostics only; endpoint decompositions are not forecasts."""
import csv
import numpy as np
from .common import write_csv,figure_module,savefig

def trajectory_report(out):
    rng=np.random.default_rng(27);points=[];seen=0;groups={}
    for path in sorted(out.glob('*-seed*/query_transitions.csv')):
        with open(path,encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['split']!='test':continue
                if float(r['P'])+float(r['Q'])==0:continue
                conflict=r['opposite_sign']=='True';key=(r['loss'],r['seed'],conflict)
                acc=groups.setdefault(key,{'count':0,**{f'{sign}_{reg}':0. for sign in ['positive','negative'] for reg in ['head','middle','tail']}})
                acc['count']+=1
                for sign in ['positive','negative']:
                    for reg in ['head','middle','tail']:acc[f'{sign}_{reg}']+=float(r[f'ndcg_{sign}_{reg}'])
                point=[float(r['delta_AUC']),float(r['delta_NDCG']),float(r['ndcg_head'])]
                seen+=1
                if len(points)<5000:points.append(point)
                else:
                    j=int(rng.integers(seen))
                    if j<5000:points[j]=point
    rows=[]
    for (loss,seed,conflict),acc in groups.items():
        rows.append({'loss':loss,'seed':seed,'conflict':conflict,'moving_query_windows':acc['count'],
                     **{k:v/acc['count'] for k,v in acc.items() if k!='count'}})
    write_csv(out/'position_contributions.csv',rows)
    if not points:return
    plt=figure_module();fig,axes=plt.subplots(1,2,figsize=(10,3.8));arr=np.array(points)
    sc=axes[0].scatter(arr[:,0],arr[:,1],c=arr[:,2],cmap='coolwarm',s=8,alpha=.5)
    axes[0].axhline(0,color='gray',lw=.6);axes[0].axvline(0,color='gray',lw=.6)
    axes[0].set(xlabel='Per-query AUC change',ylabel='Per-query full NDCG change')
    fig.colorbar(sc,ax=axes[0],label='Head-prefix NDCG contribution')
    x=np.arange(3)
    for conflict,offset,label in [(False,-.18,'No sign conflict'),(True,.18,'Sign conflict')]:
        sub=[r for r in rows if r['conflict']==conflict];total=sum(r['moving_query_windows'] for r in sub)
        if not total:continue
        for sign,alpha in [('positive',.9),('negative',.35)]:
            vals=[sum(r[f'{sign}_{reg}']*r['moving_query_windows'] for r in sub)/total for reg in ['head','middle','tail']]
            axes[1].bar(x+offset,np.array(vals)*(1 if sign=='positive' else -1),width=.34,alpha=alpha,label=f'{label}: {sign}')
    axes[1].set_xticks(x,['Head','Middle','Tail']);axes[1].axhline(0,color='gray',lw=.6)
    axes[1].set(ylabel='Mean signed NDCG contribution');axes[1].legend(fontsize=7)
    savefig(fig,out,'position_and_conflict');plt.close(fig)
