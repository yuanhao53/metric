import itertools
import numpy as np
from .metrics import ranked_state, discounts, names, cutoff_constant
from .common import write_csv, write_json, figure_module, savefig


def observed_constant(values, source, target):
    a,b=values[:,source],values[:,target]
    witnesses=np.flatnonzero((a<=1e-12)&(b>1e-10))
    if len(witnesses):return np.inf,int(witnesses[0])
    use=np.flatnonzero(a>1e-12)
    if not len(use):return 0.,0
    j=int(use[np.argmax(b[use]/a[use])])
    return float(b[j]/a[j]),j


def enumerate_lists(config,out):
    cfg=config['enumeration'];nmax=cfg['n_max'];rows=[];cross=[];psi=[];count=0
    for n in range(2,nmax+1):
        for m in range(1,n):
            patterns=[];states=[]
            for positions in itertools.combinations(range(n),m):
                y=np.zeros(n,dtype=int);y[list(positions)]=1
                patterns.append(''.join(map(str,y)))
                states.append(ranked_state(y,range(1,n+1)))
            count+=len(states)
            for k in range(1,n+1):
                ns=names(k);reg=np.array([[s.regrets[name] for name in ns] for s in states])
                for i,source in enumerate(ns):
                    for j,target in enumerate(ns):
                        theoretical,finite=cutoff_constant(n,m,k,source,target)
                        actual,witness=observed_constant(reg,i,j)
                        if np.isfinite(actual)!=finite:
                            raise AssertionError((n,m,k,source,target,'finiteness',actual,finite))
                        error=abs(actual-theoretical) if finite and theoretical is not None else None
                        if error is not None and error>1e-9*max(1,abs(theoretical)):
                            raise AssertionError((n,m,k,source,target,actual,theoretical))
                        rows.append(dict(n=n,m=m,k=k,source=source,target=target,observed=actual,
                            theoretical=theoretical,theory_status='sharp' if theoretical is not None else 'finite_only',
                            error=error,witness=patterns[witness]))
            if n<=cfg['cross_n_max']:
                for k1 in range(1,n):
                    for k2 in range(k1+1,n+1):
                        for metric in ['Precision','Recall','NDCG']:
                            reg=np.array([[s.regrets[f'{metric}@{k}'] for k in [k1,k2]] for s in states])
                            for i,j in [(0,1),(1,0)]:
                                actual,witness=observed_constant(reg,i,j)
                                if metric=='NDCG':
                                    finite=(i==1 or k1>=m);expected=None if finite else np.inf
                                elif k2==n:
                                    finite=i==0;expected=0. if finite else np.inf
                                else:
                                    finite=(k1>=m if i==0 else k2<=m)
                                    expected=((k1/k2 if i==0 else k2/k1) if metric=='Precision' else 1.) if finite else np.inf
                                assert np.isfinite(actual)==finite,(n,m,k1,k2,metric,i,j)
                                if finite and expected is not None:assert np.isclose(actual,expected,atol=1e-10)
                                cross.append(dict(n=n,m=m,k_source=[k1,k2][i],k_target=[k1,k2][j],metric=metric,
                                    observed=actual,theoretical=expected,witness=patterns[witness]))
            if n==min(8,nmax) and m==max(1,n//3):
                for a,b in [('AUC','NDCG'),('NDCG','AUC')]:
                    src=np.array([s.regrets[a] for s in states]);dst=np.array([s.regrets[b] for s in states])
                    for e in sorted(set(src)):
                        psi.append(dict(n=n,m=m,source=a,target=b,epsilon=e,psi=float(dst[src<=e+1e-14].max())))
        print(f'Enumeration n={n}: passed',flush=True)
    write_csv(out/'constants.csv',rows);write_csv(out/'cross_cutoffs.csv',cross);write_csv(out/'transfer_functions.csv',psi)
    plt=figure_module();fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    for ax,(a,b) in zip(axes,[('AUC','NDCG'),('NDCG','AUC')]):
        sub=[r for r in psi if r['source']==a];x=[r['epsilon'] for r in sub];y=[r['psi'] for r in sub]
        n,m=sub[0]['n'],sub[0]['m'];c=cutoff_constant(n,m,1,a,b)[0]
        xx=np.linspace(0,1,301);ax.step(x,y,where='post',label='Enumerated transfer function')
        ax.plot(xx,np.minimum(1,c*xx),'--',label='Optimal linear envelope')
        ax.set(xlabel=f'{a} regret budget',ylabel=f'Worst {b} regret',title=f'{a} to {b}: n={n}, m={m}')
        ax.legend(fontsize=8)
    savefig(fig,out,'transfer_functions');plt.close(fig)
    return {'binary_patterns':count,'constant_checks':len(rows),'cross_cutoff_checks':len(cross),
            'max_closed_form_error':max(r['error'] or 0 for r in rows),'all_checks_passed':True}


def scaling(config,out):
    cfg=config['scaling'];rows=[]
    for n in cfg['sizes']:
        regimes=[('proportional',a,max(1,min(n-1,int(a*n)))) for a in cfg['alphas']]
        regimes += [('fixed_positive',m,m) for m in cfg['positive_counts'] if m<n]
        for regime,param,m in regimes:
            ell=n-m;w=discounts(n);z=w[:m].sum();d=w[:-1]-w[1:]
            c1=ell*(w[0]-w[m])/z;c2=z/(m*(w[m-1]-w[-1]))
            first=np.r_[0,np.ones(m,dtype=int),np.zeros(ell-1,dtype=int)]
            second=np.r_[np.ones(m-1,dtype=int),np.zeros(ell,dtype=int),1]
            s1=ranked_state(first);s2=ranked_state(second)
            observed1=s1.regrets['NDCG']/s1.regrets['AUC']
            observed2=s2.regrets['AUC']/s2.regrets['NDCG']
            assert np.isclose(c1,observed1,rtol=1e-8) and np.isclose(c2,observed2,rtol=1e-8)
            # Local ratios are computed using two actual permutations with 01 at the requested boundary.
            local=[]
            for r in [0,n-2]:
                y=np.zeros(n,dtype=int);y[r+1]=1
                available=np.r_[np.arange(r),np.arange(r+2,n)]
                y[available[:m-1]]=1
                a=ranked_state(y);y[r],y[r+1]=1,0;b=ranked_state(y)
                local.append((b.values['NDCG']-a.values['NDCG'])/(b.values['AUC']-a.values['AUC']))
            rows.append(dict(n=n,m=m,regime=regime,parameter=param,auc_to_ndcg=c1,ndcg_to_auc=c2,
                attained_auc_to_ndcg=observed1,attained_ndcg_to_auc=observed2,
                head_gain_ratio=local[0],tail_gain_ratio=local[1],
                local_head_formula=m*ell*d[0]/z,local_tail_formula=m*ell*d[-1]/z))
    write_csv(out/'scaling.csv',rows)
    plt=figure_module();fig,axes=plt.subplots(1,3,figsize=(12,3.4))
    for regime,param in [('proportional',a) for a in cfg['alphas']]+[('fixed_positive',m) for m in cfg['positive_counts']]:
        sub=[r for r in rows if r['regime']==regime and r['parameter']==param]
        if not sub:continue
        ax=axes[0 if regime=='proportional' else 1];x=np.array([r['n'] for r in sub])
        ax.loglog(x,[r['auc_to_ndcg'] for r in sub],label=f'A to N, {param}')
        ax.loglog(x,[r['ndcg_to_auc'] for r in sub],'--',label=f'N to A, {param}')
    sub=[r for r in rows if r['regime']=='proportional' and r['parameter']==cfg['alphas'][-1]]
    for key,label in [('auc_to_ndcg','Global A to N'),('head_gain_ratio','Head swap'),('tail_gain_ratio','Tail swap')]:
        axes[2].loglog([r['n'] for r in sub],[r[key] for r in sub],label=label)
    for ax,title in zip(axes,['Fixed positive fraction','Fixed positive count','Global versus local']):
        ax.set(xlabel='List size n',ylabel='Coefficient',title=title);ax.legend(fontsize=7)
    savefig(fig,out,'scaling');plt.close(fig)
    # Dividing by the claimed rate is more informative than fitting a power law to log n.
    normalized=[]
    for r in rows:
        factor=np.log(r['n']) if r['regime']=='proportional' else r['n']
        reverse=np.log(r['n']) if r['regime']=='proportional' else 1
        normalized.append({**r,'forward_rate_normalized':r['auc_to_ndcg']/factor,'reverse_rate_normalized':r['ndcg_to_auc']/reverse})
    write_csv(out/'normalized_scaling.csv',normalized)
    return {'settings':len(rows),'extremizer_checks_passed':True,'interpretation':'numerical illustration plus independent extremizer evaluation; not training evidence'}
