"""Small NumPy MF scorer with analytically differentiated BCE/BPR losses."""
import numpy as np
from .dynamics import sigmoid

class MatrixFactorization:
    def __init__(self,nu,ni,dim,seed):
        rng=np.random.default_rng(seed)
        self.params={'U':rng.normal(0,.1,(nu,dim)),'V':rng.normal(0,.1,(ni,dim)),
                     'bu':np.zeros(nu),'bi':np.zeros(ni),'bias':np.zeros(1)}

    def score(self,u,i):
        p=self.params
        return np.sum(p['U'][u]*p['V'][i],axis=-1)+p['bu'][u]+p['bi'][i]+p['bias'][0]

    def gradients(self,loss,u,i,y=None,j=None,l2=0.):
        p=self.params;g={k:np.zeros_like(v) for k,v in p.items()};b=len(u)
        if loss=='bce':
            s=self.score(u,i);value=float(np.mean(np.logaddexp(0,s)-y*s));d=(sigmoid(s)-y)/b
            np.add.at(g['U'],u,d[:,None]*p['V'][i]);np.add.at(g['V'],i,d[:,None]*p['U'][u])
            np.add.at(g['bu'],u,d);np.add.at(g['bi'],i,d);g['bias'][0]=d.sum()
        elif loss=='bpr':
            s=self.score(u,i)-self.score(u,j);value=float(np.mean(np.logaddexp(0,-s)));d=(sigmoid(s)-1)/b
            np.add.at(g['U'],u,d[:,None]*(p['V'][i]-p['V'][j]))
            np.add.at(g['V'],i,d[:,None]*p['U'][u]);np.add.at(g['V'],j,-d[:,None]*p['U'][u])
            np.add.at(g['bi'],i,d);np.add.at(g['bi'],j,-d)
        else:raise ValueError(loss)
        # Define regularization explicitly, including the coefficient in recorded objective.
        for key in ['U','V','bu','bi']:
            value+=.5*l2*float(np.sum(p[key]**2));g[key]+=l2*p[key]
        return value,g

    def save(self,path):np.savez_compressed(path,**self.params)

class Adam:
    def __init__(self,params,lr):
        self.lr=lr;self.t=0;self.m={k:np.zeros_like(v) for k,v in params.items()};self.v={k:np.zeros_like(v) for k,v in params.items()}
    def step(self,params,grads):
        self.t+=1
        for k,p in params.items():
            g=grads[k];self.m[k]=.9*self.m[k]+.1*g;self.v[k]=.999*self.v[k]+.001*g*g
            p-=self.lr*(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)
