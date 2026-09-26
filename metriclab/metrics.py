"""Metrics use fixed labels and a label-independent item-ID tie break.

Regret is the realized-list oracle gap, NOT population Bayes regret.
No sklearn AUC: half-credit for tied scores would violate this convention.
"""
from dataclasses import dataclass
import numpy as np


def discounts(n):
    return 1.0 / np.log2(np.arange(1, n + 1, dtype=float) + 1)


def names(k):
    return ["AUC", "NDCG", f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"]


@dataclass
class State:
    y: np.ndarray
    prefix: np.ndarray
    values: dict
    regrets: dict
    order: np.ndarray | None = None


def ranked_state(y, cutoffs=()):
    y = np.asarray(y)
    if y.ndim != 1 or not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels must be a one-dimensional binary array")
    y = y.astype(np.int64,copy=True)
    n, m = len(y), int(y.sum())
    ell = n - m
    if not (m > 0 and ell > 0):
        raise ValueError("Fixed-list experiments require both classes")
    w = discounts(n)
    h = np.cumsum(y)
    deficit = np.minimum(np.arange(1, n + 1), m) - h
    # Direct positive-rank inversion count, independent of the prefix identity.
    pos = np.flatnonzero(y) + 1
    inversions = int(np.sum(pos - np.arange(1, m + 1)))
    auc = 1 - inversions / (m * ell)
    z = float(w[:m].sum())
    ndcg = float(w @ y / z)
    values = {"AUC": auc, "NDCG": ndcg}
    regrets = {"AUC": inversions/(m*ell), "NDCG": (z-float(w@y))/z}
    for k in sorted(set(cutoffs)):
        if not 1 <= k <= n:
            raise ValueError(f"Invalid cutoff k={k} for n={n}; no silent clipping")
        zk = float(w[:min(k,m)].sum())
        dcg = float(w[:k] @ y[:k])
        values.update({f"Precision@{k}": float(h[k-1]/k),
                       f"Recall@{k}": float(h[k-1]/m), f"NDCG@{k}": dcg/zk})
        regrets.update({f"Precision@{k}": float(deficit[k-1]/k),
                        f"Recall@{k}": float(deficit[k-1]/m), f"NDCG@{k}": (zk-dcg)/zk})
    return State(y, h, values, regrets)


def score_state(labels, scores, cutoffs=(), item_ids=None):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    ids = np.arange(len(labels)) if item_ids is None else np.asarray(item_ids)
    if labels.shape != scores.shape or ids.shape != labels.shape:
        raise ValueError("Scores, labels, and IDs must have identical shapes")
    if not np.all(np.isfinite(scores)) or len(np.unique(ids)) != len(ids):
        raise ValueError("Scores must be finite and item IDs unique within a list")
    order = np.lexsort((ids, -scores))
    state = ranked_state(labels[order], cutoffs)
    state.order = order
    return state


def transition(before, after, cutoffs=(), eps=1e-12):
    if len(before.y) != len(after.y) or before.y.sum() != after.y.sum():
        raise ValueError("Transitions require a fixed list and fixed positive count")
    n, m = len(before.y), int(before.y.sum())
    ell = n-m
    h = (after.prefix-before.prefix)[:-1]
    w = discounts(n); d = w[:-1]-w[1:]; z = w[:m].sum()
    hp, hm = np.maximum(h,0), np.maximum(-h,0)
    p,q = int(hp.sum()),int(hm.sum())
    gains = {name: after.values[name]-v for name,v in before.values.items()}
    predicted = {"AUC": float(h.sum()/(m*ell)), "NDCG": float(d@h/z)}
    for k in cutoffs:
        hk = int(after.prefix[k-1]-before.prefix[k-1])
        predicted[f"Precision@{k}"] = hk/k
        predicted[f"Recall@{k}"] = hk/m
        predicted[f"NDCG@{k}"] = float((d[:k-1]@h[:k-1]+w[k-1]*hk)/w[:min(k,m)].sum())
    residual = max(abs(gains[k]-v) for k,v in predicted.items())
    result = {"n":n,"m":m,"P":p,"Q":q,"d_plus":float(d@hp/p) if p else 0.,
              "d_minus":float(d@hm/q) if q else 0.,"prefix_nonnegative":bool(np.all(h>=0)),
              "identity_error":float(residual),"auc_up_ndcg_down":gains['AUC']>eps and gains['NDCG'] < -eps,
              "opposite_sign":gains['AUC']*gains['NDCG'] < -eps*eps,
              "effective_coefficient":float(m*ell/z*(d@h)/h.sum()) if q==0 and p>0 else None}
    result.update({f"delta_{k}":float(v) for k,v in gains.items()})
    # Region boundaries are prefix indices, not numbers of swaps.
    head = min(10,n-1); mid = min(max(head,int(np.ceil(n/2))),n-1)
    for region,sl in {"head":slice(0,head),"middle":slice(head,mid),"tail":slice(mid,n-1)}.items():
        result[f"P_{region}"]=int(hp[sl].sum());result[f"Q_{region}"]=int(hm[sl].sum())
        result[f"ndcg_{region}"]=float(d[sl]@h[sl]/z)
        result[f"ndcg_positive_{region}"]=float(d[sl]@hp[sl]/z)
        result[f"ndcg_negative_{region}"]=float(d[sl]@hm[sl]/z)
    return result


def cutoff_constant(n,m,k,source,target):
    """Return (closed_form_or_None, finite): None means finite but not solved."""
    ell=n-m;w=discounts(n);z=w[:m].sum();zk=w[:min(k,m)].sum()
    a,nd,p,r,nk=names(k)
    constant_targets={p,r} if k==n else set()
    if target in constant_targets:return 0.,True
    if source in constant_targets:return float('inf'),False
    if source==target:return 1.,True
    exact={(a,nd):ell*(w[0]-w[m])/z,(nd,a):z/(m*(w[m-1]-w[-1]))}
    if k==n:
        if source==nk:source=nd
        if target==nk:target=nd
        if source==target:return 1.,True
        return float(exact[source,target]),True
    exact.update({(p,r):k/m,(r,p):m/k,(nk,p):zk/(k*w[min(k,m)-1]),
                  (nk,r):zk/(m*w[min(k,m)-1]),
                  (a,p):m*ell/(k*(abs(k-m)+1)),(a,r):ell/(abs(k-m)+1)})
    if (source,target) in exact:return float(exact[source,target]),True
    if source in [p,r] and target==nk:
        return ((k if source==p else m)*w[0]/zk,True) if k<=m else (float('inf'),False)
    if source in [p,r] and target in [a,nd]:
        if k!=m:return float('inf'),False
        return ((n-1)/ell if target==a else None),True
    if source==nk and target in [a,nd]:
        return (None,True) if k>=m else (float('inf'),False)
    if source in [a,nd]:return None,True
    raise ValueError((source,target))
