"""MovieLens observed-rating protocol, not full-catalog implicit-feedback evaluation."""
import io
import json
import urllib.request
import urllib.error
import ssl
import os
import subprocess
import zipfile
from pathlib import Path
import numpy as np
from .common import ROOT,sha256,write_json,write_csv

URL='https://files.grouplens.org/datasets/movielens/ml-1m.zip'


def download_movielens():
    directory=ROOT/'data'/'raw';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'ml-1m.zip'
    if not path.exists():
        print(f'Downloading public dataset: {URL}',flush=True)
        temp=directory/'ml-1m.zip.part'
        request=urllib.request.Request(URL,headers={'User-Agent':'metric-ICLR27-research/1.0'})
        try:
            with urllib.request.urlopen(request,timeout=120) as response,open(temp,'wb') as f:
                while chunk:=response.read(1024*1024):f.write(chunk)
        except urllib.error.URLError as error:
            if os.name!='nt' or not isinstance(error.reason,ssl.SSLCertVerificationError):raise
            # Use the Windows system trust store; never disable TLS verification.
            print('Python certificate chain unavailable; retrying with Windows system trust.',flush=True)
            env=os.environ.copy();env['METRIC_DOWNLOAD_URL']=URL;env['METRIC_DOWNLOAD_PATH']=str(temp)
            subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
                "$ErrorActionPreference='Stop'; Invoke-WebRequest -UseBasicParsing -Uri $env:METRIC_DOWNLOAD_URL -OutFile $env:METRIC_DOWNLOAD_PATH"],env=env,check=True,timeout=180)
        # Validate content before publishing the local cache; no arbitrary extraction paths.
        with zipfile.ZipFile(temp) as archive:
            archive.getinfo('ml-1m/ratings.dat')
            if archive.testzip() is not None:raise ValueError('Corrupt MovieLens archive')
        temp.replace(path)
    with zipfile.ZipFile(path) as archive:
        raw=archive.read('ml-1m/ratings.dat')
        readme=archive.read('ml-1m/README')
    (directory/'MovieLens-README.txt').write_bytes(readme)
    write_json(directory/'download.json',{'url':URL,'sha256':sha256(path),'bytes':path.stat().st_size,
        'source':'https://grouplens.org/datasets/movielens/1m/',
        'license':'See MovieLens-README.txt; dataset is not redistributed with code'})
    return path,raw


def prepare(config,out):
    path,raw=download_movielens();cfg=config['training']
    # Column order: original user ID, original item ID, rating, timestamp.
    rows=np.loadtxt(io.StringIO(raw.decode('ascii').replace('::',' ')),dtype=np.int64)
    user_ids=np.unique(rows[:,0]);rng=np.random.default_rng(config['seed'])
    if cfg.get('max_users'):
        selected=np.sort(rng.choice(user_ids,size=min(cfg['max_users'],len(user_ids)),replace=False))
        rows=rows[np.isin(rows[:,0],selected)]
    user_ids=np.unique(rows[:,0]);item_ids=np.unique(rows[:,1])
    users=np.searchsorted(user_ids,rows[:,0]);items=np.searchsorted(item_ids,rows[:,1])
    labels=(rows[:,2]>=cfg['positive_rating']).astype(np.int64)
    order=np.lexsort((rows[:,1],rows[:,3],users))
    users,items,labels=users[order],items[order],labels[order]
    original=rows[order]
    split=np.empty(len(users),dtype=np.int64)
    boundaries=np.r_[0,np.flatnonzero(np.diff(users))+1,len(users)]
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        n=hi-lo;a=lo+int(n*cfg['train_fraction']);b=a+int(n*cfg['validation_fraction'])
        split[lo:a]=0;split[a:b]=1;split[b:hi]=2
    eligible=[];cohort=[];maxk=max(cfg['cutoffs']);groups={'validation':{},'test':{}}
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        u=int(users[lo]);stats={};valid=True
        for sid,name in [(1,'validation'),(2,'test')]:
            ix=np.arange(lo,hi)[split[lo:hi]==sid];n=len(ix);m=int(labels[ix].sum())
            stats.update({name+'_n':n,name+'_m':m})
            ok=(n>=maxk and cfg['min_eval_positives']<=m<n)
            valid &= ok
            groups[name][u]=ix
        cohort.append({'user_id':int(user_ids[u]),'included':valid,**stats})
        if valid:eligible.append(u)
    if not eligible:raise ValueError('No eligible evaluation users; inspect label/split/cutoff settings')
    groups={name:{u:ix for u,ix in mapping.items() if u in set(eligible)} for name,mapping in groups.items()}
    # BPR draws a positive and an explicitly rated negative for the same training user.
    train=np.flatnonzero(split==0);train_groups={}
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        u=int(users[lo]);ix=np.arange(lo,hi)[split[lo:hi]==0]
        p=items[ix][labels[ix]==1];neg=items[ix][labels[ix]==0]
        if len(p) and len(neg):train_groups[u]=(p,neg)
    if not train_groups:raise ValueError('BPR needs training users with both rated classes')
    np.savez_compressed(out/'fixed_data_split.npz',users=users,items=items,labels=labels,split=split,
        original_user_ids=user_ids,original_item_ids=item_ids,timestamps=original[:,3],ratings=original[:,2],eligible_users=eligible)
    write_csv(out/'cohort.csv',cohort)
    metadata={'archive_sha256':sha256(path),'source_url':URL,'ratings':len(users),'users':len(user_ids),'items':len(item_ids),
        'train_ratings':len(train),'eligible_eval_users':len(eligible),'bpr_eligible_train_users':len(train_groups),
        'split':'per-user chronological; timestamp ties use item ID; NOT a global time split',
        'label_rule':f'observed rating >= {cfg["positive_rating"]} positive, lower observed ratings negative',
        'candidate_pool':'held-out observed ratings only, fixed across all checkpoints and runs; unrated items are not labeled negative',
        'cohort_rule':f'both validation/test have n >= {maxk}, m >= {cfg["min_eval_positives"]}, and at least one negative',
        'scope':'conditional evaluation on held-out rated items, not unbiased full-catalog preference evaluation',
        'split_counts':{name:int(np.sum(split==sid)) for sid,name in enumerate(['train','validation','test'])}}
    write_json(out/'data_protocol.json',metadata)
    return {'users':users,'items':items,'labels':labels,'train':train,'groups':groups,'train_groups':train_groups,
            'num_users':len(user_ids),'num_items':len(item_ids),'item_ids':item_ids,'user_ids':user_ids,'metadata':metadata}
