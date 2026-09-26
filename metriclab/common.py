import csv
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return 'inf' if value>0 else '-inf' if value<0 else None
    if isinstance(value,Path):return str(value)
    return value

def write_json(path,value):
    Path(path).write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def write_csv(path,rows):
    rows=list(rows)
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with open(path,'w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows([{k:clean(v) for k,v in row.items()} for row in rows])

def start_run(command,config,out=None):
    if out is None:
        stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        out=ROOT/'results'/f'{stamp}-{command}'
    out=Path(out).resolve()
    # Never merge experimental output into an existing nonempty directory.
    if out.exists() and any(out.iterdir()):raise ValueError(f'Output directory is not empty: {out}')
    out.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'metriclab').glob('*.py'))}
    hashes['run.py']=sha256(ROOT/'run.py')
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in [ROOT/'run.py',ROOT/'requirements.txt',ROOT/'README.md']+list((ROOT/'metriclab').glob('*.py'))+list((ROOT/'configs').glob('*.json'))+list((ROOT/'tests').glob('*.py')):
            archive.write(file,str(file.relative_to(ROOT)))
    write_json(out/'environment.json',{name:importlib.metadata.version(name) for name in ['numpy','matplotlib']})
    write_json(out/'manifest.json',{'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'config':config,'source_sha256':hashes,
        'status':'running','regret':'realized fixed-list oracle gap; not population Bayes regret'})
    write_json(out/'config.json',config)
    return out

def finish(out,summary):
    write_json(out/'summary.json',summary)
    p=out/'manifest.json';meta=json.loads(p.read_text(encoding='utf-8'))
    meta.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(p,meta)
    print(f'Completed: {out}',flush=True)

def figure_module():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    return plt

def savefig(fig,out,name):
    fig.tight_layout()
    fig.savefig(out/f'{name}.png',bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
