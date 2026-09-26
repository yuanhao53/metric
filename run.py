"""Reproducible CPU experiments for fixed-list metric transfer."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
import traceback
from pathlib import Path
from metriclab.common import ROOT,start_run,finish,write_json
from metriclab.theory import enumerate_lists,scaling
from metriclab.dynamics import controlled
from metriclab.training import train_all

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['enumerate','scaling','dynamics','train','core','all'])
    p.add_argument('--config',default=str(ROOT/'configs'/'smoke.json'))
    p.add_argument('--out',default=None)
    args=p.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    out=start_run(args.command,config,args.out)
    jobs={'enumerate':enumerate_lists,'scaling':scaling,'dynamics':controlled,'train':train_all}
    selected=list(jobs) if args.command=='all' else list(jobs)[:3] if args.command=='core' else [args.command]
    try:
        summary={}
        for name in selected:
            folder=out/name;folder.mkdir()
            print(f'Running {name}: {folder}',flush=True)
            summary[name]=jobs[name](config,folder)
            write_json(folder/'summary.json',summary[name])
        finish(out,summary)
    except Exception:
        error=traceback.format_exc();(out/'error.txt').write_text(error,encoding='utf-8')
        meta=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        meta['status']='failed';write_json(out/'manifest.json',meta)
        raise

if __name__=='__main__':main()
