"""Persist the exact independent CEM saved-consumer result, no neural runtime."""
import argparse
import os
from pathlib import Path
import shutil
from experiments.spectral_reachability import agent_baselines_resume_20261027_next as source


def run(directory,report_sha,output):
    destination=Path(output).resolve()
    source.require(not destination.exists() and destination.parent.is_dir(),'exclusive actual consumer output')
    result=source.consume(directory,report_sha)
    payload=source.canonical(result)
    source.require(len(payload)<4*1024**2 and shutil.disk_usage(destination.parent).free>=source.FLOOR+len(payload),'bounded saved release/floor')
    with destination.open('xb') as f:
        source.require(f.write(payload)==len(payload),'short consumer result')
        f.flush();os.fsync(f.fileno())
    fd=os.open(destination.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(status=result['status'],plans=len(result['plans']),sha256=source.sha(destination),
                model_Q=[p['model_Q'] for p in result['plans']],new_neural_calls=0,new_physics_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('input','report-sha','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(source.canonical(run(a.input,a.report_sha,a.output)).decode())
