"""Saved-only projection of a previously authenticated six-target comparison."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
NAME='agent_p2_six_target_resource_panel_20261029_v1'
REFS={
 'comparison':('agent_p2_six_target_prediction_comparison_20261029_v1_MAIN01.json','eb4e9acc5be07d4c9381d15a932acc14de2454d4db5aecb72f17a9b65fa447b7'),
 'receipt':('agent_p2_six_target_controller_cuda_20261029_v1_consume_MAIN01.json','592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103'),
 'report':('agent_p2_six_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json','b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934')}
ARMS=('prior_FIRST','program_first_postcondition','arbitrary')


def need(ok,message):
    if not ok:raise ValueError(message)


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    p=Path(path);need(p.is_file() and not p.is_symlink(),'regular file')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def refs():return {k:dict(path=str(ROOT/'results'/p),sha256=h) for k,(p,h) in REFS.items()}


def read(ref):
    p=Path(ref['path']);need(p.stat().st_size<=128*1024**2 and sha(p)==ref['sha256'],'bounded input SHA')
    return json.loads(p.read_bytes())


def assemble(comparison,receipt,report,authority):
    need(comparison['authorities']['receipt']==authority['receipt'] and comparison['authorities']['report']==authority['report'],
         'exact inherited authority')
    need(comparison['full_strict_receipt']==receipt and comparison['resource_accounting27']==receipt['accounting27']
         and receipt['accounting27']==report['accounting27'] and receipt['side_axes']==report['side_axes'],
         'entire receipt and resource join')
    need(receipt['status']=='all6_new_target_controller_saved_verified' and
         receipt['denominator']==dict(total=6,returned=6,missing=0) and
         comparison['denominator']==dict(checkpoints_per_slot=4,pairs=24,roots=2,slots=6,total_rows=24),
         'fixed full denominator')
    need(len(comparison['rows'])==24 and len(report['rows'])==6 and len(receipt['local_checks'])==6,
         'all rows retained')
    actual=receipt['accounting27']['actual'];need(len(actual)==27 and actual['FLOPs'] is None,'27 axes and unknown FLOPs')
    panel=[]
    for i in range(6):
        root=i//3;arm=ARMS[i%3];r=report['rows'][i];c=receipt['local_checks'][i]
        need((r['slot'],r['root'],r['arm'])==(i,root,arm) and (c['slot'],c['root'],c['arm'])==(i,root,arm),
             'fixed root and arm order')
        checkpoints=[]
        for j,cp in enumerate((0,5,10,15)):
            row=comparison['rows'][4*i+j]
            need((row['slot'],row['root'],row['arm'],row['cp_public_step'])==(i,root,arm,cp),'fixed checkpoint order')
            need(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('tracking_target_mse64','task_goal_mse64')),
                 'finite saved MSE')
            checkpoints.append({k:row[k] for k in ('cp_public_step','latent_sha256','latent_source',
                'tracking_target_sha256','original_task_goal_sha256','tracking_target_mse64','task_goal_mse64','source_time')})
        need(r['status']=='returned' and c['local_check']['all_mean_replays_verified'] is True,'returned and replay certified')
        panel.append(dict(slot=i,root=root,arm=arm,checkpoints=checkpoints,axes=r['axes'],debits=r['debits'],
            capture=r['capture'],wall_seconds=r['wall_seconds'],input_files=r['input_files'],
            trace=r['trace'],arrays=r['arrays'],local_check=c['local_check']))
    return dict(schema=NAME+'.analysis',authority=authority,manifest=comparison['authorities']['manifest'],
        goal_prepare=comparison['goal_prepare'],goal_sources=comparison['goal_sources'],slots=panel,
        accounting27=receipt['accounting27'],side_axes=receipt['side_axes'],
        authority_scope='inherits existing full strict and comparison; no reauthentication or independent gradient',
        prediction_only=True,known_roots_only=True,source_time_prior=None,equal_generation_budget=False,
        equal_FLOPs=False,physical_benefit_claim=False,scientific_admission=False,winner=None)


def build():
    a=refs();out=assemble(*(read(a[k]) for k in ('comparison','receipt','report')),a)
    need(all(sha(v['path'])==v['sha256'] for v in a.values()),'sources unchanged after analysis');return out


def write(path,value):
    p=Path(path).absolute();b=canonical(value);need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<16*1024**2,'exclusive output')
    with p.open('xb') as f:need(f.write(b)==len(b),'full write');f.flush();os.fsync(f.fileno())
    return dict(path=str(p),sha256=sha(p))


def main():
    q=argparse.ArgumentParser();q.add_argument('command',choices=('build','verify'));q.add_argument('--output',required=True);q.add_argument('--sha256')
    a=q.parse_args()
    if a.command=='build':v=write(a.output,build())
    else:
        need(a.sha256 and sha(a.output)==a.sha256,'external output SHA')
        need(json.loads(Path(a.output).read_bytes())==build(),'whole deterministic rebuild');v={'verified':True,'sha256':a.sha256}
    print(json.dumps(v,sort_keys=True))


if __name__=='__main__':main()
