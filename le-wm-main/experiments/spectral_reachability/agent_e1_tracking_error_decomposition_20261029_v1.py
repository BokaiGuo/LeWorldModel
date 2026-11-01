"""E1 fixed 18-cell saved arithmetic; imports no producer or fullstrict."""
import argparse
from copy import deepcopy
import itertools
import math
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p2_six_target_error_decomposition_20261029_v1 as algebra
from experiments.spectral_reachability import agent_e1_six_slot_physical_summary_20261029_v1 as physical

NAME='agent_e1_tracking_error_decomposition_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
RGB='agent_e1_six_slot_rgb_20261029_v1'
ACTUAL={
 'prepare':('results/'+RGB+'_prepare_MAIN01.json','b58c5eaea94ea7ecdcb5b6c07fdb6b535643f252d705c73938c843840b3e5159'),
 'report':('results/'+RGB+'_cuda_MAIN01/report.json','7de6796f87878bb862cb2f27781412649bb99ce6310a8c7de339595d29491405'),
 'completion':('results/'+RGB+'_cuda_MAIN01/completion.json','4eb943a1e8695de124c35535d3ce065889829c4d4b1a24c0892898403878bdf4'),
 'receipt':('results/'+RGB+'_consume_MAIN01.json','86ee030e2b7a50e0412e41d9c16ae6a10c80a1b6beaf89831f04e943db2f33b5'),
 'physical_report':('results/agent_e1_six_slot_physical_20261029_v1_MAIN01/report.json','77cd42de14e1c7d75f8e7b5b4b3d9ad58fb87e94a77c209333ec8a5fc739c5ce'),
 'physical_receipt':('results/agent_e1_six_slot_physical_20261029_v1_consume_MAIN01.json','6644156f48a79a30f5118d9c56c3c3a06c3bc818e44edb55e6fe8bf4c29d38d4')}
PINS={str(Path(algebra.__file__).resolve()):'6aacaf5fb77546c6987f8d3f3de9318716c66953742ab944d4cdba8f50cf7b9f',
      str(Path(physical.__file__).resolve()):'afeeebbc8b50689ded8b09102d33e1b0a9aa9bf353e5ef373528fcbcce2bfbc1',
      str(ROOT/'experiments/spectral_reachability'/ (RGB+'.py')):'a9f2162d9281db816dfe2ebab667e68f4f2486ce45c5328cea9e0197d3b5d952'}
need=algebra.need
eq=algebra.equal


def refs():return {k:dict(path=str(ROOT/path),sha256=h) for k,(path,h) in ACTUAL.items()}


def original_mse(x,y):
    x,y=algebra.vector(x),algebra.vector(y)
    return math.fsum((float(a)-float(b))**2 for a,b in zip(x,y,strict=True))/192


def analyze(p,r,c,native_receipt,native_report,latents):
    need(c['schema']==RGB+'.consume' and c['status']=='all18_saved_RGB_positions_verified_science_blocked' and
         c['fixed_slots']==6 and c['fixed_cells']==18 and c['FLOPs'] is None and c['scientific_admission'] is False,'full RGB receipt')
    need(r['schema']==RGB+'.report' and r['completed'] is True and r['errors']==[] and r['contract']==p['contract'] and
         r['model_before']==r['model_after']==p['expected_model_hash'],'completed frozen RGB model')
    need(eq(p['imported_native_receipt'],native_receipt) and native_receipt['report_sha256']==ACTUAL['physical_report'][1], 'exact inherited native authority')
    native_summary=physical.summarize(native_receipt,native_report)
    need(eq(p['imported_controller_receipt'],c['inherited_controller_receipt']) and
         eq(native_receipt['counts'],c['inherited_native_counts']) and eq(p['bindings'],c['bindings']) and
         eq(r['resources'],c['resources']) and eq(r['counts'],c['counts']),'full inherited resources and bindings')
    need(len(p['rows'])==len(r['rows'])==len(c['rows'])==len(latents)==18,'all 18 cells')
    out=[]
    for i,(spec,row,receipt,z) in enumerate(zip(p['rows'],r['rows'],c['rows'],latents,strict=True)):
        slot,j=divmod(i,3);root=slot//3;arm=physical.ARMS[slot%3];step=physical.STEPS[j]
        need((spec['index'],spec['slot'],spec['root'],spec['arm'],spec['step'])==(i,slot,root,arm,step),'fixed original order')
        need((receipt['index'],receipt['slot'],receipt['root'],receipt['arm'],receipt['step'])==(i,slot,root,arm,step),'receipt row identity')
        mutable={'status','latent','pixels','preprocessed','modules'}
        need(eq({k:v for k,v in spec.items() if k not in mutable},{k:v for k,v in row.items() if k not in mutable}),'sealed input fields')
        nr=native_receipt['rows'][slot];plan=nr['selected_plan'];found=[x for x in nr['checkpoints'] if x['step']==step]
        need(spec['selection_digest']==plan['selection_digest'] and spec['task_id']==plan['task_id'] and
             eq(spec['commands'],plan['commands']) and eq(spec['target192'],plan['target']),'same plan target/raw/task')
        need(eq(spec['checkpoint'],found[0] if found else None) and len(found)<=1 and eq(receipt['checkpoint'],spec['checkpoint']),'actual native checkpoint join')
        pred,target,goal=(algebra.f32(spec[k]) for k in ('predicted192','target192','goal192'))
        if z is None:need(spec['checkpoint'] is None and row['latent'] is None and receipt['missing'] is True,'no invented missing data')
        else:
            algebra.vector(z);need(spec['checkpoint'] is not None and row['status']=='encoded' and receipt['missing'] is False,'actual encoded cell')
            if step==0:need(z.tobytes()==pred.tobytes(),'cp0 exact source latent')
        a=algebra.decompose(pred,z,target,goal);res=a['residual']
        expected=dict(model_tracking_Q=original_mse(pred,target),model_task_Q=original_mse(pred,goal),
            actual_tracking_Q=None if z is None else original_mse(z,target),actual_task_Q=None if z is None else original_mse(z,goal),
            prediction_error=None if z is None else original_mse(pred,z),
            whole_segmented_prediction_error=original_mse(pred,algebra.f32(spec['segmented192'])))
        need(all(receipt[k]==v for k,v in expected.items()),'independent RGB scalar arithmetic')
        if step:
            need(eq(spec['whole_context']['latent_history'][-1],spec['predicted192']) and
                 eq(spec['segmented_context']['latent_history'][-1],spec['segmented192']),'whole/segment actual saved contexts')
        cell=native_summary['rows'][slot]['cells'][j]
        out.append(dict(index=i,slot=slot,root=root,arm=arm,step=step,missing=z is None,
            prediction_source='paid raw10 whole replay' if step else 'original root latent',
            inputs_sha256={k:algebra.digest(x.tobytes()) for k,x in (('prediction',pred),('target',target),('goal',goal))}|{'actual':None if z is None else algebra.digest(z.tobytes())},
            arithmetic=a,original_scalar_recomputed=expected,
            original_minus_decomposition_model_tracking=expected['model_tracking_Q']-res['tracking_target']['model_goal_MSE'],
            whole_segmented_context_bits=receipt['whole_segmented_context_bits'],native=cell['value'],
            native_checkpoint=deepcopy(spec['checkpoint']),task_id=spec['task_id'],selection_digest=spec['selection_digest']))
    pairs=[]
    for root in (0,1):
        for left,right in itertools.combinations(range(root*3,root*3+3),2):
            for j,step in enumerate(physical.STEPS):
                a,b=out[3*left+j],out[3*right+j];sa,sb=p['rows'][3*left+j],p['rows'][3*right+j]
                need(a['task_id']==b['task_id'] and algebra.f32(sa['goal192']).tobytes()==algebra.f32(sb['goal192']).tobytes(),'common task/goal before comparison')
                na,nb=a['native'],b['native'];same_native=na is not None and nb is not None and na['goal']==nb['goal']
                pairs.append(dict(root=root,left_slot=left,right_slot=right,step=step,direction='right-minus-left',
                    common_goal_model_delta=b['arithmetic']['residual']['task_goal']['model_goal_MSE']-a['arithmetic']['residual']['task_goal']['model_goal_MSE'],
                    common_goal_actual_delta=algebra.sub(b['arithmetic']['residual']['task_goal']['actual_goal_MSE'],a['arithmetic']['residual']['task_goal']['actual_goal_MSE']),
                    native7_delta=nb['native7']-na['native7'] if same_native else None,
                    native_joint_xy_delta=nb['all_xy']-na['all_xy'] if same_native else None,
                    cross_target_tracking_delta=None))
    return dict(rows=out,pairs=pairs,fixed_slots=6,fixed_cells=18,fixed_pairs=18,winner=None,
        inherited_resources=dict(rgb=c['resources'],native=c['inherited_native_counts'],controller=c['inherited_controller_receipt']),
        known_roots=2,fresh=False,FLOPs=None,scientific_admission=False,independent_fullstrict=False,
        cross_target_tracking_ranking=False,new_calls=dict(encoder=0,predict=0,AD=0,physics=0))


def build():
    authority=refs();docs={k:algebra.read(v['path'],v['sha256']) for k,v in authority.items()}
    p,r,d,c,nr,nreport=(docs[k] for k in ('prepare','report','completion','receipt','physical_receipt','physical_report'))
    need(p['schema']==RGB+'.prepare' and r['prepare_sha256']==authority['prepare']['sha256'] and
         c['prepare_sha256']==authority['prepare']['sha256'] and c['report_sha256']==authority['report']['sha256'] and
         c['completion_sha256']==authority['completion']['sha256'] and d['schema']==RGB+'.completion' and
         d['report_sha256']==authority['report']['sha256'] and d['complete'] is True,'four RGB authority joins')
    need(Path(r['prepared']).resolve()==Path(authority['prepare']['path']) and
         r['model_before']==r['model_after']==p['expected_model_hash'],'prepare/model join')
    sources={v['path']:v['sha256'] for v in authority.values()}
    for path,h in PINS.items():need(algebra.sha(path)==h,'pinned arithmetic/source');sources[path]=h
    for path in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        sources[str(path)]=algebra.sha(path)
    folder=Path(authority['report']['path']).parent;sealpath=folder/'native_seal.json'
    need(r['artifact_sha256'].get('native_seal.json')==r['native_seal_sha256'],'sealed source inventory')
    seal=algebra.read(sealpath,r['native_seal_sha256'])
    need(eq(seal,{k:r[k] for k in ('rows','modules','counts','errors')}),'exact complete original seal')
    sources[str(sealpath)]=r['native_seal_sha256'];latents=[]
    for row in r['rows']:
        desc=row['latent'];latents.append(None if desc is None else algebra.load_latent(folder,desc,r['artifact_sha256']))
        if desc is not None:sources[str(folder/desc['file'])]=desc['sha256']
    result=analyze(p,r,c,nr,nreport,latents)
    for path,h in sources.items():need(algebra.sha(path)==h,'source unchanged after arithmetic')
    return dict(schema=NAME+'.analysis',authorities=authority,local_sources=sources,**result,
        trust_boundary='inherits MAIN full strict; no upstream fullstrict, NN, gradients or physics rerun',
        arithmetic='original f32 values; binary64 differences and fsum/192, roundoff retained')


def write(path,value):
    path=Path(path).absolute();data=algebra.encode(value)
    need(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(data)<=algebra.CAP,'exclusive bounded output')
    with path.open('xb') as f:need(f.write(data)==len(data),'complete write');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path=str(path),sha256=algebra.sha(path))


def main():
    q=argparse.ArgumentParser();q.add_argument('command',choices=('build','verify'));q.add_argument('--output',required=True);q.add_argument('--sha256');a=q.parse_args()
    if a.command=='build':result=write(a.output,build())
    else:
        value=algebra.read(a.output,a.sha256);need(eq(value,build()),'complete saved reconstruction');result=dict(verified=True,sha256=a.sha256)
    print(algebra.encode(result).decode(),end='')


if __name__=='__main__':main()
