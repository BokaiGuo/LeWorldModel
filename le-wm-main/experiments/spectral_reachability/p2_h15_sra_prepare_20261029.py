"""Saved-only H15 no-composition spectral shooting preparation; never runs a model."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability.pusht_full_packet_compiler_20261027 import bounded_ridge

ROOT = Path(__file__).resolve().parents[2]
NAME = 'p2_h15_sra_prepare_20261029'
PREPARE = ROOT/'results/agent_p2_intersection_physical_20261029_v2_prepare_MAIN01.json'
PREPARE_SHA = '619a3202a9e14c6d164375a46ee8e51e7e9fe217678e6fb6f3c2725e5219341c'
RECEIPT = ROOT/'results/agent_p2_intersection_physical_20261029_v2_consume_MAIN01.json'
RECEIPT_SHA = 'e96462847db675dc0b0c5376067df3654edd40805d6e374ffa3f8f1d5224e6c4'
CAP = .75*np.sqrt(15)
RADII = (.25*np.sqrt(15), CAP)
ORDER = ((0,'joint'),(0,'composition'),(0,'grasp'),(1,'grasp'),(1,'composition'),(1,'joint'))


def need(ok, msg):
    if not ok: raise ValueError(msg)


def encoded(x):
    return (json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def read(path, digest):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size < 128*1024**2, 'bounded regular file')
    b = path.read_bytes()
    need(hashlib.sha256(b).hexdigest() == digest, 'pinned input SHA')
    def pairs(items):
        result = {}
        for k,v in items:
            need(k not in result, 'duplicate key'); result[k] = v
        return result
    def bad(_): raise ValueError('nonfinite JSON')
    return json.loads(b, object_pairs_hook=pairs, parse_constant=bad)


def finite(x, shape):
    a = np.asarray(x, dtype=np.float64)
    need(a.shape == shape and np.isfinite(a).all(), 'finite array shape')
    return a


def spectral_bank(jacobian, nominal, goal):
    """Pure H15 adaptation of 40-candidate H2 shooting; no rollout or model."""
    j = finite(jacobian, (192,30)); z = finite(nominal, (192,)); g = finite(goal, (192,))
    ridge = bounded_ridge(j, g-z, damping=.002, cap=CAP)
    _, singular, right = np.linalg.svd(j, full_matrices=False)
    for row in right:
        if row[np.argmax(np.abs(row))] < 0: row *= -1
    vectors = [np.zeros(30)] + [s*np.asarray(ridge['params']) for s in (.25,.5,1.)]
    labels = ['zero','dls_.25','dls_.5','dls_1']
    for mode in range(9):
        for ri, radius in enumerate(RADII):
            for sign in (-1,1):
                vectors.append(sign*radius*right[mode]); labels.append(f'mode_{mode}_radius_{ri}_sign_{sign}')
    vectors = np.stack(vectors)
    need(vectors.shape == (40,30) and np.max(np.linalg.norm(vectors,axis=1)) <= CAP*(1+1e-12), 'joint ball')
    return dict(labels=labels, logits=vectors.tolist(), raw15=np.tanh(vectors).astype(np.float32).reshape(40,15,2).tolist(),
        singular_values=singular.tolist(), right_vectors=right.tolist(), ridge=ridge)


def assemble(prepared, receipt):
    need(prepared['schema']=='agent_p2_intersection_physical_20261029_v2.prepare', 'prepare schema')
    need(receipt['schema']=='agent_p2_intersection_physical_20261029_v2.consume' and
         receipt['status']=='intersection6_native_recorded_not_admitted' and not receipt['errors'], 'physical receipt')
    need(receipt['bindings'][str(PREPARE)] == PREPARE_SHA, 'receipt prepare binding')
    plans, rows = prepared['plans'], receipt['rows']
    need(len(plans)==len(rows)==6 and len(prepared['source_roots'])==2, 'complete fixed cohort')
    for i,(plan,row) in enumerate(zip(plans,rows)):
        root,arm = ORDER[i]
        need((plan['index'],plan['root_slot'],plan['arm']) == (i,root,arm) ==
             (row['index'],row['root_slot'],row['arm']), 'fixed order')
        need(plan['task_id']==row['task_id'] and plan['task_sha256']==row['plan']['task_sha256'] and
             plan['raw_steps']==len(plan['raw15'])==15, 'plan task/horizon')
        need(plan['native_root']==prepared['source_roots'][root] and
             row['root_state']==plan['native_root']['root_state'] and row['goal']==plan['native_root']['goal'], 'native root/goal')
    cases=[]
    for root in (0,1):
        group=[p for p in plans if p['root_slot']==root]
        first=group[0]
        for p in group[1:]:
            need(all(p[k]==first[k] for k in ('task_id','task_sha256','root_context','goal','q','warm_raw10','native_root')), 'cross-arm task/history/goal')
        finite(first['goal'],(192,)); finite(first['warm_raw10'],(10,2))
        cases.append({k:first[k] for k in ('root_slot','task_id','task_file','task_sha256','root_context','goal','q','warm_raw10','native_root')})
    return dict(schema=NAME+'.prepare', inputs={'physical_prepare_sha256':PREPARE_SHA,'physical_receipt_sha256':RECEIPT_SHA},
        cases=cases, fixed_roots=2, known_roots_only=True,
        design=dict(arm='joint_H3_single_spectral_bank',horizon_packets=3,raw_steps=15,action_shape=[15,2],
            jacobian_shape=[192,30],metric='identity192 terminal mean squared error',damping=.002,
            cap=float(CAP),radii=list(map(float,RADII)),candidate_count=40,selection='minimum finite full-horizon Q, first tie',
            failure='missing or failed candidate invalidates slot; no retry, padding or substitution',
            candidate_predict_samples_planned=120,compilation_predict_samples_planned=6,replay_predict_samples_planned=6,
            estimated_total_predict_samples=132,counts_are_not_measurements=True),
        existing_control='intersection joint H15 is already same-task root-compiled uncomposed iterative control; this is a different one-bank variant',
        source_time='retrospective on already observed outcomes; no prospective or independent confirmation',
        normalization='inherit original H15 task and action normalizer; implementation and bytes must be revalidated by future runner',
        ready_to_execute=False,blockers=['H15 CUDA runner and model-call instrumentation not implemented',
            'H15 physical adapter and independent saved consumer not implemented',
            'runtime closure, budget and resource accounting need independent audit and authorization'],
        new_neural_calls=0,new_physical_steps=0,fair_total_budget_comparison=False,scientific_admission=False)


def build():
    p=read(PREPARE, PREPARE_SHA); r=read(RECEIPT, RECEIPT_SHA)
    out=assemble(p,r)
    for case in out['cases']:
        path=Path(case['task_file'])
        need(path.is_file() and not path.is_symlink() and path.stat().st_size < 16*1024**2, 'bounded task file')
        data=path.read_bytes()
        need(hashlib.sha256(data).hexdigest()==case['task_sha256'], 'task file SHA')
        with gzip.open(path,'rb') as f: raw=f.read(16*1024**2+1)
        need(len(raw)<=16*1024**2, 'bounded decoded task')
        task=json.loads(raw)
        need(task['task_id']==case['task_id'] and task['task']['goal']==case['goal'] and
             task['task']['root_context']==case['root_context'], 'task context and goal match')
        t=task['task']; mean=finite(t['action_mean'],(2,)); std=finite(t['action_std'],(2,))
        need(np.all(std>0), 'positive normalizer std')
        case['action_normalizer']=dict(mean=mean.tolist(),std=std.tolist())
        case['model_sha256']=t['model_sha256']; case['model_config_sha256']=t['model_config_sha256']
        need(hashlib.sha256(path.read_bytes()).hexdigest()==case['task_sha256'], 'task stable after read')
    read(PREPARE, PREPARE_SHA); read(RECEIPT, RECEIPT_SHA)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('prepare','verify'))
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--sha256')
    a=ap.parse_args(); need(a.output.name.startswith(NAME+'_'), 'exclusive output name')
    value=encoded(build()); h=hashlib.sha256(value).hexdigest()
    if a.command=='prepare':
        need(a.sha256 is None, 'no prepare SHA argument')
        with a.output.open('xb') as f: f.write(value); f.flush(); os.fsync(f.fileno())
    else:
        need(a.sha256==h and a.output.read_bytes()==value, 'whole rebuild and external SHA')
    print(json.dumps(dict(output=str(a.output),sha256=h,verified=a.command=='verify',ready_to_execute=False)))


if __name__=='__main__': main()
