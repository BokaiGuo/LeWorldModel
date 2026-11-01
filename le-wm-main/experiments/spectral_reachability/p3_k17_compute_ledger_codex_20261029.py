"""Recount original K17 planning hook events; saved data only, no model execution."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import struct

from experiments.spectral_reachability import agent_p3_c3_compact_host_cuda_20261029_v1 as host

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / 'results/agent_p3_c3_compact_host_cuda_20261029_v1_consume_MAIN01.json'
RECEIPT_SHA = 'cdd61fc20ff4b819e64092f226cbe84ce3777edfc2a2c8472322a01ec3a0601f'
HANDOFF = ROOT / 'results/agent_p3_c3_k17_physical_handoff_20261029_v1_MAIN01.json'
HANDOFF_SHA = '3b2caf555f390426234b34a96d8bd84670933da5364eba4a1f22df9a424a6df9'
PHYSICAL = ROOT / 'results/p3_k17_host_same_time_pairs_20261029_CODEX01.json'
PHYSICAL_SHA = '0a8fa2c7a822aa702f5a5ed1bbbe6d168dfd24deddc33feaa251da39ba09debb'
SOURCE = ROOT / 'results/agent_p3_c3_compact_host_cuda_20261029_v1_cuda_MAIN01'
ORDER = [(3,0,'fixedH1'),(4,0,'staticH3_progress'),(5,0,'adaptive'),(9,1,'fixedH1'),(10,1,'staticH3_progress'),(11,1,'adaptive')]
SCALARS = {'jacobian':'jacobian_API','grad':'grad_API','raw_backward':'raw_backward_callbacks',
           'model_svd':'model_SVD_calls','array_svd':'array_SVD_calls','solve':'solve_calls',
           'signal_compile':'signal_compile_calls','program_compile':'program_compile_calls',
           'replay':'replay_calls','metric':'metric_calls','schedule':'schedule_decisions'}
AXES = ('predict_API','predict_samples','predict_context_tokens','action_encoder_calls',
        'action_encoder_samples','action_encoder_context_tokens','predictor_calls','predictor_samples',
        'predictor_context_tokens','pred_proj_calls','pred_proj_rows','encoder_calls','encoder_samples',
        'projector_calls','projector_samples','jacobian_API','VJP_directions','grad_API',
        'raw_backward_callbacks','model_SVD_calls','array_SVD_calls','solve_calls','signal_compile_calls',
        'program_compile_calls','replay_calls','metric_calls','schedule_decisions')


def need(value, message):
    if not value: raise ValueError(message)


def digest(data): return hashlib.sha256(data).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def pinned(path, expected):
    raw = path.read_bytes(); need(digest(raw) == expected, 'anchor ' + str(path)); return json.loads(raw)


def delta(event):
    r, shape, directions = event['resource'], event['shape'], event['directions']
    out = dict.fromkeys(AXES, 0)
    if r in SCALARS:
        need(shape is None and directions is None, 'scalar metadata'); out[SCALARS[r]] = 1
    elif r == 'vjp':
        need(shape is None and type(directions) is int and directions > 0, 'VJP directions'); out['VJP_directions'] = directions
    else:
        need(directions is None and type(shape) is list and all(type(n) is int and n > 0 for n in shape), 'shape metadata')
        if r in ('predict','action_encoder','predictor'):
            need(len(shape)==3 and shape[1]==3, 'batch/context shape')
            if r == 'predict': out.update(predict_API=1,predict_samples=shape[0],predict_context_tokens=shape[0]*shape[1])
            else: out.update({r+'_calls':1,r+'_samples':shape[0],r+'_context_tokens':shape[0]*shape[1]})
        elif r == 'pred_proj':
            need(len(shape)==2, 'projection shape'); out.update(pred_proj_calls=1,pred_proj_rows=shape[0])
        elif r in ('encoder','projector'):
            need(len(shape)>=2, 'encoder shape'); out.update({r+'_calls':1,r+'_samples':shape[0]})
        else: raise ValueError('unknown resource ' + r)
    return out


def recount(receipt, mode):
    caps = receipt['caps']; need(set(caps)==set(AXES) and all(type(v) is int and v>=0 for v in caps.values()), 'caps')
    attempted = dict.fromkeys(AXES, 0); completed = dict.fromkeys(AXES, 0); phases = {}; wall = Counter()
    events = receipt['events']; need(type(events) is list, 'events')
    for i,e in enumerate(events):
        need(e['index']==i and e['scope']['mode']==mode and e['scope']['K']==17, 'event identity')
        change = delta(e); need(e['delta']==change, 'event delta')
        phase = e['scope']['phase']; need(type(phase) is str and phase, 'phase')
        p = phases.setdefault(phase, {'attempted':dict.fromkeys(AXES,0),'completed':dict.fromkeys(AXES,0),'events':0})
        p['events'] += 1
        status = e['status']
        if status == 'blocked_before_dispatch':
            need(any(attempted[k]+change[k]>caps[k] for k in AXES) and e['ended_ns'] is None, 'blocked event'); continue
        need(status in ('returned','failed'), 'event disposition')
        start,end=e['started_ns'],e['ended_ns']
        need(type(start) is int and type(end) is int and 0 <= start <= end <= receipt['wall_ns'], 'event wall')
        wall[e['resource']] += end-start
        for k in AXES:
            attempted[k]+=change[k]; p['attempted'][k]+=change[k]
            if status=='returned': completed[k]+=change[k]; p['completed'][k]+=change[k]
        need(all(attempted[k]<=caps[k] for k in AXES), 'cap exceeded')
    need(attempted==receipt['attempted'] and completed==receipt['completed'], 'recount mismatch')
    need(dict(wall)==receipt['inclusive_operation_wall_ns'], 'inclusive wall mismatch')
    need(all(receipt[k] is None for k in ('FLOPs','cuda_ns','backward_engine_calls')) and receipt['physics_steps']==0, 'unknown costs')
    return {'caps':caps,'attempted':attempted,'completed':completed,'phases':phases,
            'event_count':len(events),'wall_ns':receipt['wall_ns'],'inclusive_operation_wall_ns':dict(wall)}


def build():
    receipt=pinned(RECEIPT,RECEIPT_SHA); handoff=pinned(HANDOFF,HANDOFF_SHA); physical=pinned(PHYSICAL,PHYSICAL_SHA)
    need(receipt['status']=='saved12_common_cap_engineering_verified' and len(receipt['arms'])==12, 'planning receipt')
    need(handoff['k17_receipt_sha256']==RECEIPT_SHA and len(handoff['rows'])==6, 'handoff binding')
    reader=host.Reader(SOURCE,receipt['manifest_sha256'],receipt['completion_sha256'])
    rows=[]
    try:
        need(host.codec.file_sha(SOURCE/'report.json')==receipt['report_sha256'], 'report hash')
        for i,(slot,root,mode) in enumerate(ORDER):
            arm=receipt['arms'][slot]; rr=reader.report['rows'][slot]; h=handoff['rows'][i]; p=physical['branches'][i]
            need(all((x['slot'],x['root'],x['mode'])==(slot,root,mode) for x in (arm,rr,h,p)), 'fixed identity')
            tree=reader.result(slot,rr['result']); need(tree['mode']==mode and tree['K']==17 and tree['status']=='tree_complete_not_admitted', 'tree identity')
            plan=tree['model_plan']; need(digest(canonical(plan))==h['plan_sha256'] and tree['root_task_sha256']==h['root_task_sha256'], 'plan binding')
            bits=b''.join(struct.pack('<f',x) for pair in plan['actions'][:5] for x in pair).hex()
            need(bits==h['first5_raw_f32_hex']==''.join(s['action_bits'] for s in p['steps']), 'physical first5 action binding')
            ledger=tree['resource_receipt']; result=recount(ledger,mode); saved=arm['resources']
            need(all(result[k]==saved[k] for k in ('caps','attempted','completed')), 'strict receipt vectors')
            need(len(ledger['events'])==saved['events_count'] and host.codec.semantic(ledger['events'])==saved['events_semantic_sha256'], 'strict event binding')
            rows.append(dict(slot=slot,root=root,mode=mode,plan_sha256=h['plan_sha256'],plan_action_steps=len(plan['actions']),physical_first5_bits=bits,ledger=result,physical_resources=p['resources']))
    finally: reader.close()
    pairs=[]
    for a,b in itertools.combinations(rows,2):
        if a['root']!=b['root']: continue
        pairs.append(dict(root=a['root'],a=a['mode'],b=b['mode'],slots=[a['slot'],b['slot']],
             equal_caps=a['ledger']['caps']==b['ledger']['caps'],
             completed_b_minus_a={k:b['ledger']['completed'][k]-a['ledger']['completed'][k] for k in AXES},
             whole_tree_wall_ns_b_minus_a=b['ledger']['wall_ns']-a['ledger']['wall_ns']))
    return dict(schema='p3_k17_compute_ledger_codex.v1',anchors=dict(planning_receipt=RECEIPT_SHA,handoff=HANDOFF_SHA,physical_pairs=PHYSICAL_SHA,manifest=receipt['manifest_sha256'],completion=receipt['completion_sha256'],report=receipt['report_sha256']),axes=list(AXES),rows=rows,pairs=pairs,
      limits=['saved whole-tree planning hooks, not physical first5 incremental cost','nested and correlated axes must not be summed','hook census cannot exclude hidden calls','FLOPs CUDA time backward engine calls unavailable','no equal total compute or fairness certification','two known roots, no independent generalization','physical provenance remains unverified'])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('output'); parser.add_argument('--verify',action='store_true'); parser.add_argument('--sha256')
    args=parser.parse_args(); data=canonical(build())+b'\n'; path=Path(args.output)
    if args.verify:
        need(args.sha256 is not None and digest(path.read_bytes())==args.sha256 and path.read_bytes()==data, 'output mismatch')
    else:
        need(args.sha256 is None, 'sha only for verify')
        with path.open('xb') as f: need(f.write(data)==len(data),'short write'); f.flush(); os.fsync(f.fileno())
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    print(digest(data))


if __name__=='__main__': main()
