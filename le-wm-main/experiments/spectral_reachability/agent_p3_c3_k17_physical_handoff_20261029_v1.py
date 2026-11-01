"""K17 fixed six-slot saved handoff; never dispatches a model or physics."""
import argparse
import gzip
import hashlib
import json
import math
import os
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = 'agent_p3_c3_k17_physical_handoff_20261029_v1'
RECEIPT = ROOT/'results/agent_p3_c3_compact_host_cuda_20261029_v1_consume_MAIN01.json'
RECEIPT_SHA = 'cdd61fc20ff4b819e64092f226cbe84ce3777edfc2a2c8472322a01ec3a0601f'
OUTPUT = ROOT/'results/agent_p3_c3_compact_host_cuda_20261029_v1_cuda_MAIN01'
PHYSICAL = ROOT/'results/agent_p3_h3_witness_physical_20261029_v1_MAIN01/report.json'
PHYSICAL_SHA = 'c8fa06be8eaaf6f7515e6e9550c900c71e04f2141d745699b33d1585eb21a2ce'
PHYS_RECEIPT = ROOT/'results/agent_p3_h3_witness_physical_20261029_v1_consume_MAIN01.json'
PHYS_RECEIPT_SHA = '48a1a5ecadf7043e3162ca104d8f43b0d59baf88655a076f675c9f96d697b9f7'
SLOTS = ((3,0,'fixedH1'),(4,0,'staticH3_progress'),(5,0,'adaptive'),
         (9,1,'fixedH1'),(10,1,'staticH3_progress'),(11,1,'adaptive'))

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def pinned(path, expected):
    p=Path(path); require(p.is_file() and not p.is_symlink() and sha(p)==expected,'source SHA: '+str(p))
    return json.loads(p.read_bytes())

def f32(x):
    require(type(x) in (int,float) and math.isfinite(x),'finite action')
    try: b=struct.pack('<f',x)
    except (OverflowError,struct.error) as exc: raise ValueError('float32 action') from exc
    require(struct.unpack('<f',b)[0] == x,'literal float32 action')
    return b

def rounded(x): return struct.unpack('<f',struct.pack('<f',x))[0]

def history_export(task, export, file_sha):
    b=task['binding']; t=export['task']
    require(file_sha==b['task_sha256'] and export['task_id']==b['task_id'] and
      export['blind_sha256']==b['blind_sha256'] and t['task_contract_sha256']==b['task_contract_sha256'],
      'original exported task identity')
    require(export['history_semantics']=='actual_moving_raw_between_observed0_5_10', 'actual history timeline')
    for k in ('root_context','goal','projection','action_mean','action_std'):
        require(canonical(t[k])==canonical(task[k]), 'exact saved task field '+k)
    raw=t['root_raw_action_history']; require(type(raw) is list and len(raw)==10,'raw10 history')
    normalized=[]; raw_bytes=b''
    for pair in raw:
        require(type(pair) is list and len(pair)==2,'raw history axis')
        for j,x in enumerate(pair):
            require(type(x) in (float,int) and -1<=x<=1,'history raw box')
            raw_bytes+=f32(x)
            mean=struct.unpack('<f',f32(task['action_mean'][j]))[0]
            std=struct.unpack('<f',f32(task['action_std'][j]))[0]
            require(std>0,'positive f32 standard deviation')
            normalized.append(rounded(rounded(x-mean)/std))
    expected=task['root_context']['action_history']; require(len(expected)==2 and all(len(p)==10 for p in expected),'two token history')
    require(b''.join(f32(x) for x in normalized)==b''.join(f32(x) for token in expected for x in token),
      'once normalized history byte identity')
    return dict(task_export_sha256=file_sha,history_semantics=export['history_semantics'],
      observation_times=[0,5,10],raw_history_f32_hex=raw_bytes.hex(),
      normalized_history_f32_sha256=hashlib.sha256(b''.join(f32(x) for x in normalized)).hexdigest(),
      saved_history_binding_verified=True,live_history_verified=False)

def plan_row(slot, root, mode, task, row, arm, tree):
    identity=dict(slot=slot,root=root,mode=mode,K=17)
    require(all(row.get(k)==arm.get(k)==v for k,v in identity.items()),'exact K17 identity')
    require(row['status']==arm['status']==tree['status']=='tree_complete_not_admitted'
            and arm['plan_available'] is True and tree['mode']==mode and tree['K']==17
            and tree['backend_kind']=='HOST_CUDA','completed sealed model plan required')
    require(tree['root_task_sha256']==digest(task),'exact model root task')
    plan=tree['model_plan']; require(type(plan) is dict,'model plan')
    actions=plan['actions']; require(type(actions) is list and 5<=len(actions)<=45 and len(actions)%5==0,'whole packet count')
    raw=b''
    for pair in actions:
        require(type(pair) is list and len(pair)==2,'raw [step,axis]')
        for x in pair:
            require(type(x) in (int,float) and math.isfinite(x) and -1<=x<=1,'raw box')
            raw+=f32(x)
    replay=tree['replays']
    require(len(replay)==2 and [x['mode'] for x in replay]==['whole','segmented'] and
            all(x['status']=='returned' and canonical(x['path'])==canonical(plan['path']) for x in replay),
            'actual full path replay witnesses')
    return dict(**identity,root_task_sha256=digest(task),task_binding=task['binding'],
      plan_sha256=digest(plan),full_action_steps=len(actions),raw_f32_sha256=hashlib.sha256(raw).hexdigest(),
      actions=actions,first5_raw_f32_hex=raw[:40].hex(),whole_segmented_saved_exact=True,
      native_mapping_certified=False,physical_dispatch_ready=False,physical_outcome=None)

def physical_roots(tasks, report, receipt):
    require(receipt.get('schema')=='agent_p3_h3_witness_physical_20261029_v1.consume' and
      receipt.get('status')=='h3_witness4_control20_native_recorded_not_admitted' and
      receipt.get('physical_report_sha256')==PHYSICAL_SHA and type(receipt.get('saved_PNG_verified')) is int and receipt['saved_PNG_verified']==384,
      'actual physical strict authority')
    rr=report['rows']; sr=receipt['rows']; require(len(rr)==len(sr)==4,'all original physical rows')
    out=[]
    for root in (0,1):
        chosen=[i for i in range(4) if rr[i]['root_slot']==root]
        require(len(chosen)==2,'two fixed physical rows per root')
        keys=('root_slot','task_id','root_sha256','goal','goal_pose','root_state','initial_exact')
        for i in chosen:
            require(all(canonical(rr[i][k])==canonical(sr[i][k]) for k in keys),'strict/report root fields')
            require(rr[i]['task_id']==tasks[root]['binding']['task_id'] and
              rr[i]['root_sha256']==tasks[root]['binding']['root_sha256'] and rr[i]['initial_exact'] is True,
              'model to physical task/root identity')
        require(all(canonical(rr[chosen[0]][k])==canonical(rr[chosen[1]][k]) for k in keys),'same root/goal across rows')
        out.append({k:rr[chosen[0]][k] for k in keys})
    return out

def budget():
    # Merely a proposed first-five transport with the same instrumentation as
    # the cited H3 adapter; no resource or engineering cap is approved here.
    per=dict(constructor=1,close=1,reset=2,set_state=4,goal_render=2,goal_setter=3,goal_bind=1,
      warm_public=10,commit_public=5,physics_step=154,explicit_render=51,png=51)
    return dict(scenario='PROPOSAL_ONLY warm10 + first5, off/on/off at 0..15',per_slot=per,
      six_slot_upper={k:6*v for k,v in per.items()},wall_seconds=None,output_bytes=None,
      floor_bytes=None,approved=False,storage_fit_proved=False,wall_fit_proved=False,
      assumptions='H3 adapter instrumentation and no early stop; not a new dispatch authority')

def build():
    from experiments.spectral_reachability import agent_p3_c3_compact_host_cuda_20261029_v1 as host
    rec=pinned(RECEIPT,RECEIPT_SHA); report=pinned(PHYSICAL,PHYSICAL_SHA); phys=pinned(PHYS_RECEIPT,PHYS_RECEIPT_SHA)
    require(rec.get('schema')==host.NAME+'.consume' and rec.get('status')=='saved12_common_cap_engineering_verified'
      and rec.get('denominator')==12 and rec.get('request_slots_per_arm')==144
      and rec.get('total_request_denominator')==1728,'original full strict authority')
    require(all(rec.get(k) is False for k in host.FLAGS),'no release flags')
    prep=pinned(rec['prepared'],rec['prepare_sha256']); require(len(prep['tasks'])==2,'two original tasks')
    histories=[]
    for root,task in enumerate(prep['tasks']):
        path=ROOT/f'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_{root:02}.json.gz'
        require(path.is_file() and not path.is_symlink() and sha(path)==task['binding']['task_sha256'],'export bytes')
        with gzip.open(path,'rt',encoding='utf-8') as stream: export=json.load(stream)
        histories.append(history_export(task,export,sha(path)))
    roots=physical_roots(prep['tasks'],report,phys)
    reader=host.Reader(OUTPUT,rec['manifest_sha256'],rec['completion_sha256'])
    try:
        require(sha(OUTPUT/'report.json')==rec['report_sha256'],'original report')
        saved,rows=reader.execution(); require(canonical(saved)==canonical(prep) and
          canonical(rows)==canonical(reader.report['rows']) and len(rows)==len(rec['arms'])==12,'complete saved roster')
        result=[]
        for slot,root,mode in SLOTS:
            tree=reader.result(slot,rows[slot]['result'])
            result.append(plan_row(slot,root,mode,prep['tasks'][root],rows[slot],rec['arms'][slot],tree))
    finally: reader.close()
    return dict(schema=NAME+'.analysis',denominator=6,source_request_denominator=864,
      k17_receipt_sha256=RECEIPT_SHA,k17_report_sha256=rec['report_sha256'],
      k17_prepare_sha256=rec['prepare_sha256'],physical_report_sha256=PHYSICAL_SHA,
      physical_receipt_sha256=PHYS_RECEIPT_SHA,roots=roots,histories=histories,rows=result,budget=budget(),
      missing=['approve exact committed horizon and per-slot/whole wall, storage/floor and failure contract',
        'exact live reset/warm10 RGB/state/history/goal guard before first physical action',
        'native first-action mapping and normalized model-input bit/roundtrip guard',
        'new adapter instrumentation, integer PNG/journal storage proof and independent strict consumer'],
      inherited_authority='pinned MAIN saved receipts; not fullstrict reconstruction',
      execution_authorized=False,physical_release=False,scientific_admission=False)

def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('analyze','verify')); p.add_argument('--output',required=True)
    p.add_argument('--sha256'); a=p.parse_args(); path=Path(a.output).absolute()
    require(path.resolve()==path and path.parent==ROOT/'results' and path.name.startswith(NAME),'own path')
    value=build(); raw=canonical(value); require(len(raw)<=16*1024**2,'bounded metadata')
    if a.command=='analyze':
        require(a.sha256 is None and not path.exists(),'exclusive output')
        with path.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        print(json.dumps(dict(path=str(path),sha256=sha(path))))
    else:
        require(a.sha256 is not None and pinned(path,a.sha256)==value,'whole analysis rebuild')
        print(json.dumps(dict(verified=True,sha256=a.sha256)))

if __name__=='__main__': main()
