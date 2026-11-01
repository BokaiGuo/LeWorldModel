"""Extract paid K17 replay inputs and connect a non-dispatching live guard."""
import argparse
import gzip
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p3_c3_k17_live_guard_20261029_v1 as guard
from experiments.spectral_reachability import agent_p3_c3_k17_physical_handoff_20261029_v1 as handoff

NAME='agent_p3_c3_k17_replay_guard_bridge_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
HANDOFF=ROOT/'results/agent_p3_c3_k17_physical_handoff_20261029_v1_MAIN01.json'
HANDOFF_SHA='3b2caf555f390426234b34a96d8bd84670933da5364eba4a1f22df9a424a6df9'
PHYS_PREP=ROOT/'results/agent_p3_h3_witness_physical_20261029_v1_prepare_MAIN01.json'
PHYS_PREP_SHA='0e31243a683a7383066ea3b495840610a4b1f2df811fcae318764d3b5acbac4e'
SOURCE_PINS={Path(guard.__file__).resolve():'5267f9618cbf13f94e076c5861d6449c4d25316d2ec274c78411c5b7c8730a1d',
 Path(handoff.__file__).resolve():'eacaf68a67337b68c6d4c462a8524165a7856585c8ac340ba3d186f5ec7ee259'}

def need(value,message):
    if not value: raise ValueError(message)
def canon(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(canon(value)).hexdigest()
def exact(a,b): return canon(a)==canon(b)

def get_array(unblob, event, key, shape):
    need(key in event['arrays'],'missing actual native array '+key)
    value=unblob(event['arrays'][key]); need(value.dtype.str=='<f4' and value.shape==shape,'actual f32 shape '+key)
    return value

def extract(task,tree,identity,capsule,audit,unblob):
    """From the validated original slot, bind BOTH final replay invocations.

    capsule and audit are supplied by the original immutable HOST reader in
    production; fixture callbacks alone cannot confer source authority.
    """
    slot,root,mode=identity['slot'],identity['root'],identity['mode']
    need((slot,root,mode) in handoff.SLOTS and identity.get('K')==17 and tree['mode']==mode and tree['K']==17 and
      tree['backend_kind']=='HOST_CUDA' and tree['status']=='tree_complete_not_admitted' and
      tree['root_task_sha256']==digest(task),'fixed task/slot/tree identity')
    plan=tree['model_plan']; need(type(plan) is dict,'completed original model plan')
    actions=plan['actions']; raw=guard.f32(actions,(len(actions),2),box=True)
    need(5<=len(actions)<=45 and len(actions)%5==0,'original whole packet shape')
    rep=tree['replays']; need(len(rep)==2 and [x['mode'] for x in rep]==['whole','segmented'] and
      all(x['status']=='returned' and exact(x['path'],plan['path']) for x in rep),'both actual replay paths')
    events=tree['resource_receipt']['events']; calls=tree['native_calls']; out=[]
    for name in ('whole','segmented'):
        selected=[e for e in events if e['resource']=='replay' and e['scope'].get('phase')=='verification'
                  and e['scope'].get('replay_mode')==name]
        need(len(selected)==1,'unique paid final verification ledger entry')
        e=selected[0]; need(e['status']=='returned' and e['scope'].get('K')==17 and e['scope'].get('mode')==mode,
          'paid return and exact scope')
        matches=[c for c in calls if c['outer_index']==e['index']]
        need(len(matches)==1,'one native call for ledger replay')
        record=matches[0]; call=record['transcript']; cid=record['call_id']
        need(type(cid) is int and 0<=cid<len(calls) and calls[cid] is record,'original call order')
        need(call[0]['kind']=='native_call_attempt' and call[0]['method']=='replay' and
          exact(call[0]['task'],task) and call[-1]['kind']=='native_call_return' and
          exact(call[-1]['result'],plan['path']),'same task and saved full path')
        req=call[0]['request']; need(exact(req['parent_context'],task['root_context']) and
          exact(req['actions'],actions) and req['segmented'] is (name=='segmented'),'actual full root/action/replay request')
        pointer=capsule(slot,cid,call); need(type(pointer) is dict and type(pointer.get('sha256')) is str,'authenticated native capsule')
        need(audit(call) is True,'original actual native IO / trajectory audit')
        packet=[x for x in call if x.get('kind')=='program_packet' and x.get('phase')==name and x.get('step')==0]
        ae=[x for x in call if x.get('kind')=='io_input' and x.get('phase')==name and x.get('step')==0 and 'action_encoder:0' in x.get('arrays',{})]
        need(len(packet)==len(ae)==1,'unique actual step0 packet and AE input')
        p=packet[0]; i=ae[0]
        token=get_array(unblob,p,'token',(10,)); rawbuf=get_array(unblob,p,'raw',(5,2))
        history=get_array(unblob,p,'history',(1,2,10)); latent=get_array(unblob,p,'latent',(1,3,192))
        mean=get_array(unblob,p,'mean',(2,)); std=get_array(unblob,p,'std',(2,))
        outgoing=get_array(unblob,p,'outgoing',(1,3,10)); actual=get_array(unblob,i,'action_encoder:0',(1,3,10))
        need(rawbuf.tobytes()==raw[:5].tobytes() and
          mean.tobytes()==guard.f32(task['action_mean'],(2,)).tobytes() and
          std.tobytes()==guard.f32(task['action_std'],(2,)).tobytes() and
          history.tobytes()==guard.f32(task['root_context']['action_history'],(2,10)).tobytes() and
          latent.tobytes()==guard.f32(task['root_context']['latent_history'],(3,192)).tobytes(),
          'step0 raw/stats/full original history exact bits')
        expected=np.concatenate((history,token.reshape(1,1,10)),axis=1)
        need(outgoing.tobytes()==actual.tobytes()==expected.tobytes(),'actual paid AE input includes exact original token')
        action=guard.action_guard(raw[:5],mean,std,token)
        out.append(dict(mode=name,call_id=cid,outer_index=e['index'],capsule=pointer,
          token_f32_hex=token.tobytes().hex(),token_sha256=hashlib.sha256(token.tobytes()).hexdigest(),
          action_encoder_input_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
          native_packet_event_sha256=digest(p),native_ae_event_sha256=digest(i),action_guard=action))
    need(out[0]['token_f32_hex']==out[1]['token_f32_hex'] and
      out[0]['action_encoder_input_sha256']==out[1]['action_encoder_input_sha256'],
      'whole/segmented paid step0 input bitexact')
    return dict(**identity,root_task_sha256=digest(task),model_plan_sha256=digest(plan),
      first5_raw_f32_hex=raw[:5].tobytes().hex(),replays=out,physical_dispatch_ready=False)

def build():
    from experiments.spectral_reachability import agent_p3_c3_compact_host_cuda_20261029_v1 as host
    for path,expected in SOURCE_PINS.items(): need(path.is_file() and not path.is_symlink() and sha(path)==expected,'frozen source SHA')
    saved=handoff.pinned(HANDOFF,HANDOFF_SHA); rec=handoff.pinned(handoff.RECEIPT,handoff.RECEIPT_SHA)
    need(saved['schema']==handoff.NAME+'.analysis' and saved['denominator']==6 and
      saved['k17_receipt_sha256']==handoff.RECEIPT_SHA and saved['execution_authorized'] is False,
      'saved handoff / strict authority')
    need(rec['schema']==host.NAME+'.consume' and rec['status']=='saved12_common_cap_engineering_verified' and
      rec['denominator']==12 and rec['total_request_denominator']==1728 and all(rec.get(k) is False for k in host.FLAGS),
      'complete MAIN original receipt')
    prepared=handoff.pinned(rec['prepared'],rec['prepare_sha256'])
    reader=host.Reader(handoff.OUTPUT,rec['manifest_sha256'],rec['completion_sha256'])
    try:
        need(sha(handoff.OUTPUT/'report.json')==rec['report_sha256']==saved['k17_report_sha256'],'original report binding')
        p,rows=reader.execution(); need(exact(p,prepared) and exact(rows,reader.report['rows']) and len(rows)==len(rec['arms'])==12,'all saved rows')
        result=[]
        for identity,old in zip(handoff.SLOTS,saved['rows'],strict=True):
            slot,root,mode=identity; row=rows[slot]; arm=rec['arms'][slot]
            want=dict(slot=slot,root=root,mode=mode,K=17)
            need(all(row[k]==arm[k]==old[k]==v for k,v in want.items()) and arm['plan_available'] is True,
              'saved strict/handoff all-six identity')
            tree=reader.result(slot,row['result']); task=p['tasks'][root]
            item=extract(task,tree,want,reader.capsule,host.old.audit_native,host.old.signal.unblob)
            need(item['model_plan_sha256']==old['plan_sha256'] and item['root_task_sha256']==old['root_task_sha256'] and
              item['first5_raw_f32_hex']==old['first5_raw_f32_hex'],'exact handoff plan/raw')
            result.append(item)
    finally: reader.close()
    return dict(schema=NAME+'.analysis',denominator=6,rows=result,handoff_sha256=HANDOFF_SHA,
      strict_receipt_sha256=handoff.RECEIPT_SHA,report_sha256=rec['report_sha256'],
      physical_budget_authority=None,live_source_authority=None,execution_authorized=False,physical_release=False,
      scientific_admission=False,missing=['approved complete fixed six-slot native physical budget and stop contract',
        'source-bound actual reset/warm10/goal reference and genuine runtime capture adapter',
        'bounded durable physical writer, complete remaining-action dispatch and saved strict consumer'],
      trust='inherits MAIN complete receipt; native selected replay audit; no fullstrict reconstruction')

def frame_projection(frame):
    """Remove only the branch-specific relative filename, never PNG hashes."""
    result=deepcopy(frame); name=result['receipt'].pop('png')
    need(type(name) is str and Path(name).name==name and name.endswith('.png'),'relative PNG name')
    return result

def source_reference(root,task,phys_prepared,report,receipt,export,export_sha):
    need(root in (0,1),'fixed root')
    handoff.history_export(task,export,export_sha)
    need(receipt['physical_report_sha256']==handoff.PHYSICAL_SHA and
      receipt['status']=='h3_witness4_control20_native_recorded_not_admitted' and
      type(receipt['saved_PNG_verified']) is int and receipt['saved_PNG_verified']==384,'physical saved receipt')
    rows=report['rows']; saved=receipt['rows']; need(len(rows)==len(saved)==4,'all physical rows')
    ix=[i for i in range(4) if rows[i]['root_slot']==root]; need(len(ix)==2,'two physical row identities')
    frames=[]
    for i in ix:
        r=rows[i]; s=saved[i];
        for k in ('root_slot','task_id','root_sha256','root_state','goal','goal_pose','initial_exact'):
            need(exact(r[k],s[k]),'physical receipt/report field '+k)
        need(r['initial_exact'] is True and r['task_id']==task['binding']['task_id'] and
          r['root_sha256']==task['binding']['root_sha256'],'source original task/root')
        candidates=[x for x in r['prefix51']['frames'] if x['receipt']['phase']=='plan' and
                    x['receipt']['step']==0 and x['receipt']['overlay_index']==0]
        need(len(candidates)==1,'one original plan0 frame')
        rawframe=candidates[0]; name=rawframe['receipt']['png']
        need(type(name) is str and Path(name).name==name,'original relative PNG')
        path=handoff.PHYSICAL.parent/name
        need(path.is_file() and not path.is_symlink() and sha(path)==rawframe['receipt']['file_sha256'],'actual original PNG bytes')
        frames.append(frame_projection(rawframe))
    need(exact(frames[0],frames[1]),'two fixed original branches same root goal frame ignoring filename')
    root0=phys_prepared['source_roots'][root]; slot=phys_prepared['source_slots'][root]
    need(root0['root_slot']==root and root0['root_sha256']==task['binding']['root_sha256'] and
      exact(root0['root_state'],rows[ix[0]]['root_state']) and exact(root0['goal'],rows[ix[0]]['goal']) and
      exact(root0['goal_pose'],rows[ix[0]]['goal_pose']),'original physical prepare/root/goal')
    trace=phys_prepared['warm_traces'][root]; guard.ref_map(trace)
    bound=frames[0]['pre']; need(exact(bound,frames[0]['post']) and exact(bound['goal_state'],root0['goal']) and
      exact(bound['goal_pose'],root0['goal_pose']) and exact(bound['renderer'],root0['renderer']), 'exact bound root goal renderer')
    return dict(task_sha256=digest(task),task_id=task['binding']['task_id'],root_sha256=root0['root_sha256'],
      reset_identity=slot,warm_trace=trace,warm_raw=export['task']['root_raw_action_history'],
      root_state=root0['root_state'],goal_state=root0['goal'],bound_goal=bound,goal_frame=frames[0])

def check_live_only(item,task,reference,live):
    """Connect the certified token to frozen live guard; intentionally never arms dispatch."""
    identity={k:item[k] for k in ('slot','root','mode','K')}
    need(item['root_task_sha256']==digest(task) and len(item['replays'])==2,'same certified task and replay pair')
    a,b=item['replays']; need(a['token_f32_hex']==b['token_f32_hex'],'two paid replay tokens exact')
    raw=np.frombuffer(bytes.fromhex(item['first5_raw_f32_hex']),dtype='<f4').reshape(5,2)
    token=np.frombuffer(bytes.fromhex(a['token_f32_hex']),dtype='<f4').reshape(10)
    live=deepcopy(live); live['goal_frame']=frame_projection(live['goal_frame'])
    history=guard.history_guard(identity=identity,task=task,reference=reference,live=live)
    action=guard.action_guard(raw,task['action_mean'],task['action_std'],token)
    return dict(history=history,action=action,physical_dispatch_ready=False,
      reason='live fixture or caller input is not an independently authenticated native source and no approved physical budget')

def check_live_from_sources(slot,live):
    """Saved authorities plus caller capture; no physical/GPU, no dispatch."""
    result=build(); rows=[r for r in result['rows'] if r['slot']==slot]
    need(len(rows)==1,'fixed K17 slot'); item=rows[0]; root=item['root']
    rec=handoff.pinned(handoff.RECEIPT,handoff.RECEIPT_SHA)
    task=handoff.pinned(rec['prepared'],rec['prepare_sha256'])['tasks'][root]
    pp=handoff.pinned(PHYS_PREP,PHYS_PREP_SHA)
    report=handoff.pinned(handoff.PHYSICAL,handoff.PHYSICAL_SHA)
    receipt=handoff.pinned(handoff.PHYS_RECEIPT,handoff.PHYS_RECEIPT_SHA)
    path=ROOT/f'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_{root:02}.json.gz'
    need(path.is_file() and not path.is_symlink() and sha(path)==task['binding']['task_sha256'],'original gzip bytes')
    with gzip.open(path,'rt',encoding='utf-8') as stream: export=json.load(stream)
    reference=source_reference(root,task,pp,report,receipt,export,sha(path))
    return check_live_only(item,task,reference,live)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('analyze','verify'));p.add_argument('--output',required=True);p.add_argument('--sha256')
    a=p.parse_args(); path=Path(a.output).absolute()
    need(path.resolve()==path and path.parent==ROOT/'results' and path.name.startswith(NAME),'own output path')
    value=build(); raw=canon(value); need(len(raw)<=8*1024**2,'metadata cap')
    if a.command=='analyze':
        need(a.sha256 is None and not path.exists(),'exclusive output')
        with path.open('xb') as f: need(f.write(raw)==len(raw),'complete write');f.flush();os.fsync(f.fileno())
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        print(json.dumps(dict(path=str(path),sha256=sha(path))))
    else:
        need(a.sha256 is not None and handoff.pinned(path,a.sha256)==value,'complete recomputation')
        print(json.dumps(dict(verified=True,sha256=a.sha256)))

if __name__=='__main__': main()
