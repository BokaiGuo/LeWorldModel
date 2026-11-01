"""Recognized saved H3 authority -> physical/cp5 partial typed projections.

Never signs a strict receipt, R release, new encoder provenance or readiness.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import io as buffer
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p3_physical_commit_preconditions_20261029_v1 as typed

NAME='agent_p3_physical_commit_authority_bridge_20261029_v1'
ROOT=typed.ROOT
PHYS='agent_p3_h3_witness_physical_20261029_v1'
HIST='agent_p3_h3_witness_encoding_20261029_v1'
CP5='agent_p3_actual_cp5_task_builder_20261029_v1'
PSTATUS='h3_witness4_control20_native_recorded_not_admitted'
HSTATUS='all16_H3_witness_RGB_verified_NOT_admitted'
PINS={
 PHYS:'f06772639f5477d9bd9bbe962a7b37ece34878226c30b0b43a0efa3a30eb7fe0',
 HIST:'c3c7f4421ffe843acc845ba9ef1bc19424ae66e4d08fa96b7a3fc63f22bb4748',
 CP5:'703f5f552c1911159b2144f72711c542ca48c408ed4e306b1fc17b019df889f1',
 typed.NAME:'690411f710667e816a1148b74f895f1a00750dcf910082f4477a06810001b7f7'}
require,sha,canonical,digest=typed.require,typed.sha,typed.canonical,typed.digest


def ref(alias,pointer):return {'$ref':alias,'pointer':pointer}
def escaped(s):return s.replace('~','~0').replace('/','~1')


def read(args):
    typed.keys(args,('physical_report','physical_receipt','history_report','history_receipt','cp5_tasks','cp5_source'))
    require(args['physical_report'] is not None and args['physical_receipt'] is not None,'real physical authority required')
    require((args['history_report'] is None)==(args['history_receipt'] is None),'history report/strict all or neither')
    require((args['cp5_tasks'] is None)==(args['cp5_source'] is None),'Nash actual output and final frozen source both required')
    docs={k:None if v is None else typed.io.load(v) for k,v in args.items() if k!='cp5_source'}
    if args['cp5_source'] is not None:
        typed.file_binding(args['cp5_source'])
        require(args['cp5_source']['path']==str(ROOT/'experiments/spectral_reachability'/(CP5+'.py')),'recognized Nash builder source path')
        require(args['cp5_source']['sha256']==PINS[CP5],'recognized frozen Nash builder bytes; no arbitrary source attestation')
    return docs


def source(name,bindings):
    p=ROOT/'experiments/spectral_reachability'/(name+'.py')
    require(bindings.get(str(p))==PINS[name] and sha(p)==PINS[name],'recognized pinned source authority '+name)
    return dict(path=str(p),sha256=PINS[name])


def attest_physical(args,r,s):
    require(r['schema']==PHYS+'.report' and s['schema']==PHYS+'.consume' and r['status']==s['status']==PSTATUS,
        'recognized physical schema/status, not caller-supplied success strings')
    require(s['physical_report_sha256']==args['physical_report']['sha256'],'physical full-strict/report SHA join')
    require(r['errors']==s['errors']==[] and r['known_roots']==s['known_roots']==2 and r['fresh_roots']==s['fresh_roots']==0,'fixed known2 clean')
    require(type(s['saved_PNG_verified']) is int and s['saved_PNG_verified']==r['counts']['explicit_render']['completed']
        and s['all_four_physically_executed'] is True,'original full physical strict PNG count/coverage')
    require(typed.same(r['rows'],s['rows']) and typed.same(r['counts'],s['counts']),'entire physical rows/counters strict join')
    require(s['bindings'].get(args['physical_report']['path'])==args['physical_report']['sha256'],'physical report in actual bindings')
    pin=source(PHYS,s['bindings'])
    for k in ('P1_admission','P3_admission','physical_release','freshness'):require(r[k] is False and s[k] is False,'no inherited admission')
    require(len(r['rows'])==4 and [(x['index'],x['root_slot'],x['arm']) for x in r['rows']]==
        [(i,i//2,('endpoints','midpoints')[i%2]) for i in range(4)],'fixed original4 order')
    ops=r['actual_operation_receipts']
    require([x['index'] for x in ops]==list(range(len(ops))) and all(x['status']=='returned' for x in ops),'complete original operation census')
    census=Counter(x['resource'] for x in ops)
    require(set(census)==set(r['counts']) and all(r['counts'][k]==dict(attempted=v,completed=v) for k,v in census.items()),'actual physical event/resource totals')
    return pin


def raw_steps(r,branch,phase,n):
    selected=[(i,e) for i,e in enumerate(r['actual_operation_receipts']) if
        e['branch']==branch and e['resource']=='public_step' and e['phase']==phase]
    require(len(selected)>=n and [e['step'] for _,e in selected]==list(range(1,len(selected)+1)),'all chronological public raw steps')
    rows=[]
    for _,e in selected[:n]:
        b=bytes.fromhex(e['raw_bits']);require(len(b)==8,'two raw float32 bits')
        a=np.frombuffer(b,dtype='<f4');require(np.isfinite(a).all() and np.all(abs(a)<=1),'actual raw box')
        rows.append(a)
    return np.stack(rows),[ref('physical_report',f'/actual_operation_receipts/{i}/raw_bits') for i,_ in selected[:n]]


def frame(f,output,absolute,plan):
    # Derived path and time are disclosed, not forged original fields.
    p=Path(f['png']);require(not p.is_absolute() and len(p.parts)==1 and p.name not in ('.','..'),'local original PNG name')
    out={k:deepcopy(f[k]) for k in ('file_sha256','pixel_sha256','state','snapshot')}
    out.update(png=str(output/p),absolute_step=absolute,plan_step=plan)
    typed.png(out);return out


def physical_fragment(r,index,output):
    row=r['rows'][index];base=f'/rows/{index}';prefix=row['prefix51']
    require(prefix['complete'] is True and len(prefix['frames'])==51 and row['initial_exact'] is True and row['warmup_returned']==10,'complete real prefix51')
    seq=[('warmup',s,k) for s in range(11) for k in range(3)]+[('plan',s,k) for s in range(6) for k in range(3)]
    require([(x['receipt']['phase'],x['receipt']['step'],x['receipt']['overlay_index']) for x in prefix['frames']]==seq,'prefix chronological off/on/off')
    for x in prefix['frames']:
        f=x['receipt']
        require(typed.same(x['pre'],x['post']) and f['condition']==('on' if f['overlay_index']==1 else 'off') and
            typed.same(f['state'],f['snapshot']['state']),'same-state renderer and native snapshot')
    checkpoints=[(j,c) for j,c in enumerate(row['checkpoints']) if c['step']==5]
    require(len(checkpoints)==1,'one real cp5 checkpoint');ci,cp=checkpoints[0]
    terminal=prefix['frames'][-3]['receipt']
    require(all(typed.same(cp[k],terminal[k]) for k in ('state','snapshot','file_sha256','pixel_sha256','png')),'cp5/source off-frame exact')
    warm,wrefs=raw_steps(r,index,'warmup',10);actual_prefix,prefs=raw_steps(r,index,'plan',5)
    require(typed.bits(actual_prefix,typed.f32(row['plan']['prefix_commands'],(5,2),box=True)),'actual prefix bits vs sealed original plan')
    task=row['plan']['source_witness']['task'];typed.core.validate_task(task)
    require(row['root_sha256']==task['binding']['root_sha256'] and row['task_id']==task['binding']['task_id'] and row['condition']=='off','original root/task/off')
    frames=[];frame_refs=[]
    for phase,step,absolute in (('warmup',5,5),('warmup',10,10),('plan',5,15)):
        j=next(j for j,x in enumerate(prefix['frames']) if (x['receipt']['phase'],x['receipt']['step'],x['receipt']['overlay_index'])==(phase,step,0))
        f=prefix['frames'][j]['receipt'];frames.append(frame(f,output,absolute,absolute-10));frame_refs.append(ref('physical_report',base+f'/prefix51/frames/{j}/receipt'))
    anchor=dict(root_sha256=row['root_sha256'],task_id=row['task_id'],condition=row['condition'],goal=row['goal'],goal_pose=row['goal_pose'],
        checkpoint=frame(cp,output,15,5),prefix51=deepcopy(prefix['frames']),warm_raw10=warm.tolist(),prefix_raw5=actual_prefix.tolist())
    projection={k:ref('physical_report',base+'/'+k) for k in ('root_sha256','task_id','condition','goal','goal_pose')}
    projection.update(prefix51=ref('physical_report',base+'/prefix51/frames'),prefix_raw5=ref('physical_report',base+'/plan/prefix_commands'),
        checkpoint=ref('physical_report',base+f'/checkpoints/{ci}'))
    return dict(physical_status='original_saved_prefix_verified_NOT_new_restore',anchor=anchor,restore_reference=deepcopy(anchor),
        frames=frames,candidate_base_task=task,source_projection=projection,frame_refs=frame_refs,
        derivations=dict(warm_raw10=dict(operation='decode_little_endian_float32_raw_bits',inputs=wrefs),
            prefix_raw5=dict(operation='decode_little_endian_float32_raw_bits',inputs=prefs),
            checkpoint_absolute_step=dict(operation='warmup_returned_plus_checkpoint_step',inputs=[ref('physical_report',base+'/warmup_returned'),ref('physical_report',base+f'/checkpoints/{ci}/step')]),
            png_path=dict(operation='report_parent_join_local_png_name',input=ref('physical_report',base+f'/checkpoints/{ci}/png'))))


def latent(output,meta):
    typed.keys(meta,('file','sha256','array_sha256','dtype','shape'))
    name=Path(meta['file']);require(len(name.parts)==1 and not name.is_absolute(),'local array artifact')
    p=output/name;typed.file_binding(dict(path=str(p),sha256=meta['sha256']))
    require(p.stat().st_size<=1024**2,'compressed latent cap')
    with gzip.open(p,'rb') as stream:body=stream.read(1024**2+1)
    require(len(body)<=1024**2,'decoded latent cap')
    data=buffer.BytesIO(body);a=np.load(data,allow_pickle=False)
    require(data.tell()==len(body) and a.shape==(192,) and a.dtype==np.float32 and np.isfinite(a).all() and
        meta['dtype']=='float32' and meta['shape']==[192] and typed.io.hashlib.sha256(a.tobytes()).hexdigest()==meta['array_sha256'],'exact original NPY latent bits')
    return a


def attest_history(args,r,s,p_r,p_s):
    require(r['schema']==HIST+'.report' and s['schema']==HIST+'.consume' and s['status']==HSTATUS and
        r['status']=='terminal_encoded_science_blocked','recognized original history authority')
    require(s['report_sha256']==args['history_report']['sha256'] and s['physical_report_sha256']==args['physical_report']['sha256']
        and s['physical_receipt_sha256']==args['physical_receipt']['sha256'],'exact encoder/physical strict ancestry')
    require(r['errors']==r['blocked']==[] and r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after'] and typed.same(r['counts'],s['counts']),'original history complete model/census')
    prepare_ref=dict(path=r['prepared'],sha256=r['prepare_sha256']);p=typed.io.load(prepare_ref)
    source(HIST,p['bindings'])
    require(all(p['bindings'].get(args[k]['path'])==args[k]['sha256'] for k in ('physical_report','physical_receipt')),'history prepare physical source bindings')
    require(len(r['rows'])==len(s['rows'])==16 and len(s['actual_context_chains'])==4,'fixed16/four context chains')
    for k in ('P1_admission','P3_admission','physical_release'):require(r[k] is False and s[k] is False,'no inherited release')
    return prepare_ref,p


def history_fragment(args,r,s,physical,index):
    j=4*index;row=r['rows'][j];consumed=s['rows'][j]
    require((row['index'],row['plan_index'],row['root_slot'],row['checkpoint'],row['arm'],row['status'])==
        (j,index,index//2,5,'endpoints','encoded'),'fixed cp5 encoder row; no outcome selection')
    require((consumed['index'],consumed['plan_index'],consumed['checkpoint'])==(j,index,5) and
        consumed['latent_missing'] is False and consumed['observed_anchor_reencode_MSE']==0,'exact actual cp5 reencode receipt')
    anchor=physical['anchor'];src=row['source'];cp=anchor['checkpoint']
    require(src['path']==cp['png'] and all(typed.same(src[k],cp[k]) for k in ('file_sha256','pixel_sha256','state','snapshot')),'encoder actual cp5 same native/RGB anchor')
    task=physical['candidate_base_task'];raw=typed.f32(row['causal_raw10'],(10,2),box=True)
    expected=np.concatenate([typed.f32(anchor['warm_raw10'],(10,2),box=True)[-5:],typed.f32(anchor['prefix_raw5'],(5,2),box=True)])
    require(typed.bits(raw,expected) and raw.tobytes().hex()==row['causal_raw10_bits'],'history actual warmtail+prefix rawbits')
    mean,std=(np.asarray(task[k],np.float32) for k in ('action_mean','action_std'))
    tokens=typed.tokens(raw,mean,std).reshape(2,10)
    require(typed.bits(tokens,typed.f32(row['normalized_action_history'],(2,10))) and typed.same(row['task_binding'],task['binding'])
        and row['model_hash']==r['model_hash_before']==task['binding']['model_sha256'],'once-normalized actual history/model/task')
    z=latent(Path(args['history_report']['path']).parent,row['latent'])
    require(typed.bits(z,typed.f32(consumed['actual192'],(192,))) and typed.bits(z,typed.f32(row['observed_anchor192'],(192,))),'saved NPY/receipt/observed cp5 latent')
    chain=s['actual_context_chains'][index]
    require((chain['plan_index'],chain['root_slot'],chain['bank'])==(index,index//2,'endpoints'),'original context-chain identity')
    entry=chain['checkpoints'][0];context=entry['context']
    require(entry['step']==5 and entry['context_missing'] is False and typed.bits(typed.f32(context['action_history'],(2,10)),tokens)
        and typed.bits(typed.f32(context['latent_history'],(3,192))[-1],z),'candidate full context vs last actual frame/history')
    return dict(history_status='last_actual_cp5_and_raw10_verified; first_two_frame_encoder_provenance_pending',
        raw10=raw.tolist(),history_tokens=tokens.tolist(),actual_cp5_latent192=z.tolist(),candidate_context=context,
        source_projection=dict(raw10=ref('history_report',f'/rows/{j}/causal_raw10'),history_tokens=ref('history_report',f'/rows/{j}/normalized_action_history'),
            candidate_context=ref('history_receipt',f'/actual_context_chains/{index}/checkpoints/0/context'),
            actual_cp5_latent192=ref('history_receipt',f'/rows/{j}/actual192')),
        latent_artifact=dict(path=str(Path(args['history_report']['path']).parent/row['latent']['file']),sha256=row['latent']['sha256']),
        all_three_frames_actual_verified=False)


def join_cp5(args,result,roots):
    if result is None:return False
    require(result['schema']==CP5+'.tasks' and result['status']=='known2_actual_cp5_saved_join_ready_NOT_admitted'
        and len(result['tasks'])==len(result['provenance'])==2,'recognized actual cp5 task-builder ABI, not R planner receipt')
    require(result['bindings'].get(args['cp5_source']['path'])==args['cp5_source']['sha256'],'MAIN supplied frozen Nash source bound to saved output')
    a=result['authorities'];require(set(a)=={'prepared','report','receipt'},'Nash actual upstream authority set')
    prepared,report,receipt=(typed.io.load(a[k]) for k in ('prepared','report','receipt'))
    require(prepared['schema']=='agent_p3_shared_panel_20261029_v1.prepare' and report['schema']=='agent_p3_shared_panel_20261029_v1.report'
        and receipt['schema']=='agent_p3_shared_panel_20261029_v1.consume' and receipt['status']=='saved_shared_panel_verified_NOT_admitted'
        and receipt['complete'] is True and receipt['report_sha256']==a['report']['sha256'] and
        receipt['prepare_sha256']==report['prepare_sha256']==a['prepared']['sha256'],'actual original full shared-panel authority joins')
    require(receipt['prepared']==report['prepared']==a['prepared']['path'],'original shared prepare path join')
    require(all(result['bindings'].get(v['path'])==v['sha256'] for v in a.values()),'all original authorities bound by Nash output')
    for path,h in result['bindings'].items():typed.file_binding(dict(path=path,sha256=h))
    for root,item in enumerate(roots):
        task=result['tasks'][root];pro=result['provenance'][root];typed.core.validate_task(task)
        require(pro['source_slot']==root and pro['actual_cp5_absolute_steps']==[5,10,15] and pro['actual_cp5_control_steps']==[-5,0,5]
            and pro['local_task_sha256']==typed.core.digest(task),'Nash exact root/timing/local identity')
        require(typed.same(pro['original_task'],item['candidate_base_task']) and
            typed.bits(typed.f32(pro['prefix5'],(5,2),box=True),typed.f32(item['anchor']['prefix_raw5'],(5,2),box=True)),
            'same original task and actual physical prefix')
        history=item['history'];require(history is not None,'actual RGB strict required to bind Nash cp5 join')
        require(typed.same(task['root_context'],history['candidate_context']) and
            typed.bits(typed.f32(pro['raw10'],(10,2),box=True),typed.f32(history['raw10'],(10,2),box=True)),
            'Nash full cp5 context/raw bits vs independent RGB history')
        require(all(typed.same(task[k],item['candidate_base_task'][k]) for k in ('goal','projection','action_mean','action_std'))
            and task['binding']['root_sha256']==item['anchor']['root_sha256'],'rebased task keeps goal/Q/normalizer/root')
        require(len(pro['branch_witnesses'])==4 and {w['episode_id'] for w in pro['branch_witnesses']}==
            {f's{root}_{mode}' for mode in ('fixedH1_endpoints','fixedH1_midpoints','adaptive_endpoints','adaptive_midpoints')},'all4 common cp5 branches, no outcome-selected root')
        for w in pro['branch_witnesses']:
            ns=[n for n in report['nodes'] if n['episode_id']==w['episode_id'] and n['boundary']==5]
            audits=[x for x in receipt['audited'] if x['episode_id']==w['episode_id'] and x['boundary']==5]
            require(len(ns)==len(audits)==1 and audits[0]['node_verified'] is True,'actual unique audited cp5 node')
            request=ns[0]['request']
            require(w['request_sha256']==typed.core.digest(request) and typed.same(w['saved_task_binding'],request['task']['binding'])
                and typed.same(w['observation'],request['observation']) and typed.same(request['task']['root_context'],task['root_context']),
                'saved branch witness matches authenticated original request, not arbitrary identity strings')
        item['actual_cp5_task']=task
        item['actual_cp5_task_source']=ref('cp5_tasks',f'/tasks/{root}')
        item['history']['history_status']='saved_all_actual_cp5_context_joined_to_Nash_and_original_shared_strict'
        item['history']['all_three_frames_actual_verified']=True
    return True


def build(args):
    require(sha(typed.__file__)==PINS[typed.NAME],'frozen typed helper')
    d=read(args);r,s=d['physical_report'],d['physical_receipt'];pin=attest_physical(args,r,s)
    history_prepare=None
    if d['history_report'] is not None:
        history_prepare,_=attest_history(args,d['history_report'],d['history_receipt'],r,s)
    by_root=[];output=Path(args['physical_report']['path']).parent
    for index in (0,2):
        physical=physical_fragment(r,index,output)
        physical['history']=None if history_prepare is None else history_fragment(args,d['history_report'],d['history_receipt'],physical,index)
        by_root.append(physical)
    cp5_joined=join_cp5(args,d['cp5_tasks'],by_root)
    # Pure source-bound fields usable now. Derived path/time/warm bytes are
    # disclosed separately: never inject a fake source report or strict receipt.
    contract=typed.template();contract['documents'].update({k:v for k,v in deepcopy(args).items() if k!='cp5_source'})
    contract['authorities']['physical']=dict(report='physical_report',receipt='physical_receipt',report_schema=PHYS+'.report',
        receipt_schema=PHYS+'.consume',receipt_status=PSTATUS,report_sha_pointer='/physical_report_sha256',
        source_bindings=[dict(file=pin,sha_ref=ref('physical_receipt','/bindings/'+escaped(pin['path'])))])
    if history_prepare is not None:
        inventory=d['history_report']['model_inventory_before'];require(inventory,'actual encoder installed source inventory')
        f=inventory[0];typed.file_binding(dict(path=f['path'],sha256=f['sha256']))
        contract['authorities']['history']=dict(report='history_report',receipt='history_receipt',report_schema=HIST+'.report',receipt_schema=HIST+'.consume',
            receipt_status=HSTATUS,report_sha_pointer='/report_sha256',source_bindings=[dict(file=dict(path=f['path'],sha256=f['sha256']),
                sha_ref=ref('history_report','/model_inventory_before/0/sha256'))])
    for row in contract['slots']:
        p=by_root[row['root']]['source_projection']
        row['projection']=dict(anchor=deepcopy(p),restore_reference=deepcopy(p))
    blocked=typed.build(contract)
    require(blocked['status']=='BLOCKED' and blocked['ready'] is False,'no planner authority; must remain blocked')
    for refvalue in args.values():
        if refvalue is not None:require(sha(refvalue['path'])==refvalue['sha256'],'all input bytes stable')
    return dict(schema=NAME+'.bridge',status='BLOCKED',inputs=args,source_sha256=sha(__file__),
        recognized_sources=PINS,history_prepare=history_prepare,denominator=dict(slots=4,known_roots=2,fresh_roots=0),
        source_bank_selection='endpoints index0/root0 and index2/root1 fixed before outcomes; same anchor shared by future fixed/adaptive',
        roots=by_root,typed_contract_partial=contract,typed_blocked_manifest=blocked,
        missing=['planner.actual_R_report_and_full_strict','planner.release_model_IO_and_resource_ledger',
            *([] if cp5_joined else ['history.Nash_actual_cp5_tasks_and_frozen_source_join']),
            *([] if history_prepare else ['history.actual_encoder_report_and_full_strict']),
            'typed_projection.derived_time_path_warmraw_and_rebased_task_identity_adapter','runtime.new_environment_cp5_pre_step6_guard'],
        strict_receipt_issued=False,all_three_frames_actual_verified=cp5_joined,**typed.FLAGS)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    names=('physical_report','physical_receipt','history_report','history_receipt','cp5_tasks','cp5_source')
    for kind in names:
        p.add_argument('--'+kind.replace('_','-'),required=kind.startswith('physical'))
        p.add_argument('--'+kind.replace('_','-')+'-sha',required=kind.startswith('physical'))
    p.add_argument('--output',required=True);a=p.parse_args(argv);args={}
    for kind in names:
        path,h=getattr(a,kind),getattr(a,kind+'_sha');require((path is None)==(h is None),'path/SHA pair')
        args[kind]=None if path is None else dict(path=str(Path(path).absolute()),sha256=h)
    out=build(args);target=Path(a.output).absolute();body=canonical(out)
    require(target.parent==ROOT/'results' and target.name.startswith(NAME) and target.resolve()==target and not target.exists(),'exclusive own bridge output')
    require(len(body)<=16*1024**2,'bounded bridge')
    with target.open('xb') as f:require(f.write(body)==len(body),'whole write');f.flush();typed.os.fsync(f.fileno())
    print(canonical(dict(path=str(target),sha256=sha(target),status=out['status'])).decode())


if __name__=='__main__':main()
