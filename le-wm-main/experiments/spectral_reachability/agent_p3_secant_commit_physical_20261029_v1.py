"""Own-schema secant actual-cp5 FIRST5 native execution, MAIN only.

No NN, reselection, old-schema projection or scientific admission.  A durable
cp5 guard precedes control6 dispatch; H3 lookahead is never dispatched here.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_observed_suffix_physical_20261028_v1 as primitives
from experiments.spectral_reachability import agent_p3_physical_commit_authority_bridge_20261029_v1 as bridge

NAME='agent_p3_secant_commit_physical_20261029_v1'
PRODUCER='agent_p3_secant_delivery_20261029_v1'
AUTH='MAIN_P3_SECANT_COMMIT5_PHYSICS_ONLY'
RECORDED='secant_commit5_native_recorded_NOT_admitted'
ROOT=bridge.ROOT
native=primitives.mixed
source,moving,paired=native.source,native.moving,native.paired
require,sha,raw,read=native.require,native.sha,native.raw,native.read
# Physical own schema retains spaced JSON; producer task/provenance use compact.
digest=native.core.digest
producer_digest=bridge.typed.core.digest
ORDER=[(0,1),(0,3),(1,1),(1,3)]
CAP,FINAL,FLOOR,WALL,META=64*1024**2,8*1024**2,200*1024**2,900,128*1024**2
PROTOCOL=ROOT/'innerdocs/agent_p3_secant_commit_physical_protocol_20261029_v1.md'
PROTOCOL_SHA='9c49b00afb37008c0a33b248d31e0258d368b54ec459d81ad8db98b345e3b049'
DELIVERY_PROTOCOL_SHA='500bfc08e0b542f365c6912dfa1ec6333276035a33ac596ec3f21bf3f1b3d065'
BRIDGE_SHA='fd136726fc67eac13e91b3b21514eb8a2049d398b2e86fee0dadfdca588b6647'
ARGUMENTS={'delivery_output','delivery_report_sha','delivery_receipt','delivery_receipt_sha',
    'delivery_prepared','delivery_prepare_sha','delivery_source_sha','delivery_completion_sha','bridge','bridge_sha'}
FLAGS=dict(P1_admission=False,P3_admission=False,full_C3_demonstrated=False,scientific_admission=False,
    physical_release=False,execution_authorized=False,freshness=False,equal_total_compute=False,
    equal_FLOPs=False,training=False,independent_confirmation=False)
METADATA_CAP=17*1024**2
DEFAULT_ARGUMENTS=dict(
    delivery_output=str(ROOT/'results/agent_p3_secant_delivery_20261029_v1_cuda_MAIN01'),
    delivery_report_sha='7630a2751129fd7c28a9e708494c26ebe160b1cf2656bc480084838fd7f5d9fc',
    delivery_completion_sha='b346de1fca8777fc8734dcfce68ef24d92156309b0ff062a636229c166a3ad2b',
    delivery_receipt=str(ROOT/'results/agent_p3_secant_delivery_20261029_v1_consume_MAIN01.json'),
    delivery_receipt_sha='6f583507a9e5ce6d0f859032344fc303527c250b5633fa8d3482ded3615e4fb6',
    delivery_prepared=str(ROOT/'results/agent_p3_secant_delivery_20261029_v1_prepare_MAIN01.json'),
    delivery_prepare_sha='ad289da249cb21190ee4b59748c2944c0866cea8c24be0be0a9af37e1ec7a15b',
    delivery_source_sha='7b3449b27ec68d7f923ac0af93f186c651ccb97231894152cbfa9a01649f57d2',
    bridge=str(ROOT/'results/agent_p3_physical_commit_authority_bridge_20261029_v1_MAIN01.json'),
    bridge_sha='9c47743431166ddb2564bf5dea5d5335e794d98ec0873a8b90417e632eefc97d')


def shape_census():
    # RGB8, one filter byte/scanline, zlib compressBound, plus PNG chunk/header
    # allowance. Store also enforces this bound on ACTUAL encoded PNG bytes.
    scan=224*(224*3+1)
    png=scan+(scan>>12)+(scan>>14)+(scan>>25)+13+1024
    count=4*3*(11+11)
    working=count*png+METADATA_CAP
    return dict(groups=4,PNG_count=count,PNG_per_group=66,scan_bytes=scan,
        PNG_file_upper_bytes=png,PNG_total_upper_bytes=count*png,
        metadata_upper_bytes=METADATA_CAP,working_upper_bytes=working,
        final_reserve_bytes=FINAL,total_upper_bytes=working+FINAL,output_cap_bytes=CAP,
        prefix_frames_per_group=51,prefix_persisted_copies=2,
        operation_receipt_persisted_copies=3,observation_event_copies=1,
        row_persisted_copies=2,NPY_count=0)


# Actual producer FLAGS, distinct from this physical schema's own FLAGS.
# Missing keys and legacy fake C3 aliases are rejected, never defaulted false.
UPSTREAM_FLAGS=('P1_admission','P3_admission','equal_FLOPs','equal_cap_tree',
    'execution_authorized','freshness','full_C3_demonstrated',
    'full_policy_executed','full_tree_executed','original_R_equivalence',
    'physical_release','policy_tree_executed','scientific_admission')


def upstream_flags(value,label):
    require(type(value) is dict and 'C3' not in value and
        all(k in value and value[k] is False for k in UPSTREAM_FLAGS),
        label+': all actual producer flags present and strictly false; full_C3_demonstrated, not C3')


def contract():
    c=deepcopy(native.allmode.contract())
    c.update(schema=NAME+'.contract',order=[list(x) for x in ORDER],condition='off',
        arm='secant_H1_or_H3_first5',producer=PRODUCER,prefix=5,commit=5,horizon=10,
        checkpoints=[0,5,10],raw_authority='original prefix5 + literal delivery.commit_raw5; NO lookahead dispatch',
        no_witness='retain slot; no environment or dispatch; no fallback',
        cp5_guard='durable accepted guard BEFORE control6 intent; full51 + raw history + model input identity',
        inheritance=False,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        wall_seconds=WALL,prepare_cap_bytes=META,protocol_sha256=PROTOCOL_SHA,
        model_input='frozen original goal/Q/normalizer, actual abs5/10/15 history, no new encoder',
        shape_census=shape_census(),
        interpretation='known2 candidate witness only; H3 model endpoint NOT observed here',**FLAGS)
    return c


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def f32(value,shape):return bridge.typed.f32(value,shape,box=True)
def bytehash(value):return hashlib.sha256(np.asarray(value,np.dtype('<f4')).tobytes()).hexdigest()


def validate_delivery(delivery,task,pro,h):
    require(type(delivery['candidate_index']) is int and delivery['eligible_order'] and
        delivery['candidate_index']==delivery['eligible_order'][0] and delivery['selection_key'][1]==delivery['candidate_index'],
        'single selected candidate, no fallback')
    require(delivery['schema']==PRODUCER+'.delivery' and delivery['model_delivery_verified'] is True,
        'actual own R model delivery, not generic ready strings')
    require(bridge.typed.same(delivery['task'],task) and bridge.typed.same(delivery['base_task'],pro['original_task'])
        and delivery['task_sha256']==producer_digest(task) and delivery['base_task_sha256']==producer_digest(pro['original_task'])
        and delivery['provenance_sha256']==producer_digest(pro),'exact base/rebased task and provenance producer hashes')
    edge=f32(delivery['raw_edge'],(5*h,2));commit=f32(delivery['commit_raw5'],(5,2))
    look=np.empty((0,2),np.float32) if h==1 else f32(delivery['lookahead_raw'],(10,2))
    require((h!=1 or delivery['lookahead_raw']==[]) and edge[:5].tobytes()==commit.tobytes()
        and edge[5:].tobytes()==look.tobytes() and all(delivery[k]==bytehash(a) for k,a in
        (('raw_edge_float32_sha256',edge),('commit_float32_sha256',commit),('lookahead_float32_sha256',look))),
        'whole edge, commit and lookahead literal bits; no clamp/scale')
    require(delivery['control_start']==5 and delivery['commit_control_steps']==[6,7,8,9,10]
        and delivery['prediction_end_control']==5+5*h and delivery['prediction_end_absolute']==15+5*h
        and delivery['suffix_in_plan'] is False,'only first5 commit, diagnostic suffix not execution')
    require([v['mode'] for v in delivery['verification']]==['whole','segmented']
        and all(v['status']=='returned' and bridge.typed.same(v['path'],delivery['path']) for v in delivery['verification'])
        and [v['segments'] for v in delivery['verification']]==[[5*h],([5] if h==1 else [5,10])],
        'independent whole and segmented full edge witness')
    bridge.typed.core.checked_path(task,task['root_context'],delivery['raw_edge'],delivery['path'])
    upstream_flags(delivery,'model delivery never physical permission')
    return commit


def assemble(arguments,compiled,consumed,b,data):
    """Pure actual-schema seam AFTER MAIN rebuilds the complete saved authorities."""
    require(compiled['schema']==PRODUCER+'.prepare' and consumed['schema']==PRODUCER+'.consume'
        and consumed['status']=='all4_secant_model_deliveries_verified_NOT_physical'
        and consumed['report_sha256']==arguments['delivery_report_sha']
        and consumed['prepare_sha256']==arguments['delivery_prepare_sha']
        and consumed['prepared']==arguments['delivery_prepared'] and consumed['completion_sha256']==arguments['delivery_completion_sha'] and consumed['groups']==4 and consumed['candidate_denominator']==16,
        'actual external R report/prepare/fullstrict authority')
    require(compiled['rules']['protocol_sha256']==consumed['protocol_sha256']==DELIVERY_PROTOCOL_SHA
        and len(compiled['tasks'])==len(compiled['provenance'])==2 and len(consumed['arms'])==4,
        'fixed protocol and all4 independent slots')
    require(b['schema']==bridge.NAME+'.bridge' and b['status']=='BLOCKED' and b['all_three_frames_actual_verified'] is True
        and b['typed_contract_partial']['authorities']['planner'] is None and len(b['roots'])==2,
        'actual three-frame bridge, NOT a physical release certificate')
    upstream_flags(compiled,'actual R prepare')
    upstream_flags(compiled['rules'],'actual R rules')
    upstream_flags(consumed,'actual R consume')
    require(compiled['rules']['root_context_role']==consumed['root_context_role']==
        'actual_observed_abs5_10_15_control_minus5_0_5' and compiled['cp5_tasks']==consumed['cp5_tasks'],
        'actual cp5 role and source task descriptor, not substituted control0')
    plans=[];refs=[];meta=native.allmode.root_metadata(data)
    for i,((root,h),arm) in enumerate(zip(ORDER,consumed['arms'],strict=True)):
        task=compiled['tasks'][root];pro=compiled['provenance'][root];fragment=b['roots'][root]
        bridge.typed.core.validate_task(task);bridge.typed.core.validate_task(pro['original_task'])
        require(bridge.typed.same(task,fragment['actual_cp5_task']) and bridge.typed.same(pro['original_task'],fragment['candidate_base_task'])
            and bridge.typed.same(task['root_context'],fragment['history']['candidate_context'])
            and fragment['history']['all_three_frames_actual_verified'] is True,'physical bridge vs actual model history/task')
        require(pro['local_task_sha256']==producer_digest(task) and pro['original_task_sha256']==producer_digest(pro['original_task'])
            and pro['actual_cp5_absolute_steps']==[5,10,15] and pro['actual_cp5_control_steps']==[-5,0,5],
            'actual cp5 time, no R control0 task substitution')
        for k in ('goal','projection','action_mean','action_std'):
            require(bridge.typed.same(task[k],pro['original_task'][k]),'unchanged original goal/Q/normalizer')
        require(task['binding']['task_id']!=pro['original_task']['binding']['task_id'],'base and local identity must stay distinct')
        require(type(arm['slot']) is type(arm['root']) is int and (arm['slot'],arm['root'],arm['horizon'])==(i,root,h)
            and arm['complete'] is True and arm['status'] in ('model_delivery_verified','no_witness')
            and arm['model_delivery_verified'] is (arm['status']=='model_delivery_verified')
            and arm['no_witness'] is (arm['status']=='no_witness'),'settled all4 witness/no-witness denominator')
        upstream_flags(arm,'actual R arm '+str(i))
        resource=arm['resources'];bridge.typed.core.audit_ledger(resource)
        require(arm['selected_index']==compiled['groups'][i]['selected_index']
            and arm['eligible_order']==compiled['groups'][i]['eligible_order'],'fixed original selected order')
        prefix=f32(pro['prefix5'],(5,2));a=fragment['anchor'];native_root=meta[root]
        require(prefix.tobytes()==f32(a['prefix_raw5'],(5,2)).tobytes() and a['condition']=='off'
            and a['root_sha256']==task['binding']['root_sha256']==native_root['root_sha256']
            and bridge.typed.same(a['goal'],native_root['goal']) and bridge.typed.same(a['goal_pose'],native_root['goal_pose']),
            'native root/goal/off and executed prefix')
        warm=f32(a['warm_raw10'],(10,2));incoming=np.concatenate((warm[-5:],prefix))
        require(warm.tobytes()==moving.warmup_commands().tobytes()
            and incoming.tobytes()==f32(pro['raw10'],(10,2)).tobytes()==f32(fragment['history']['raw10'],(10,2)).tobytes(),
            'complete actual warm10 and incoming causal raw10')
        tokens=bridge.typed.tokens(incoming,np.asarray(task['action_mean'],np.float32),np.asarray(task['action_std'],np.float32)).reshape(2,10)
        require(tokens.tobytes()==np.asarray(fragment['history']['history_tokens'],np.float32).tobytes()
            ==np.asarray(task['root_context']['action_history'],np.float32).tobytes(),
            'once normalized actual raw10 exact history tokens before prepare')
        cp=deepcopy(a['checkpoint']);cp={k:cp[k] for k in ('png','file_sha256','pixel_sha256','state','snapshot')}|{'step':5}
        reference=dict(index=i,complete=True,frames=deepcopy(a['prefix51']),checkpoint=cp,
            warm_raw10=warm.tolist(),prefix_raw5=prefix.tolist(),history_frames=deepcopy(fragment['frames']),
            raw10=incoming.tolist(),history_tokens=deepcopy(fragment['history']['history_tokens']),
            actual_context=deepcopy(fragment['history']['candidate_context']),bridge_sha256=arguments['bridge_sha'])
        d=arm['delivery'];commit=None
        if arm['model_delivery_verified']:
            require(d['candidate_index']==arm['selected_index'] and d['eligible_order']==arm['eligible_order']
                and d['source_row_sha256']==compiled['groups'][i]['items'][arm['selected_index']]['row_sha256']
                and d['source_row_pointer']==compiled['groups'][i]['items'][arm['selected_index']]['origin_pointer']
                and bridge.typed.same(d['raw_edge'],compiled['groups'][i]['items'][arm['selected_index']]['actions'])
                and d['source_closure_sha256']==producer_digest(compiled['sources'])
                and d['root_context_role']=='actual_observed_abs5_10_15_control_minus5_0_5',
                'actual full source closure and physical cp5 role, not original R control0')
            commit=validate_delivery(d,task,pro,h)
        else:require(d is None and arm['selected_index'] is None and not arm['eligible_order'],'no witness must not contain a fallback edge')
        commands=None if commit is None else np.concatenate((prefix,commit)).tolist()
        plan=dict(schema=NAME+'.execution_plan',index=i,root_slot=root,source_slot=root,condition='off',arm='H'+str(h),horizon=h,
            task_id=pro['original_task']['binding']['task_id'],local_task_id=task['binding']['task_id'],model_task=deepcopy(task),
            base_task=deepcopy(pro['original_task']),provenance=deepcopy(pro),model_delivery=deepcopy(d),
            witness_status=arm['status'],commands=commands,raw_steps=0 if commands is None else 10,
            raw_float32_sha256=None if commands is None else bytehash(f32(commands,(10,2))),
            prefix_commands=prefix.tolist(),prefix_reference_sha256=digest(reference),
            native_goal=deepcopy(native_root['goal']),native_goal_pose=deepcopy(native_root['goal_pose']),
            native_root_sha256=native_root['root_sha256'],native_goal_pixel_sha256=native_root['goal_pixels']['off'],
            selection_uses_physics=False,**FLAGS)
        plan['selection_digest']=digest(plan);plans.append(plan);refs.append(reference)
    p=dict(schema=NAME+'.prepare',contract=contract(),arguments=deepcopy(arguments),plans=plans,prefix_reference=refs,
        planner_authority=dict(producer=PRODUCER,external=deepcopy(arguments),complete_saved_consume_rebuilt=True,
            persisted_receipt_equal=True,model_resources=[deepcopy(a['resources']) for a in consumed['arms']],
            scalar_backward_calls=None,total_FLOPs=None,plans_sha256=digest(plans),
            inherited_accounting=deepcopy(consumed['accounting']),**FLAGS),
        source_report_sha256=native.SOURCE_SHA,source_roots=meta,source_slots=[deepcopy(x['slot']) for x in data['roots']],
        warm_traces=[deepcopy(x['warm_trace']) for x in data['roots']],bindings={})
    validate_execution(p);return p


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and len(p['plans'])==len(p['prefix_reference'])==4
        and len(p['source_roots'])==len(p['source_slots'])==len(p['warm_traces'])==2
        and p['source_report_sha256']==native.SOURCE_SHA,'own actual cp5 physical schema/denominator')
    authority=p['planner_authority']
    require(set(p['arguments'])==ARGUMENTS and authority['external']==p['arguments']
        and authority['producer']==PRODUCER and authority['complete_saved_consume_rebuilt'] is authority['persisted_receipt_equal'] is True
        and authority['plans_sha256']==digest(p['plans']) and len(authority['model_resources'])==4,'complete external actual model authorities')
    for i,(plan,ref,(root,h)) in enumerate(zip(p['plans'],p['prefix_reference'],ORDER,strict=True)):
        require(plan['schema']==NAME+'.execution_plan' and type(plan['index']) is type(plan['root_slot']) is int
            and (plan['index'],plan['root_slot'],plan['horizon'],plan['condition'],plan['arm'])==(i,root,h,'off','H'+str(h))
            and plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})
            and plan['prefix_reference_sha256']==digest(ref),'ordered own plan/reference binding')
        for flags in (plan,authority):require(all(flags[k] is False for k in FLAGS),'no physical/scientific admission')
        t=bridge.typed.core.validate_task(plan['model_task']);pro=plan['provenance'];a=p['source_roots'][root]
        require(plan['task_id']==plan['base_task']['binding']['task_id'] and plan['local_task_id']==t['binding']['task_id']
            and bridge.typed.same(plan['base_task'],pro['original_task']) and bridge.typed.same(ref['actual_context'],t['root_context'])
            and ref['complete'] is True and len(ref['frames'])==51 and ref['checkpoint']['step']==5,'full base/local task and cp5 restore witness')
        require(all(bridge.typed.same(plan[k],a[v]) for k,v in (('native_goal','goal'),('native_goal_pose','goal_pose'),('native_root_sha256','root_sha256')))
            and plan['native_goal_pixel_sha256']==a['goal_pixels']['off'],'unchanged native root/goal')
        if plan['witness_status']=='no_witness':
            require(plan['commands'] is plan['model_delivery'] is plan['raw_float32_sha256'] is None and plan['raw_steps']==0,'no witness no dispatchable plan')
        else:
            require(plan['witness_status']=='model_delivery_verified' and plan['raw_steps']==10,'settled witness only')
            commit=validate_delivery(plan['model_delivery'],t,pro,h);commands=f32(plan['commands'],(10,2))
            require(commands.tobytes()==np.concatenate((f32(plan['prefix_commands'],(5,2)),commit)).tobytes()
                and plan['raw_float32_sha256']==bytehash(commands),'prefix + FIRST5 only, raw literal identity')


def rebuild_delivery(arguments,producer):
    rebuilt=producer.consume(Path(arguments['delivery_output']),arguments['delivery_report_sha'],
        arguments['delivery_prepared'],arguments['delivery_prepare_sha'],arguments['delivery_completion_sha'])
    persisted=read(arguments['delivery_receipt'],META)
    require(bridge.typed.same(rebuilt,persisted),'entire actual saved secant consume equal; not status-only')
    return rebuilt


def material(arguments):
    require(type(arguments) is dict and set(arguments)==ARGUMENTS,'all explicit actual authority paths/SHA required')
    for k,v in arguments.items():native.actual_hash(v) if k.endswith('_sha') else native.absolute(v)
    require(sha(PROTOCOL)==PROTOCOL_SHA and sha(bridge.__file__)==BRIDGE_SHA,'frozen MAIN protocol/bridge source')
    code=ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py')
    require(sha(code)==arguments['delivery_source_sha'],'MAIN supplied actual delivery source SHA')
    producer=importlib.import_module('experiments.spectral_reachability.'+PRODUCER)
    out=Path(arguments['delivery_output']);rp=out/'report.json'
    for path,h in ((out/'completion.json',arguments['delivery_completion_sha']),(rp,arguments['delivery_report_sha']),(arguments['delivery_receipt'],arguments['delivery_receipt_sha']),
        (arguments['bridge'],arguments['bridge_sha'])):require(sha(path)==h,'external actual authority bytes')
    u=rebuild_delivery(arguments,producer)
    r=read(rp,producer.profile()['final_reserve_bytes']);c=read(arguments['delivery_prepared'],META)
    require(r['prepare_sha256']==arguments['delivery_prepare_sha'] and sha(arguments['delivery_prepared'])==arguments['delivery_prepare_sha']
        and c['sources'].get(str(code))==arguments['delivery_source_sha'],'actual complete R source/prepare binding')
    b=read(arguments['bridge'],META)
    require(bridge.typed.same(bridge.build(b['inputs']),b),'entire actual physical/history bridge reconstruction')
    require(shape_census()['total_upper_bytes']<=CAP,'all4 PNG + metadata + reserve census fits')
    data=native.source_material();p=assemble(arguments,c,u,b,data)
    bindings=deepcopy(c['sources']);native.merge(bindings,data['bindings'])
    for path,h in ((out/'completion.json',arguments['delivery_completion_sha']),(rp,arguments['delivery_report_sha']),(arguments['delivery_receipt'],arguments['delivery_receipt_sha']),
        (arguments['delivery_prepared'],arguments['delivery_prepare_sha']),(arguments['bridge'],arguments['bridge_sha'])):
        native.merge(bindings,{str(path):h})
    for desc in b['inputs'].values():
        if desc is not None:native.merge(bindings,{desc['path']:desc['sha256']})
    physical_strict=read(b['inputs']['physical_receipt']['path'],META)
    native.merge(bindings,physical_strict['bindings'])
    if b['history_prepare'] is not None:
        hp=b['history_prepare'];native.merge(bindings,{hp['path']:hp['sha256']})
        native.merge(bindings,read(hp['path'],META)['bindings'])
    for fragment in b['roots']:
        native.merge(bindings,{fragment['history']['latent_artifact']['path']:fragment['history']['latent_artifact']['sha256']})
        for frame in fragment['frames']:
            native.merge(bindings,{frame['png']:frame['file_sha256']})
    for path,h in r['artifact_sha256'].items():native.merge(bindings,{str(out/path):h})
    for f in [*own_files(),PROTOCOL,*[Path(m.__file__).resolve() for m in
        (native,native.allmode,native.base,source,moving,paired,primitives,bridge,bridge.typed)]]:
        native.merge(bindings,{str(f):sha(f)})
    for f,h in b['recognized_sources'].items():
        path=ROOT/'experiments/spectral_reachability'/(f+'.py');native.merge(bindings,{str(path):h})
    constructor,h=native.allmode.constructor_contract();native.merge(bindings,{constructor:h})
    p['bindings']=bindings
    for f,h in bindings.items():require(sha(f)==h,'complete source/input identity after saved reconstruction')
    require(len(raw(p))<=META,'full128MiB metadata, never drop ledger/prefix');return p,data


def write_new(path,value):
    p=Path(native.absolute(str(path)));body=raw(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink()
        and len(body)<=META and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'exclusive new own metadata')
    with p.open('xb') as f:require(f.write(body)==len(body),'complete metadata');f.flush();os.fsync(f.fileno())
    paired.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p),bytes=len(body))


def prepare(output,arguments):
    require(not Path(output).exists(),'no overwrite before saved reconstruction');p,_=material(arguments);return write_new(output,p)


def verify(prepared,expected):
    native.actual_hash(expected);require(sha(prepared)==expected,'actual own prepare SHA')
    p=read(prepared,META);rebuilt,data=material(p['arguments'])
    require(raw(rebuilt)==raw(p) and sha(prepared)==expected,'whole own prepare/source reconstruction');return p,data


def guard_proof(row,ref,plan,receipts):
    require(plan['witness_status']=='model_delivery_verified' and plan['model_delivery'] is not None,'no model witness means no commit')
    require(primitives.prefix_match(row,ref),'full actual51/cp5 exact state/snapshot/three-condition RGB before control6')
    ordered=[]
    for phase,n,key in (('warmup',10,'warm_raw10'),('plan',5,'prefix_raw5')):
        ops=[x for x in receipts if x['branch']==row['index'] and x['resource']=='public_step' and x['phase']==phase]
        expected=f32(ref[key],(n,2))
        require(len(ops)==n and [x['step'] for x in ops]==list(range(1,n+1))
            and all(x['status']=='returned' and x['raw_bits']==expected[j].tobytes().hex() for j,x in enumerate(ops)),
            'all actual warm10/prefix5 raw returns before control6')
        ordered.append(np.stack([np.frombuffer(bytes.fromhex(x['raw_bits']),dtype='<f4') for x in ops]))
    actual=np.concatenate((ordered[0][-5:],ordered[1]));t=plan['model_task']
    require(actual.tobytes()==f32(ref['raw10'],(10,2)).tobytes()==f32(plan['provenance']['raw10'],(10,2)).tobytes(),
        'actual causal incoming raw10 identity')
    tokens=bridge.typed.tokens(actual,np.asarray(t['action_mean'],np.float32),np.asarray(t['action_std'],np.float32)).reshape(2,10)
    require(tokens.tobytes()==np.asarray(ref['history_tokens'],np.float32).tobytes()==np.asarray(t['root_context']['action_history'],np.float32).tobytes()
        and bridge.typed.same(ref['actual_context'],t['root_context']),'once-normalized tokens and authenticated actual latent3 model context')
    selected=[x['receipt'] for x in row['prefix51']['frames'] if x['receipt']['overlay_index']==0 and
        (x['receipt']['phase'],x['receipt']['step']) in (('warmup',5),('warmup',10),('plan',5))]
    require(len(selected)==3 and [x['absolute_step'] for x in ref['history_frames']]==[5,10,15],'all three actual history times')
    fields=('state','snapshot','file_sha256','pixel_sha256')
    require(all(all(bridge.typed.same(a[k],b[k]) for k in fields) for a,b in zip(selected,ref['history_frames'],strict=True)),
        'all three actual RGB/native history anchors before first commit')
    return dict(prefix_reference_sha256=digest(ref),prefix51_sha256=digest(row['prefix51']),
        checkpoint_sha256=digest([x for x in row['checkpoints'] if x['step']==5]),
        actual_raw10_float32_sha256=bytehash(actual),history_tokens_float32_sha256=bytehash(tokens),
        model_context_sha256=producer_digest(t['root_context']),local_task_sha256=producer_digest(t),bridge_sha256=ref['bridge_sha256'])


class Store(native.allmode.Store):
    def __init__(self,path):
        self.path=Path(native.absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink()
            and self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new64MiB output +200MiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent);self.previous=None;self.index=0
    def check(self,size=4096,final=False):
        files=list(self.path.iterdir())
        require(all(x.is_file() and not x.is_symlink() for x in files) and
            sum(x.stat().st_size for x in files)+size<=CAP-(0 if final else FINAL)
            and shutil.disk_usage(self.path).free>=FLOOR+size,'own cap/reserve/floor including partial bytes')
    def write(self,name,value,append=False,final=False):
        require(type(value) is bytes,'exact serialized bytes')
        files=list(self.path.iterdir());c=shape_census()
        if name.endswith('.png'):
            require(not append and len(value)<=c['PNG_file_upper_bytes']
                and sum(x.suffix=='.png' for x in files)<c['PNG_count'],'full264 PNG count/encoded byte bound')
        else:
            require(name in {'execution_binding.json','journal.jsonl','report.json','completion.json','completion_failure.json'},'no hidden output format')
            if name=='journal.jsonl':
                e=native.decode(value)
                require(append and value==raw(e) and value.endswith(b'\n') and not value.endswith(b'\n\n'),
                    'canonical exactly-one-newline journal')
            else:require(not append and value==raw(native.decode(value)),'canonical metadata')
            require(sum(x.stat().st_size for x in files if x.suffix!='.png')+len(value)<=METADATA_CAP,
                'aggregate metadata quota covers journal/row/operation copies')
        return super().write(name,value,append=append,final=final)


def publish(store,report,started,clock=None):
    clock=time.monotonic if clock is None else clock
    store.put('report.json',report,final=True)
    hashes={x.name:sha(x) for x in store.path.iterdir()}
    inclusive=clock()-started
    completion=dict(schema=NAME+'.completion',report_sha256=hashes['report.json'],
        prepared=report['execution_binding']['prepared'],prepare_sha256=report['execution_binding']['prepare_sha256'],
        artifact_sha256=hashes,bytes_before_completion=sum(x.stat().st_size for x in store.path.iterdir()),
        inclusive_wall_seconds=inclusive,wall_scope='through report fsync and full artifact hash; completion self-publication excluded',
        status='published_within_cap' if inclusive<=WALL else 'publication_wall_failed',**FLAGS)
    store.put('completion.json',completion,final=True);completion_sha=sha(store.path/'completion.json')
    end=clock()-started
    if end>WALL:
        store.put('completion_failure.json',dict(schema=NAME+'.completion_failure',wall_seconds=end,
            report_sha256=hashes['report.json'],completion_sha256=completion_sha,**FLAGS),final=True)
    require(end<=WALL,'final publication wall failed; durable failure retained')
    return dict(report_sha256=hashes['report.json'],completion_sha256=completion_sha)


class Meter(native.Meter):
    def guard_context(self,row,ref,plan):self.row,self.ref,self.plan=row,ref,plan;self.guarded=False
    def call(self,kind,fn,*,cleanup=False,**fields):
        if kind=='public_step' and fields.get('phase')=='plan' and fields.get('step',0)>=6:
            if not self.guarded:
                require(fields['step']==6,'first actual commit begins at control6')
                try:proof=guard_proof(self.row,self.ref,self.plan,self.receipts)
                except BaseException as exc:
                    self.fault=True;self.store.emit('cp5_guard',**self.where,accepted=False,proof=None,error=native.base.error('cp5_guard',exc));raise
                # emit fsync MUST complete before inherited intent/callback. No flag
                # is set if the journal write fails, so no native suffix callback.
                self.store.emit('cp5_guard',**self.where,accepted=True,proof=proof,error=None);self.guarded=True
            require(6<=fields['step']<=10,'no H3 lookahead dispatch')
        return super().call(kind,fn,cleanup=cleanup,**fields)


def rows_for(p):
    rows=primitives.rows_for(p)
    for row in rows:row['prefix_reference_exact']=False
    return rows


def execute(p,data,store,factory,*,authorization,prepared,prepare_sha256,started=None):
    require(authorization==AUTH,'explicit MAIN authority before env dispatch');validate_execution(p);native.actual_hash(prepare_sha256)
    require(p['source_slots']==[x['slot'] for x in data['roots']] and p['source_roots']==native.allmode.root_metadata(data)
        and p['warm_traces']==[x['warm_trace'] for x in data['roots']],'exact source reset/root/history')
    meter=Meter(store,started);rows=rows_for(p);errors=[];seen=[]
    binding=dict(prepared=native.absolute(str(prepared)),prepare_sha256=prepare_sha256,planner_authority=deepcopy(p['planner_authority']),source_report_sha256=native.SOURCE_SHA)
    def fail(stage,exc,row=None):
        item=native.base.error(stage,exc);errors.append(item)
        if row is not None:
            if row['first_error'] is None:row['first_error']=deepcopy(item)
            else:row['secondary_errors'].append(deepcopy(item))
            row['status']='engineering_failed'
    try:
        store.put('execution_binding.json',binding);store.emit('run_started',prepare_sha256=prepare_sha256,contract=contract())
        for row,plan,ref in zip(rows,p['plans'],p['prefix_reference'],strict=True):
            meter.where=dict(branch=row['index'],**{k:row[k] for k in ('root_slot','condition','arm','selection_digest')})
            if plan['witness_status']=='no_witness':
                row['status']='no_witness';store.emit('branch_closed',receipt=row);continue
            if errors:break
            meter.guard_context(row,ref,plan);env=None;row['status']='running'
            try:
                env=meter.call('constructor',factory)
                require(all(env is not e and env.unwrapped is not e.unwrapped for e in seen),'independent fresh env per executable slot');seen.append(env)
                native.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','original renderer/raw ABI')
                with native.native_hooks(env,meter,row['secondary_errors']):
                    native.episode(env,data['roots'][plan['root_slot']],data['source_plan'],plan,row,meter,store)
                require(not row['secondary_errors'],'native hooks restored')
            except native.allmode.NativeStop as exc:row.update(status='native_stopped',stop_reason=str(exc))
            except BaseException as exc:fail('branch',exc,row)
            finally:
                try:row['prefix_reference_exact']=primitives.prefix_match(row,ref)
                except BaseException as exc:fail('prefix_reference',exc,row)
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as exc:fail('close',exc,row)
                if row['secondary_errors'] and row['first_error'] is None:fail('cleanup',ValueError('native cleanup secondary'),row)
                for phase,label in (('warmup','warmup'),('plan','own')):
                    ops=[x for x in meter.receipts if x['branch']==row['index'] and x['resource']=='public_step' and x['phase']==phase]
                    row[label+'_attempted']=len(ops);row[label+'_returned']=sum(x['status']=='returned' for x in ops)
            store.emit('branch_closed',receipt=row)
    except BaseException as exc:fail('execution',exc)
    inputs_ok=False
    try:
        for f,h in p['bindings'].items():require(sha(f)==h,'final source/model/history seal')
        inputs_ok=True
    except BaseException as exc:fail('final_inputs',exc)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=WALL:fail('final_wall',ValueError('wall cap'))
    settled=all(x['status'] in ('horizon_completed','native_stopped','no_witness') for x in rows)
    status=RECORDED if not errors and inputs_ok and settled else 'partial_or_failed'
    try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    except BaseException as exc:fail('final_journal',exc);status='partial_or_failed'
    report=dict(schema=NAME+'.report',status=status,contract=contract(),execution_binding=binding,rows=rows,errors=errors,
        counts=deepcopy(meter.counts),actual_operation_receipts=deepcopy(meter.receipts),inputs_final_verified=inputs_ok,
        fixed_denominator=4,known_roots=2,fresh_roots=0,inherited=0,new_neural_calls=0,new_independent_samples=0,
        journal_sha256=None,execution_binding_sha256=None,journal_chain_tail=store.previous,wall_seconds=time.monotonic()-meter.started,**FLAGS)
    for name,key in (('journal.jsonl','journal_sha256'),('execution_binding.json','execution_binding_sha256')):
        try:report[key]=sha(store.path/name)
        except BaseException as exc:fail('final_'+key,exc);report['status']='partial_or_failed'
    try:publish(store,report,meter.started)
    except BaseException as exc:fail('final_report',exc);report['status']='partial_or_failed'
    return report


def audit_events(r,events,p):
    allowed={'run_started','run_finished','operation_dispatch_intent','operation_returned','operation_failed','placement_recipe',
        'goal_attached','invalid_observation','fresh_observation','returned_transition','own_response','branch_closed','cp5_guard'}
    require(events and events[0]['kind']=='run_started' and events[0]['contract']==contract()
        and events[0]['prepare_sha256']==r['execution_binding']['prepare_sha256'] and events[-1]['kind']=='run_finished'
        and all(events[-1][k]==r[k] for k in ('status','counts','errors')),'complete own journal bookends')
    tail=None;intents={};terminals={};stack=[];closed=0
    for i,e in enumerate(events):
        require(e['kind'] in allowed and type(e['index']) is int and e['index']==i and e['previous_sha256']==tail
            and e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'exact event whitelist/hashchain')
        tail=e['event_sha256']
        if e['kind']=='branch_closed':
            require(closed<4 and raw(e['receipt'])==raw(r['rows'][closed]),'exact ordered full branch closure');closed+=1
        if 'branch' in e:
            require(type(e['branch']) is int and e['branch']==closed and closed<4 and
                all(e[k]==r['rows'][closed][k] for k in ('root_slot','condition','arm','selection_digest')),'active own branch identity')
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        op=e['receipt'];n=op['index'];b=op['branch']
        require(type(n) is type(b) is int and b==closed and 0<=b<4 and op['resource'] in contract()['caps']
            and all(op[k]==r['rows'][b][k] for k in ('root_slot','condition','arm','selection_digest')),'operation resource/branch identity')
        if e['kind']=='operation_dispatch_intent':
            require(n==len(intents) and op['status']=='pending','contiguous native operation attempt');intents[n]=op;stack.append(n)
        else:
            require(stack and stack.pop()==n and n in intents and n not in terminals
                and op['status']==('returned' if e['kind']=='operation_returned' else 'failed')
                and dict({k:v for k,v in op.items() if k!='error'},status='pending')==intents[n],'all nested native operations close exactly')
            terminals[n]=op
    require(not stack and len(intents)==len(terminals) and tail==r['journal_chain_tail']
        and raw(r['actual_operation_receipts'])==raw([terminals[i] for i in range(len(terminals))]),'no missing/extra native operation receipts')
    counts={k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for op in r['actual_operation_receipts']:
        counts[op['resource']]['attempted']+=1;counts[op['resource']]['completed']+=op['status']=='returned'
    require(counts==r['counts'] and all(v['attempted']<=contract()['caps'][k] for k,v in counts.items()),'full physical resource vector/caps')
    require([e['receipt'] for e in events if e['kind']=='branch_closed']==[row for row in r['rows'] if row['status']!='not_started'],
        'all completed/failed/no-witness rows journaled')
    return closed


def audit_guard(r,events,p):
    for row,plan,ref in zip(r['rows'],p['plans'],p['prefix_reference'],strict=True):
        i=row['index'];scoped=[e for e in events if e.get('branch')==i];guards=[e for e in scoped if e['kind']=='cp5_guard']
        suffix=[e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['branch']==i
            and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan' and e['receipt'].get('step',0)>=6]
        require(len(guards)<=1,'single causal guard')
        if not guards:require(not suffix,'no commit intent without guard');continue
        g=guards[0]
        require(set(g)=={'kind','index','previous_sha256','event_sha256','branch','root_slot','condition','arm','selection_digest','accepted','proof','error'}
            and type(g['accepted']) is bool,'exact guard event schema')
        prior=[x for x in r['actual_operation_receipts'] if x['branch']==i and x['resource']=='public_step'
            and (x['phase']=='warmup' or x['step']<=5)]
        frames=[e for e in scoped if e['kind']=='fresh_observation' and (e['receipt']['phase']=='warmup' or e['receipt']['step']<=5)]
        returns=[e for e in events if e['kind']=='operation_returned' and e['receipt'] in prior]
        if g['accepted']:
            require(g['error'] is None and g['proof']==guard_proof(row,ref,plan,prior)
                and len(frames)==51 and len(returns)==15 and max(e['index'] for e in frames+returns)<g['index']
                and all(g['index']<e['index'] and 6<=e['receipt']['step']<=10 for e in suffix),'guard after all prefix observations/returns BEFORE first commit intent')
        else:
            require(g['proof'] is None and g['error'] is not None and not suffix and row['status']=='engineering_failed','rejected guard no commit')
            rejected=False
            try:guard_proof(row,ref,plan,prior)
            except ValueError:rejected=True
            require(rejected,'a rejected guard must have a reproducible saved identity mismatch')


def validate_saved(r,events,p):
    validate_execution(p)
    require(r['schema']==NAME+'.report' and r['contract']==contract() and r['status'] in (RECORDED,'partial_or_failed')
        and r['fixed_denominator']==len(r['rows'])==4 and r['known_roots']==2 and r['fresh_roots']==r['inherited']==0
        and r['execution_binding']['planner_authority']==p['planner_authority']
        and r['execution_binding']['source_report_sha256']==native.SOURCE_SHA,'own schema actual model/physical authority')
    require(all(r[k] is False for k in FLAGS) and r['new_neural_calls']==r['new_independent_samples']==0,'no model/fresh/science claims')
    closed=audit_events(r,events,p)
    for i,(row,plan,ref) in enumerate(zip(r['rows'],p['plans'],p['prefix_reference'],strict=True)):
        require(row['index']==i and row['plan']==plan and all(row[k]==plan[k] for k in
            ('root_slot','condition','arm','task_id','selection_digest')) and row['status'] in
            ('not_started','engineering_failed','native_stopped','horizon_completed','no_witness'),'all4 literal plans/row identities')
        scoped=[e for e in events if e.get('branch')==i];ops=[x for x in r['actual_operation_receipts'] if x['branch']==i]
        observations=[e for e in scoped if e['kind']=='fresh_observation'];transitions=[e for e in scoped if e['kind']=='returned_transition']
        responses=[e['response'] for e in scoped if e['kind']=='own_response']
        require(responses==row['responses'] and row['PNG']==len(observations),'all actual native response/render census')
        if row['status']=='no_witness':
            require(plan['witness_status']=='no_witness' and not scoped and not ops and row['own_attempted']==row['warmup_attempted']==0
                and not row['trajectory'] and not row['checkpoints'] and row['missing_checkpoints']==[0,5,10]
                and row['final_state'] is row['final_metrics'] is row['native_success'] is None,'no witness = no env/action/labels')
            continue
        require(plan['witness_status']=='model_delivery_verified','no execution without model witness')
        for phase,label,commands in (('warmup','warmup',moving.warmup_commands()),('plan','own',f32(plan['commands'],(10,2)))):
            current=[x for x in ops if x['resource']=='public_step' and x['phase']==phase]
            returned=[x for x in transitions if x['phase']==phase]
            require(len(current)<=10 and [x['step'] for x in current]==list(range(1,len(current)+1))
                and all(x['raw_bits']==commands[j].tobytes().hex() for j,x in enumerate(current))
                and row[label+'_attempted']==len(current) and row[label+'_returned']==sum(x['status']=='returned' for x in current),
                'all exact raw actions, no padding or lookahead')
            require([x['step'] for x in returned]==list(range(1,len(returned)+1)) and len(returned)<=row[label+'_returned'],'every native transition ordered')
            for x in returned:
                require(x['actual_dtype']==np.dtype(np.float32).str and x['actual_bits']==x['requested_bits']==commands[x['step']-1].tobytes().hex(),
                    'raw native requested/returned bit identity')
        trace=[{k:deepcopy(e[k]) for k in ('pre','post','receipt')} for e in observations if e['receipt']['phase']=='warmup' or e['receipt']['step']<=5]
        require(row['prefix51']==dict(complete=len(trace)==51,frames=trace),'actual full prefix buffer, no splicing')
        root=p['source_roots'][plan['root_slot']];selected=[];unknown=[]
        for e in observations:
            f=e['receipt'];phase,step,overlay=f['phase'],f['step'],f['overlay_index']
            require(phase in ('warmup','plan') and type(step) is type(overlay) is int and 0<=step<=10 and overlay in (0,1,2)
                and f['condition']==('on' if overlay==1 else 'off') and raw(e['pre'])==raw(e['post'])
                and raw(f['snapshot'])==raw(e['post']['snapshot']) and raw(f['state'])==raw(f['snapshot']['state'])
                and len(f['state'])==7 and e['pre']['renderer']==root['renderer'] and e['pre']['goal_pose']==root['goal_pose'],
                'all native observation snapshots/renderer/goalpose')
            if overlay==0:selected.append(f)
            if phase=='warmup':
                old=native.allmode.reference_for(p['warm_traces'][plan['root_slot']],step,overlay)
                if old is None:unknown.append(step)
                else:native.allmode.check_warm_frame(e,old)
            else:
                require(e['pre']['goal_state']==root['goal'] and e['pre']['goal_rgb_sha256']==root['goal_pixels']['off'],'original bound task goal')
                if step:
                    t=next(x for x in transitions if x['phase']=='plan' and x['step']==step)
                    require(raw(f['state'])==raw(t['state']) and raw(f['snapshot'])==raw(t['snapshot']),'render tied to actual native return')
        require(row['trajectory']==selected and row['source_unpaired_on_steps']==unknown,'unmodified selected off trajectory/unknown warmup sources')
        cp=[f for f in selected if f['phase']=='plan' and f['step'] in (0,5,10)]
        require(row['checkpoints']==[{k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')} for f in cp]
            and row['missing_checkpoints']==[b for b in (0,5,10) if b not in [f['step'] for f in cp]],'missing actual checkpoints stay missing')
        require([x['step'] for x in responses]==list(range(1,len(responses)+1)) and len(responses)<=row['own_returned'],'all actual response steps')
        for response in responses:
            calc=native.metrics(row['goal'],response['state']);delta=source.vector(response['state'])-source.vector(row['root_state'])
            t=next(x for x in transitions if x['phase']=='plan' and x['step']==response['step'])
            require(raw(response['native_metrics'])==raw(calc['native']) and response['reward']==calc['native7_reward']
                and response['native_success'] is response['terminated'] is calc['native_success'] and type(response['truncated']) is bool
                and np.array_equal(delta,response['delta_native7']) and response['delta_angle_wrapped']==float(np.arctan2(np.sin(delta[4]),np.cos(delta[4])))
                and all(raw(response[k])==raw(t[k]) for k in ('state','reward','terminated','truncated')),'full native7 reward/success axes/return identity')
        if responses:
            last=responses[-1];require(raw(row['final_state'])==raw(last['state']) and raw(row['final_metrics'])==raw(last['native_metrics'])
                and all(row[k] is last[k] for k in ('native_success','terminated','truncated'))
                and all(not x['terminated'] and not x['truncated'] for x in responses[:-1]),'actual endpoint/stop no padding')
        else:require(row['final_state'] is row['final_metrics'] is row['native_success'] is None,'unknown outcomes null')
        if row['initial_exact']:
            require(all(raw(row[k])==raw(root[k]) for k in ('root_state','root_sha256','goal','goal_pose'))
                and row['goal_pixel_sha256']==root['goal_pixels']['off']
                and source.root_hash(source.vector(row['root_state']),source.vector(row['goal']))==row['root_sha256'],'exact initial native identity')
        if row['status'] in ('horizon_completed','native_stopped'):
            require(row['first_error'] is None and row['secondary_errors']==[],'native stop not engineering failure')
            framekeys=[('warmup',j,k) for j in range(row['warmup_returned']+1) for k in range(3)]
            if row['initial_exact']:framekeys += [('plan',j,k) for j in range(row['own_returned']+1) for k in range(3)]
            require([(x['receipt']['phase'],x['receipt']['step'],x['receipt']['overlay_index']) for x in observations]==framekeys,'complete unpadded RGB triplets')
            for j in range(0,len(observations),3):
                a,b,c=[e['receipt'] for e in observations[j:j+3]]
                require(a['file_sha256']==c['file_sha256'] and a['pixel_sha256']==c['pixel_sha256']
                    and raw(a['state'])==raw(b['state'])==raw(c['state']),'same-state off/on/off restore')
            if responses:require(row['own_returned']==len(responses) and row['warmup_returned']==10 and row['initial_exact']
                and (row['status']=='native_stopped') is (last['terminated'] or last['truncated']),'native termination retained')
            if row['status']=='horizon_completed':require(row['own_returned']==10,'FIRST5 commit completed after original prefix5')
        try:prefix_exact=primitives.prefix_match(row,ref)
        except ValueError:
            prefix_exact=False
            require(row['status']=='engineering_failed' and row['first_error'] is not None,'saved mismatch cannot be native success')
        require(row['prefix_reference_exact'] is prefix_exact,'finally exact prefix check preserved')
    audit_guard(r,events,p);primitives.audit_all_labels(r,events,p)
    if r['status']==RECORDED:
        require(closed==4 and r['errors']==[] and r['inputs_final_verified'] is True and 0<=r['wall_seconds']<WALL
            and all(x['status'] in ('horizon_completed','native_stopped','no_witness') for x in r['rows']),'all four settled denominator only')
        c={k:v['completed'] for k,v in r['counts'].items()};executed=sum(x['status']!='no_witness' for x in r['rows'])
        require(all(v['attempted']==v['completed'] for v in r['counts'].values()) and c['constructor']==c['close']==executed
            and c['reset']==2*executed and c['set_state']==2*c['reset'] and c['goal_render']==c['reset']
            and c['goal_bind']==sum(x['initial_exact'] for x in r['rows']) and c['goal_setter']==c['reset']+c['goal_bind']
            and c['physics_step']==2*c['reset']+10*c['public_step']
            and c['public_step']==sum(x['warmup_returned']+x['own_returned'] for x in r['rows'])
            and c['explicit_render']==sum(x['PNG'] for x in r['rows']),'full actual physical resource census, no refunds')
    return deepcopy(r['rows'])


def run(prepared,expected,output,factory,*,authorization=None):
    require(authorization==AUTH,'MAIN explicit authorization before reads/output/physics')
    p,data=verify(prepared,expected);store=Store(output)
    return execute(p,data,store,factory,authorization=authorization,prepared=prepared,prepare_sha256=expected)


def consume(output,expected,completion_sha):
    native.actual_hash(expected);directory=Path(native.absolute(str(output)))
    require(directory.is_dir() and not directory.is_symlink(),'real flat output directory')
    files={}
    for p in directory.iterdir():
        require(p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,'bounded regular artifacts');files[p.name]=p.read_bytes()
    require(sum(map(len,files.values()))<=CAP and hashlib.sha256(files['report.json']).hexdigest()==expected,'own full output cap/report SHA')
    r=native.decode(files['report.json']);binding=r['execution_binding']
    native.actual_hash(completion_sha)
    require('completion_failure.json' not in files and hashlib.sha256(files['completion.json']).hexdigest()==completion_sha,
        'external completion SHA and no failed publication')
    completion=native.decode(files['completion.json'])
    require(completion['schema']==NAME+'.completion' and completion['report_sha256']==expected
        and completion['prepared']==binding['prepared'] and completion['prepare_sha256']==binding['prepare_sha256']
        and completion['status']=='published_within_cap' and all(completion[k] is False for k in FLAGS)
        and completion['wall_scope']=='through report fsync and full artifact hash; completion self-publication excluded'
        and type(completion['inclusive_wall_seconds']) in (int,float)
        and 0<=r['wall_seconds']<=completion['inclusive_wall_seconds']<=WALL,'post-publication inclusive wall binding')
    require(completion['artifact_sha256']=={k:hashlib.sha256(v).hexdigest() for k,v in files.items() if k!='completion.json'}
        and completion['bytes_before_completion']==sum(len(v) for k,v in files.items() if k!='completion.json'),
        'full completion artifact census')
    census=shape_census()
    require(sum(k.endswith('.png') for k in files)<=census['PNG_count']
        and all(len(v)<=census['PNG_file_upper_bytes'] for k,v in files.items() if k.endswith('.png'))
        and sum(len(v) for k,v in files.items() if not k.endswith('.png'))<=METADATA_CAP,'independent serialization census')
    require(native.decode(files['execution_binding.json'])==binding and hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256'],
        'actual execution binding bytes')
    p,_=verify(binding['prepared'],binding['prepare_sha256']);journal=files['journal.jsonl']
    require(journal.endswith(b'\n') and hashlib.sha256(journal).hexdigest()==r['journal_sha256'],'full saved journal SHA')
    lines=journal.splitlines(keepends=True);events=[native.decode(line) for line in lines]
    require(all(line==raw(e) for line,e in zip(lines,events,strict=True)),'strict canonical journal codec')
    rows=validate_saved(r,events,p)
    from PIL import Image
    names={'report.json','journal.jsonl','execution_binding.json','completion.json'};n=0
    for e in events:
        if e['kind']=='invalid_observation':
            name=e['file'];require(Path(name).name==name and name not in ('','.','..')
                and hashlib.sha256(files[name]).hexdigest()==e['file_sha256'],'retained invalid PNG');names.add(name)
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];name=f['png']
        require(Path(name).name==name and name not in ('','.','..') and name not in names and len(files[name])<=1024**2
            and hashlib.sha256(files[name]).hexdigest()==f['file_sha256'],'unique actual PNG/file SHA')
        with Image.open(io.BytesIO(files[name])) as im:
            require(im.format=='PNG' and im.mode=='RGB' and im.size==(224,224),'original RGB224');rgb=np.asarray(im).copy()
        require(moving.image_sha(rgb)==f['pixel_sha256'],'exact decoded pixels');moving.guard(rgb,source.vector(f['state']))
        names.add(name);n+=1
    require(names==set(files),'complete artifacts, no hidden files')
    bindings=deepcopy(p['bindings']);native.merge(bindings,{binding['prepared']:binding['prepare_sha256']})
    native.merge(bindings,{str(directory/k):hashlib.sha256(v).hexdigest() for k,v in files.items()})
    for f,h in bindings.items():require(sha(f)==h,'final input/output/source seal')
    require({f.name:sha(f) for f in directory.iterdir()}=={k:hashlib.sha256(v).hexdigest() for k,v in files.items()},'stable artifact census')
    return dict(schema=NAME+'.consume',status=r['status'],physical_report_sha256=expected,rows=rows,counts=r['counts'],errors=r['errors'],
        planner_authority=p['planner_authority'],bindings=bindings,fixed_denominator=4,known_roots=2,fresh_roots=0,inherited=0,
        completion_sha256=completion_sha,completion=completion,shape_census=census,
        saved_PNG_verified=n,new_neural_calls=0,new_physics_calls=0,**FLAGS)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True)
    q.add_argument('--fixed-authorities',action='store_true')
    for key in sorted(ARGUMENTS):q.add_argument('--'+key.replace('_','-'))
    for cmd in ('verify','run','consume'):
        q=sub.add_parser(cmd);q.add_argument('--prepared',required=cmd!='consume');q.add_argument('--prepare-sha',required=cmd!='consume')
        if cmd in ('run','consume'):q.add_argument('--output',required=True)
        if cmd=='run':q.add_argument('--authorization',required=True)
        if cmd=='consume':q.add_argument('--report-sha',required=True);q.add_argument('--completion-sha',required=True);q.add_argument('--receipt',required=True)
    a=p.parse_args(argv)
    if a.cmd=='prepare':
        supplied={k:getattr(a,k) for k in ARGUMENTS}
        require(not a.fixed_authorities or not any(v is not None for v in supplied.values()),'no mixed preset/explicit authority')
        require(a.fixed_authorities or all(v is not None for v in supplied.values()),'all explicit authorities required')
        result=prepare(a.output,deepcopy(DEFAULT_ARGUMENTS) if a.fixed_authorities else supplied)
    elif a.cmd=='verify':verify(a.prepared,a.prepare_sha);result=dict(verified=True,execution_authorized=False)
    elif a.cmd=='consume':result=write_new(a.receipt,consume(a.output,a.report_sha,a.completion_sha))
    else:
        require(a.authorization==AUTH,'MAIN authority before physics imports')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        result=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
    print(source.canonical(result));return 2 if result.get('status')=='partial_or_failed' else 0


if __name__=='__main__':raise SystemExit(main())


