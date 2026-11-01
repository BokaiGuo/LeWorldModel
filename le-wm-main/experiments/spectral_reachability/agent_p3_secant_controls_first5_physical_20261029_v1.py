"""Stage-A known2 Z/P first5 controls, native execution by MAIN only.

Own schema; C is pinned inherited evidence, never rerun. No NN or gate changes.
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
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as task_loader

NAME='agent_p3_secant_controls_first5_physical_20261029_v1'
AUTH='MAIN_P3_ZP_FIRST5_PHYSICS_ONLY'
RECORDED='all4_ZP_first5_native_recorded_NOT_admitted'
ROOT=bridge.ROOT
native=primitives.mixed
source,moving,paired=native.source,native.moving,native.paired
require,sha,raw,read=native.require,native.sha,native.raw,native.read
digest=native.core.digest
producer_digest=bridge.typed.core.digest
ORDER=[(0,'Z'),(0,'P'),(1,'Z'),(1,'P')]
CAP,FINAL,FLOOR,WALL,META=64*1024**2,8*1024**2,200*1024**2,900,128*1024**2
METADATA_CAP=17*1024**2
PROTOCOL=ROOT/'innerdocs/agent_p3_secant_same_time_capability_protocol_20261029_v1.md'
PROTOCOL_SHA='b35af390fc74e6ebd44a9879784308ec74947fb28f640ce5e795cc263e65dccf'
CNAME='agent_p3_secant_commit_physical_20261029_v1'
SHARED='agent_p3_shared_panel_20261029_v1'
FLAGS=dict(P1_admission=False,P3_admission=False,full_C3_demonstrated=False,scientific_admission=False,
    physical_release=False,execution_authorized=False,freshness=False,equal_total_compute=False,
    equal_FLOPs=False,training=False,independent_confirmation=False)
# Fixed, already published authorities, not future receipt constants.
AUTHORITIES={
 'C_prepare':('results/'+CNAME+'_prepare_MAIN01.json','a085c3c6a609664e61305b156ce68d1cbdf44296634a42a8bc5416e7091fe295'),
 'C_report':('results/'+CNAME+'_physics_MAIN01/report.json','f64d2843ecf0c217a5581d25a10fa77ccf78c372bf1ddc2aca8995ab4af0a616'),
 'C_completion':('results/'+CNAME+'_physics_MAIN01/completion.json','f68fb8ef6638a2fa3c74a8139d1b7a517a7c2d36df02122104b759998b470d75'),
 'C_receipt':('results/'+CNAME+'_consume_MAIN01.json','d0c5a34b14dc84ffece573d904e3950e54628df307987ee77fcb6265ca91b638'),
 'C_source':('experiments/spectral_reachability/'+CNAME+'.py','a065c8f652919d8315f7848c70e9b944ad6aec0717bc6bbe310b54b224c72aea'),
 'P_prepare':('results/'+SHARED+'_prepare_MAIN01.json','7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a'),
 'P_receipt':('results/'+SHARED+'_consume_MAIN01.json','d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd')}
SELECTIONS=['97bcf5e4d3fbf5ed20db407cc7bbeadedf5d8a663bd4aad640c0056e45b71ed9',
 '9120059452c4a1760e4b355f61051639b9445f29d6a628f7f419499a96f40d60']


def shape_census():
    scan=224*(224*3+1)
    png=scan+(scan>>12)+(scan>>14)+(scan>>25)+13+1024
    count=4*3*(11+11);working=count*png+METADATA_CAP
    return dict(groups=4,PNG_count=count,PNG_per_group=66,scan_bytes=scan,
        PNG_file_upper_bytes=png,PNG_total_upper_bytes=count*png,
        metadata_upper_bytes=METADATA_CAP,working_upper_bytes=working,
        final_reserve_bytes=FINAL,total_upper_bytes=working+FINAL,output_cap_bytes=CAP,
        prefix_frames_per_group=51,prefix_persisted_copies=2,
        operation_receipt_persisted_copies=3,observation_event_copies=1,
        row_persisted_copies=2,NPY_count=0,
        full_operation_receipts=1216,full_operation_journal_events=2432,
        full_observation_journal_events=264,full_branch_closures=4,
        metadata_bound_kind='aggregate enforced byte quota, not an arbitrary copy-factor extrapolation')


def contract():
    c=deepcopy(native.allmode.contract())
    c.update(schema=NAME+'.contract',order=[list(x) for x in ORDER],condition='off',
        arm='fixed_Z_or_original_P_first5',prefix=5,commit=5,horizon=10,
        checkpoints=[0,5,10],raw_authority='original prefix5 + literal Z/P raw5; no clipping/mapping',
        cp5_guard='durable accepted guard BEFORE control6 intent; full51 + actual raw history',
        missing_endpoint='native earlystop missing cp10; no padding or fallback',
        C_reference_only=True,logical_comparison_slots=12,inheritance=False,
        producer='literal_fixed_controls_NOT_model_delivery',
        output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        wall_seconds=WALL,prepare_cap_bytes=META,protocol_sha256=PROTOCOL_SHA,
        shape_census=shape_census(),
        trust_boundary='pinned MAIN complete C receipt; no recursive C fullstrict; new journal fully rebuilt',
        interpretation='known2 fixed controls only; no ability or non-greedy benefit claim',**FLAGS)
    return c


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def f32(value,shape):return bridge.typed.f32(value,shape,box=True)
def bytehash(value):return hashlib.sha256(np.asarray(value,np.dtype('<f4')).tobytes()).hexdigest()


def authority_descriptors():
    return {k:dict(path=str(ROOT/p),sha256=h) for k,(p,h) in AUTHORITIES.items()}


def validate_C(cp,r,completion,u):
    """Explicit pinned complete-receipt trust, NOT a rerun of inherited fullstrict."""
    d=authority_descriptors()
    require(cp['schema']==CNAME+'.prepare' and r['schema']==CNAME+'.report'
        and u['schema']==CNAME+'.consume' and completion['schema']==CNAME+'.completion',
        'C actual own schema, not projected legacy')
    require(u['status']==r['status']=='secant_commit5_native_recorded_NOT_admitted'
        and u['physical_report_sha256']==d['C_report']['sha256']
        and u['completion_sha256']==d['C_completion']['sha256']
        and u['completion']==completion and completion['report_sha256']==d['C_report']['sha256']
        and completion['prepared']==r['execution_binding']['prepared']==d['C_prepare']['path']
        and completion['prepare_sha256']==r['execution_binding']['prepare_sha256']==d['C_prepare']['sha256']
        and completion['status']=='published_within_cap','all four C authorities exact cross binding')
    require(u['rows']==r['rows'] and u['counts']==r['counts'] and u['errors']==r['errors']==[]
        and u['planner_authority']==r['execution_binding']['planner_authority']==cp['planner_authority']
        and u['fixed_denominator']==r['fixed_denominator']==4 and len(u['rows'])==len(cp['plans'])==4
        and u['saved_PNG_verified']==66 and r['inputs_final_verified'] is True,'entire C rows/accounting vs MAIN complete receipt')
    require([x['status'] for x in u['rows']]==['horizon_completed']+['no_witness']*3,
        'one existing C endpoint, three absent; no new selection')
    require(all(all(value[k] is False for k in FLAGS) for value in (r,u,completion)),
        'C no scientific permission')
    require(cp['bindings'][d['P_prepare']['path']]==d['P_prepare']['sha256']
        and cp['bindings'][d['C_source']['path']]==d['C_source']['sha256']
        and u['bindings'][d['C_prepare']['path']]==d['C_prepare']['sha256']
        and u['bindings'][d['C_report']['path']]==d['C_report']['sha256']
        and u['bindings'][d['C_completion']['path']]==d['C_completion']['sha256'],
        'C original source/P prepare and receipt closure')
    for i,row in enumerate(u['rows']):
        require(row['index']==i and row['plan']==cp['plans'][i]
            and (row['root_slot'],row['arm'])==[(0,'H1'),(0,'H3'),(1,'H1'),(1,'H3')][i],
            'C all4 original slots/plan identity')
        if i==0:
            require(row['prefix_reference_exact'] is True and row['own_returned']==10 and row['PNG']==66
                and row['missing_checkpoints']==[] and [x['step'] for x in row['checkpoints']]==[0,5,10]
                and row['prefix51']['complete'] is True and len(row['prefix51']['frames'])==51,
                'C real cp10, not cp5 or imagined state')
        else:
            require(row['plan']['commands'] is row['final_state'] is row['native_success'] is None
                and row['own_attempted']==row['PNG']==0,'C no_witness remains null')
    return dict(authorities=d,trust='MAIN pinned complete receipt, SHA closure and whole row agreement; no C fullstrict rerun',
        logical_C_slots=[dict(root_slot=x['root_slot'],arm=x['arm'],status=x['status'],
            endpoint=None if i else deepcopy(x['checkpoints'][-1])) for i,x in enumerate(u['rows'])],
        inherited_counts=deepcopy(u['counts']),inherited_PNG=66,new_C_dispatch=0,**FLAGS)


def assemble(cp,shared,C_reference):
    """Pure saved seam. All physical references are inherited, controls are literal."""
    require(cp['schema']==CNAME+'.prepare' and len(cp['plans'])==len(cp['prefix_reference'])==4
        and shared['schema']==SHARED+'.prepare' and len(shared['layers'])==2,'original fixed two roots')
    plans=[];refs=[]
    for i,(root,role) in enumerate(ORDER):
        base=cp['plans'][2*root];other=cp['plans'][2*root+1]
        ref=deepcopy(cp['prefix_reference'][2*root]);ref['index']=i
        t=base['model_task'];pro=base['provenance'];layer=shared['layers'][root]
        require(all(bridge.typed.same(base[k],other[k]) for k in
            ('model_task','base_task','provenance','prefix_commands','native_goal','native_goal_pose','native_root_sha256','native_goal_pixel_sha256')),
            'H1/H3 same original root goal history, not a new cohort')
        original=f32(layer['commands'],(10,2));prefix=f32(pro['prefix5'],(5,2))
        require(layer['arm']=='line_search_GN' and layer['source_slot']==root
            and layer['selection_sha256']==SELECTIONS[root] and layer['task_id']==base['task_id']
            and cp['bindings'].get(layer['task_export'])==layer['task_sha256']==base['base_task']['binding']['task_sha256']
            and prefix.tobytes()==original[:5].tobytes()==f32(base['prefix_commands'],(5,2)).tobytes()
            ==f32(ref['prefix_raw5'],(5,2)).tobytes(),'P exact original raw10/task/selection and cp5 prefix')
        control=np.zeros((5,2),dtype='<f4') if role=='Z' else original[5:].copy()
        plan={k:deepcopy(base[k]) for k in ('root_slot','source_slot','task_id','local_task_id',
            'model_task','base_task','provenance','prefix_commands','native_goal','native_goal_pose',
            'native_root_sha256','native_goal_pixel_sha256')}
        plan.update(schema=NAME+'.execution_plan',index=i,condition='off',arm=role,control_role=role,
            raw_steps=10,commands=np.concatenate((prefix,control)).tolist(),control_raw5=control.tolist(),
            control_float32_sha256=bytehash(control),control_source=dict(role=role,
                definition='positive_float32_zero' if role=='Z' else 'original_line_search_GN_commands_5_to_10',
                path=authority_descriptors()['P_prepare']['path'],
                sha256=authority_descriptors()['P_prepare']['sha256'],
                pointer='/layers/'+str(root)+'/commands',selection_sha256=SELECTIONS[root],
                task_export=layer['task_export'],task_export_sha256=layer['task_sha256'],
                original_raw10=original.tolist()),
            prefix_reference_sha256=digest(ref),selection_uses_physics=False,**FLAGS)
        plan['raw_float32_sha256']=bytehash(f32(plan['commands'],(10,2)))
        plan['selection_digest']=digest(plan);plans.append(plan);refs.append(ref)
    p=dict(schema=NAME+'.prepare',contract=contract(),plans=plans,prefix_reference=refs,
        planner_authority=deepcopy(C_reference),source_report_sha256=cp['source_report_sha256'],
        source_roots=deepcopy(cp['source_roots']),source_slots=deepcopy(cp['source_slots']),
        warm_traces=deepcopy(cp['warm_traces']),bindings={})
    validate_execution(p);return p


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract()
        and len(p['plans'])==len(p['prefix_reference'])==4
        and len(p['source_roots'])==len(p['source_slots'])==len(p['warm_traces'])==2
        and p['source_report_sha256']==native.SOURCE_SHA,'own fixed known2/four controls schema')
    require(p['planner_authority']['authorities']==authority_descriptors()
        and p['planner_authority']['new_C_dispatch']==0,'C fixed reference only')
    for i,(plan,ref,(root,role)) in enumerate(zip(p['plans'],p['prefix_reference'],ORDER,strict=True)):
        require(plan['schema']==NAME+'.execution_plan' and type(plan['index']) is type(plan['root_slot']) is int
            and (plan['index'],plan['root_slot'],plan['condition'],plan['arm'],plan['control_role'])==(i,root,'off',role,role)
            and plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})
            and plan['prefix_reference_sha256']==digest(ref) and ref['index']==i
            and 'model_delivery' not in plan and 'witness_status' not in plan,'own ordered controls not model witnesses')
        require(all(plan[k] is False for k in FLAGS) and plan['selection_uses_physics'] is False,'no scientific permission')
        t=bridge.typed.core.validate_task(plan['model_task']);pro=plan['provenance'];a=p['source_roots'][root]
        require(plan['task_id']==plan['base_task']['binding']['task_id'] and plan['local_task_id']==t['binding']['task_id']
            and bridge.typed.same(plan['base_task'],pro['original_task'])
            and bridge.typed.same(ref['actual_context'],t['root_context'])
            and ref['complete'] is True and len(ref['frames'])==51 and ref['checkpoint']['step']==5,
            'exact inherited actual cp5 task/context/reference')
        require(all(bridge.typed.same(plan[k],a[v]) for k,v in
            (('native_goal','goal'),('native_goal_pose','goal_pose'),('native_root_sha256','root_sha256')))
            and plan['native_goal_pixel_sha256']==a['goal_pixels']['off'],'original native root and goal')
        d=plan['control_source'];original=f32(d['original_raw10'],(10,2))
        expected=np.zeros((5,2),dtype='<f4') if role=='Z' else original[5:].copy()
        prefix=f32(plan['prefix_commands'],(5,2));commands=f32(plan['commands'],(10,2))
        require(d['role']==role and d['definition']==('positive_float32_zero' if role=='Z' else 'original_line_search_GN_commands_5_to_10')
            and d['path']==authority_descriptors()['P_prepare']['path'] and d['sha256']==authority_descriptors()['P_prepare']['sha256']
            and d['pointer']=='/layers/'+str(root)+'/commands' and d['selection_sha256']==SELECTIONS[root]
            and prefix.tobytes()==original[:5].tobytes()==f32(pro['prefix5'],(5,2)).tobytes()==f32(ref['prefix_raw5'],(5,2)).tobytes()
            and f32(plan['control_raw5'],(5,2)).tobytes()==expected.tobytes()
            and plan['control_float32_sha256']==bytehash(expected) and plan['raw_steps']==10
            and commands.tobytes()==np.concatenate((prefix,expected)).tobytes()
            and plan['raw_float32_sha256']==bytehash(commands),'literal once raw controls, including positive zero bits')


def validate_task_export(layer,base):
    loaded=task_loader.load_task(layer['task_export'],layer['task_sha256'])['gn_task']
    export=task_loader.read(layer['task_export'])
    require(export['schema']==task_loader.NAME+'.task_export'
        and export['source_slot']==layer['source_slot']
        and export['task_id']==layer['task_id']==base['binding']['task_id']
        and loaded['task_contract_sha256']==base['binding']['task_contract_sha256']
        and all(bridge.typed.same(loaded[k],base[k]) for k in
            ('root_context','goal','projection','action_mean','action_std')),
        'original task export body/contract; compiled binding is separate from GN metadata')


def material():
    """Saved identities only; MAIN complete C receipt is the explicit authority."""
    require(sha(PROTOCOL)==PROTOCOL_SHA and shape_census()['total_upper_bytes']<=CAP,'protocol/census frozen')
    descriptors=authority_descriptors();values={}
    for key,d in descriptors.items():
        require(sha(d['path'])==d['sha256'],'fixed authority bytes '+key)
        if key!='C_source':values[key]=read(d['path'],META)
    cp,r,co,u=(values[k] for k in ('C_prepare','C_report','C_completion','C_receipt'))
    cr=validate_C(cp,r,co,u);sp=values['P_prepare'];su=values['P_receipt']
    require(su['schema']==SHARED+'.consume' and su['status']=='saved_shared_panel_verified_NOT_admitted'
        and su['prepared']==descriptors['P_prepare']['path'] and su['prepare_sha256']==descriptors['P_prepare']['sha256']
        and su['scientific_admission'] is False,'original P full receipt authority')
    for root,layer in enumerate(sp['layers']):
        validate_task_export(layer,cp['plans'][2*root]['base_task'])
    p=assemble(cp,sp,cr);data=native.source_material()
    bindings=deepcopy(cp['bindings']);native.merge(bindings,u['bindings']);native.merge(bindings,data['bindings'])
    for d in descriptors.values():native.merge(bindings,{d['path']:d['sha256']})
    for f in [*own_files(),PROTOCOL,*[Path(m.__file__).resolve() for m in
        (native,native.allmode,native.base,source,moving,paired,primitives,bridge,bridge.typed,task_loader)]]:
        native.merge(bindings,{str(f):sha(f)})
    p['bindings']=bindings
    for f,h in bindings.items():require(sha(f)==h,'entire original and new source/artifact closure')
    require(len(raw(p))<=META,'prepare full context cap; no precision/denominator reduction')
    return p,data


def write_new(path,value):
    p=Path(native.absolute(str(path)));body=raw(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink()
        and len(body)<=META and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'exclusive new own metadata')
    with p.open('xb') as f:require(f.write(body)==len(body),'complete metadata');f.flush();os.fsync(f.fileno())
    paired.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p),bytes=len(body))


def prepare(output):
    require(not Path(output).exists(),'no overwrite before saved reconstruction');p,_=material();return write_new(output,p)


def verify(prepared,expected):
    native.actual_hash(expected);require(sha(prepared)==expected,'actual own prepare SHA')
    p=read(prepared,META);rebuilt,data=material()
    require(raw(rebuilt)==raw(p) and sha(prepared)==expected,'whole own prepare/source reconstruction');return p,data


def guard_proof(row,ref,plan,receipts):
    require(plan['control_role'] in ('Z','P'),'fixed control role before commit')
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


def endpoint_rows(rows):
    """Only the actual cp10 frame is an endpoint; early-stop finals stay separate."""
    result=[]
    for row in rows:
        frames=[x for x in row['checkpoints'] if x['step']==10]
        require(len(frames)<=1,'unique cp10')
        available=row['status']=='horizon_completed' and row['own_returned']==10 and len(frames)==1
        # A genuine native stop at exactly step10 is also a measured cp10.
        available=available or (row['status']=='native_stopped' and row['own_returned']==10 and len(frames)==1)
        result.append(dict(index=row['index'],root_slot=row['root_slot'],role=row['arm'],
            control_step=10,absolute_step=20,available=available,
            checkpoint=deepcopy(frames[0]) if available else None,
            missing_reason=None if available else row['status']))
    return result


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
    settled=all(x['status'] in ('horizon_completed','native_stopped') for x in rows)
    status=RECORDED if not errors and inputs_ok and settled else 'partial_or_failed'
    try:store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
    except BaseException as exc:fail('final_journal',exc);status='partial_or_failed'
    report=dict(schema=NAME+'.report',status=status,contract=contract(),execution_binding=binding,rows=rows,errors=errors,
        counts=deepcopy(meter.counts),actual_operation_receipts=deepcopy(meter.receipts),inputs_final_verified=inputs_ok,
        fixed_denominator=4,known_roots=2,fresh_roots=0,inherited=0,new_neural_calls=0,new_independent_samples=0,
        endpoints=endpoint_rows(rows),
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
    require(r['endpoints']==endpoint_rows(r['rows']),'all4 same-time endpoints, no early terminal substitution')
    closed=audit_events(r,events,p)
    for i,(row,plan,ref) in enumerate(zip(r['rows'],p['plans'],p['prefix_reference'],strict=True)):
        require(row['index']==i and row['plan']==plan and all(row[k]==plan[k] for k in
            ('root_slot','condition','arm','task_id','selection_digest')) and row['status'] in
            ('not_started','engineering_failed','native_stopped','horizon_completed'),'all4 literal plans/row identities')
        scoped=[e for e in events if e.get('branch')==i];ops=[x for x in r['actual_operation_receipts'] if x['branch']==i]
        observations=[e for e in scoped if e['kind']=='fresh_observation'];transitions=[e for e in scoped if e['kind']=='returned_transition']
        responses=[e['response'] for e in scoped if e['kind']=='own_response']
        require(responses==row['responses'] and row['PNG']==len(observations),'all actual native response/render census')
        require(plan['control_role'] in ('Z','P'),'only fixed controls execute')
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
            and all(x['status'] in ('horizon_completed','native_stopped') for x in r['rows']),'all four settled denominator only')
        c={k:v['completed'] for k,v in r['counts'].items()};executed=4
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
    entries=list(directory.iterdir())
    require(len(entries)<=shape_census()['PNG_count']+5 and all(p.is_file() and not p.is_symlink() for p in entries)
        and sum(p.stat().st_size for p in entries)<=CAP,'flat aggregate cap BEFORE reading artifacts')
    files={}
    for p in entries:
        require(p.stat().st_size<=CAP,'bounded regular artifacts');files[p.name]=p.read_bytes()
    require(sum(map(len,files.values()))<=CAP and hashlib.sha256(files['report.json']).hexdigest()==expected,'own full output cap/report SHA')
    for key in ('report.json','execution_binding.json','completion.json'):
        require(files[key]==raw(native.decode(files[key])),'canonical metadata bytes '+key)
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
        completion_sha256=completion_sha,completion=completion,shape_census=census,endpoints=r['endpoints'],
        actual_serialization_census=dict(PNG_count=n,
            PNG_bytes=sum(len(v) for k,v in files.items() if k.endswith('.png')),
            metadata_bytes=sum(len(v) for k,v in files.items() if not k.endswith('.png')),
            total_bytes=sum(map(len,files.values())),
            operation_receipts=len(r['actual_operation_receipts']),
            operation_journal_events=sum(e['kind'] in ('operation_dispatch_intent','operation_returned','operation_failed') for e in events),
            observation_journal_events=sum(e['kind']=='fresh_observation' for e in events),
            branch_closures=sum(e['kind']=='branch_closed' for e in events)),
        saved_PNG_verified=n,new_neural_calls=0,new_physics_calls=0,**FLAGS)



def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True);q.add_argument('--fixed-authorities',action='store_true',required=True)
    for cmd in ('verify','run','consume'):
        q=sub.add_parser(cmd)
        if cmd!='consume':q.add_argument('--prepared',required=True);q.add_argument('--prepare-sha',required=True)
        if cmd in ('run','consume'):q.add_argument('--output',required=True)
        if cmd=='run':q.add_argument('--authorization',required=True)
        if cmd=='consume':
            q.add_argument('--report-sha',required=True);q.add_argument('--completion-sha',required=True);q.add_argument('--receipt',required=True)
    a=p.parse_args(argv)
    if a.cmd=='prepare':result=prepare(a.output)
    elif a.cmd=='verify':verify(a.prepared,a.prepare_sha);result=dict(verified=True,execution_authorized=False)
    elif a.cmd=='consume':result=write_new(a.receipt,consume(a.output,a.report_sha,a.completion_sha))
    else:
        require(a.authorization==AUTH,'MAIN authority before physics imports')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
        out=Path(a.output)
        result=dict(output=str(out),status=r['status'],counts=r['counts'],
            report_sha256=sha(out/'report.json') if (out/'report.json').is_file() else None,
            completion_sha256=sha(out/'completion.json') if (out/'completion.json').is_file() else None)
    print(source.canonical(result));return 2 if result.get('status')=='partial_or_failed' else 0


if __name__=='__main__':raise SystemExit(main())

