"""Fixed7 H1 signals / fixed5 bounded U probes. Only MAIN run imports torch.

Own saved transport; original native buffers, solver, chart and Q. Upstream
RGB/physics strict receipts are explicit inherited authorities, never rerun.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native
from experiments.spectral_reachability import agent_p3_parent_r_native_components_20261029_v1 as components

core, signal = native.core, native.signal
require, raw, digest, sha = core.require, core.canonical, core.digest, signal.sha
ROOT = core.ROOT
NAME = 'agent_p3_same_time_capability_probe_cuda_20261029_v1'
BRIDGE = 'agent_p3_same_time_cp10_rgb_bridge_20261029_v1'
BRIDGE_SOURCE_SHA = '40d51e4d5ad5fce5db9417fecc639e0288f286b678a5e7b0eb4e22adea392b77'
PROTOCOL = ROOT/'innerdocs/agent_p3_secant_same_time_capability_protocol_20261029_v1.md'
PROTOCOL_SHA = 'b35af390fc74e6ebd44a9879784308ec74947fb28f640ce5e795cc263e65dccf'
AUTH = 'MAIN_APPROVED_SAME_TIME_CAPABILITY_HOST_CUDA'
ORDER = [(0, 'C'), (0, 'Z'), (0, 'P'), (1, 'Z'), (1, 'P')]
FLAGS = dict(core.FLAGS, scientific_admission=False, equal_FLOPs=False, fresh=False,
             C3=False, P1=False, P3=False, training=False, execution_authorized=False)
MiB = 1024**2
LIMITS = dict(output_bytes=2*1024**3, final_reserve_bytes=256*MiB,
              disk_floor_bytes=200*MiB, wall_seconds=1800,
              cuda_allocated_bytes=3*1024**3, cuda_reserved_bytes=4*1024**3)
FILE_CAPS = {'binding.json':32*MiB, 'native_seal.json':8*MiB, 'partial.json':8*MiB,
             'classes.json':16*MiB, 'report.json':96*MiB, 'completion.json':8*MiB,
             'publication_failure.json':MiB}
SPECIAL = dict(native_call_attempt=MiB, native_call_return=128*1024,
               signal_return=256*1024, probe_return=256*1024)
TOP = {'denominator', 'signal_request', 'signal_return', 'probe_request', 'probe_return',
       'native_call_attempt', 'native_call_return', 'native_call_failure',
       'native_sealed', 'metrics', 'execution_failure', 'execution_end', 'closure'}
ARRAYS = native.CALL_ARRAYS
RES = native.RESOURCE


def caps():
    v = native.live.predict_axes(24)
    v.update(jacobian_API=7, VJP_directions=1344, grad_API=1344, raw_backward_callbacks=1344,
             signal_compile_calls=7, solve_calls=7, array_SVD_calls=7,
             program_compile_calls=5, replay_calls=10, metric_calls=22)
    return v


def shape_census():
    # Every stored duplicate: AE input/output, predict input/output, predictor
    # input/output, flattened pred_proj input/output; no unique-buffer discount.
    forward = {'AE_input':[1,3,10], 'AE_output':[1,3,192],
               'predict_latent':[1,3,192], 'predict_action':[1,3,192], 'predict_output':[1,3,192],
               'predictor_latent':[1,3,192], 'predictor_action':[1,3,192], 'predictor_output':[1,3,192],
               'pred_proj_input':[3,192], 'pred_proj_output':[3,192]}
    io = sum(int(np.prod(s)) for s in forward.values())
    origin, packet, boundary = 576+20+10+2+2, 10+10+30, 576+20
    grad = 192+10+192+10+10  # output/raw/basis, grad return, raw-backward return
    signal_floats = 2*(io+origin+packet+boundary)+192+192+192*10+192*grad
    replay_floats = io+(576+20+10+2+2+10+30)+boundary
    # Exact maximum complete resource count, before failure unwind margin.
    operations = 7*(1+8+1+192*3)+7*2+5+10*5+22
    # Native array/direction events: 7*(25+192*4)+10*10 = 5651.
    # Resource events 8386; 34 invocation markers; <100 application records.
    common = 16000*64*1024
    large = 17*MiB+17*128*1024+7*256*1024+5*256*1024
    working = common+large+FILE_CAPS['binding.json']+FILE_CAPS['classes.json']
    final = LIMITS['final_reserve_bytes']
    return dict(forward_shapes=forward, dtype='float32', full_J_shape=[192,10],
                signal_count=7, VJP_directions=1344, grad_IO_shapes=[[192],[10],[192],[10],[10]],
                repeated_forward_IO_floats=io, per_signal_array_bytes=4*signal_floats,
                per_replay_array_bytes=4*replay_floats,
                all_repeated_native_array_bytes=4*(7*signal_floats+10*replay_floats),
                resource_operations=operations, resource_records=2*operations,
                normal_record_cap=64*1024, normal_records_cap=16000,
                recovery_records_cap=64, recovery_record_cap=MiB,
                working_publication_upper_bytes=working,
                final_file_upper_bytes=sum(FILE_CAPS[k] for k in ('native_seal.json','partial.json','report.json','completion.json','publication_failure.json')),
                final_upper_including_recovery_bytes=sum(FILE_CAPS[k] for k in ('native_seal.json','partial.json','report.json','completion.json','publication_failure.json'))+64*MiB,
                reserved_bytes=final, whole_upper_bytes=working+final,
                fits=working+final<=LIMITS['output_bytes'])


def contract():
    return dict(schema=NAME+'.contract', endpoint_order=[list(x) for x in ORDER],
                signals=['cp5/root0','cp5/root1']+[f'cp10/root{r}/{a}' for r,a in ORDER],
                resource_caps=caps(), AD_primal_predict_API=7, Jacobian_elements=13440,
                actual_Q=5, nominal_Q=7, replay_Q=10, Q_calls=22,
                limits=LIMITS, shape_census=shape_census(),
                solver_sha256=components.kernel.SOLVER_SHA256,
                damping=components.kernel.original.DAMPING, cap=components.kernel.original.CAP,
                chart='components.suffix_raw; float64 tanh then one float32 cast; native normalizes once',
                replay='independent whole and segmented H1 [5], exact full context',
                order='all7 signal/solve then all5 U/double replay durable seal, then22 original Q',
                protocol_sha256=PROTOCOL_SHA, known_roots=2, independent_roots=0,
                encoder=0, projector=0, physics=0, FLOPs=None, backward_engine_calls=None,
                upstream='MAIN full RGB/physical receipts inherited; no nested production fullstrict', **FLAGS)


def absolute(path):
    p = Path(path)
    require(p.is_absolute() and p.resolve()==p and not p.is_symlink(), 'canonical absolute path')
    return p


def read(path, expected=None, cap=96*MiB):
    p=absolute(path)
    require(p.is_file() and p.stat().st_size<=cap, 'bounded regular metadata')
    if expected is not None:
        core.hash_value(expected); require(sha(p)==expected, 'external byte SHA '+p.name)
    x=signal.read(p,cap); raw(x)
    return x


def false_flags(value):
    for k in ('P1_admission','P3_admission','physical_release','scientific_admission','full_C3_demonstrated',
              'equal_FLOPs','equal_total_compute','execution_authorized','freshness','independent_confirmation','training'):
        require(value[k] is False, 'literal upstream scientific false '+k)


def assemble(bp, br, bc, bs, authority, tasks):
    """Actual Jason ABI projection; no model, encoder or historical consumer."""
    for x,suffix in ((bp,'prepare'),(br,'report'),(bc,'completion'),(bs,'consume')):
        require(x['schema']==BRIDGE+'.'+suffix, 'own bridge schema '+suffix)
    state='all5_cp10_RGB_and_actual_history_tasks_sealed_NOT_admitted'
    require(br['status']==bs['status']==state and br['errors']==br['blocked']==[], 'complete actual five bridge')
    require(br['prepare_sha256']==authority['prepare']['sha256'] and br['prepared']==authority['prepare']['path']
            and bs['report_sha256']==bc['report_sha256']==authority['report']['sha256']
            and bs['completion_sha256']==authority['completion']['sha256'] and bs['completion']==bc,
            'four external bridge authority joins')
    require(bc['status']=='published_within_cap' and bc['prepared']==br['prepared']
            and bc['prepare_sha256']==br['prepare_sha256'], 'real bridge publication')
    require(bs['fixed_denominator']==br['fixed_denominator']==5 and bs['known_roots']==2 and
            bs['fresh_roots']==0 and bs['consume_new_neural_calls']==bs['consume_new_physics']==0,
            'inherited completed five, not new roots or repeated encoding')
    require(br['model_unchanged'] is br['source_verified'] is True and
            br['model_hash_before']==br['model_hash_after']==bp['expected_model_hash'], 'frozen bridge model')
    require(bp['contract']==br['contract'] and br['counts']==bs['counts'] and
            bs['counts']=={k:dict(attempted=5 if k in ('encoder','projector') else 0,
                                 completed=5 if k in ('encoder','projector') else 0)
                          for k in ('encoder','projector','action_encoder','predict_API','predictor','pred_proj')},
            'full inherited encoder vector, no new neural acquisition')
    for x in (br,bs,bc): false_flags(x)
    require(tasks==bs['tasks'] and len(tasks)==len(bp['rows'])==len(br['rows'])==5, 'whole five actual tasks')
    cp5={}; out=[]
    for i,((root,role),spec,item,encoded) in enumerate(zip(ORDER,bp['rows'],tasks,br['rows'],strict=True)):
        require(spec['index']==item['index']==i and spec['root_slot']==item['root_slot']==root
                and spec['role']==item['role']==role and encoded['status']=='encoded', 'fixed five order')
        t=core.validate_task(item['task']); old=core.validate_task(spec['cp5_task'])
        require(item['task_sha256']==digest(t) and item['old_cp5_task_id']==old['binding']['task_id']
                and item['base_task_id']==spec['base_task_id'], 'base/cp5/new endpoint identities')
        require(t['binding']['task_id']!=old['binding']['task_id'] and item['absolute_steps']==[10,15,20], 'actual cp10 namespace/chronology')
        ident=item['identity']
        require(ident['namespace']==BRIDGE and ident['index']==i and ident['role']==role
                and ident['root_slot']==root and ident['old_cp5_task_sha256']==digest(old)
                and ident['history_sources']==spec['history_sources']
                and ident['source_report']==spec['source_report'] and ident['source_receipt']==spec['source_receipt'], 'actual endpoint source projection')
        require(item['source']==spec['source'] and encoded['source']==spec['source']
                and ident['encoded_latent']==encoded['latent'], 'original real cp10 latent/PNG pointer')
        incoming=core.finite(spec['causal_raw10'],(10,2),True).astype('<f4')
        require(incoming.tobytes().hex()==spec['causal_raw10_bits'] and
                item['causal_raw10']==spec['causal_raw10'] and
                ident['raw10_float32_sha256']==hashlib.sha256(incoming.tobytes()).hexdigest(), 'literal physical incoming raw10')
        token=((incoming-np.asarray(old['action_mean'],np.float32))/np.asarray(old['action_std'],np.float32)).reshape(2,10)
        require(token.tobytes()==np.asarray(t['root_context']['action_history'],'<f4').tobytes()
                and token.tolist()==spec['normalized_action_history'], 'original normalization once')
        require(t['root_context']['latent_history'][:2]==old['root_context']['latent_history'][1:]
                and item['source']['state']==item['source']['snapshot']['state'], 'true inherited latent history and same-state source')
        require(ident['absolute_steps']==[10,15,20] and spec['absolute_steps']==[10,15,20]
                and [h['absolute_step'] for h in ident['history_sources']]==[10,15,20], 'all three real history frame times')
        expected_task=deepcopy(old); expected_task['root_context']=deepcopy(t['root_context'])
        expected_task['binding'].update(context_sha256=digest(t['root_context']),
            task_id=digest([old['binding']['task_id'],ident]),
            task_sha256=digest([old,t['root_context'],ident]),task_contract_sha256=digest([BRIDGE,ident]))
        require(raw(t)==raw(expected_task), 'exact new namespace and every inherited binding bit')
        for k in ('goal','projection','action_mean','action_std'):
            require(t[k]==old[k], 'unchanged original '+k)
        require(np.array_equal(t['projection'],np.eye(192)) and len(t['goal'])==192
                and t['binding']['model_sha256']==old['binding']['model_sha256']==bp['expected_model_hash'], 'native identity192/model')
        if root in cp5: require(raw(cp5[root])==raw(old), 'same-root exact cp5 dedup authority')
        else: cp5[root]=old
        out.append(dict(index=i,root=root,role=role,task=t,task_sha256=digest(t),identity=deepcopy(ident),
                        source=deepcopy(item['source']),base_task_id=item['base_task_id'],
                        causal_raw10=deepcopy(item['causal_raw10']),causal_raw10_bits=spec['causal_raw10_bits']))
    signals=[dict(index=r,time=5,root=r,role='reference',task=cp5[r],task_sha256=digest(cp5[r])) for r in (0,1)]
    signals += [dict(index=i+2,time=10,**{k:e[k] for k in ('root','role','task','task_sha256')}) for i,e in enumerate(out)]
    return dict(signals=signals,endpoints=out)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build(authority):
    require(set(authority)=={'prepare','report','completion','receipt','source'}, 'explicit finalized five bridge references')
    require(sha(PROTOCOL)==PROTOCOL_SHA, 'fixed approved protocol')
    components.check_sources()
    require(sha(native.__file__)=='eeea5710d4171b7c43f99d1f2b5ad2412e03a25ba14e8fc5589ea5ef73f57b3b', 'sealed native runner primitives')
    require(absolute(authority['source']['path'])==ROOT/'experiments/spectral_reachability'/(BRIDGE+'.py'), 'real Jason source path')
    require(authority['source']['sha256']==BRIDGE_SOURCE_SHA, 'MAIN frozen Jason source SHA')
    for ref in authority.values():
        core.keys(ref,('path','sha256')); absolute(ref['path']); core.hash_value(ref['sha256'])
        require(sha(ref['path'])==ref['sha256'], 'bridge actual final SHA')
    bp,br,bc,bs=(read(**dict(path=authority[k]['path'],expected=authority[k]['sha256'])) for k in ('prepare','report','completion','receipt'))
    output=absolute(authority['report']['path']).parent
    tasks_path=output/'cp10_tasks.json'
    tasks=read(tasks_path,br['tasks_sha256'])
    require(br['artifact_sha256']['cp10_tasks.json']==br['tasks_sha256'] and
            bs['bindings'][str(tasks_path)]==br['tasks_sha256'], 'sealed actual task file')
    joined=assemble(bp,br,bc,bs,authority,tasks)
    bindings=deepcopy(bp['bindings'])
    for k,v in bs['bindings'].items():
        require(k not in bindings or bindings[k]==v, 'no conflicting inherited binding')
        bindings[k]=v
    bindings.update({ref['path']:ref['sha256'] for ref in authority.values()})
    # Explicit primitives, not the mutable set of modules already imported by a
    # test/CLI process. Bridge prepares bind the installed offline loader closure.
    files=set(own_files()+[PROTOCOL,ROOT/'experiments/checkpoint_compat.py',ROOT/'experiments/gpu_preflight.py'])
    for module in (native,signal,core,native.native,native.parent,native.live,components,
                   components.kernel,components.kernel.original,core.existing):
        files.add(Path(module.__file__).resolve())
    for f in files:
        h=sha(f);require(str(f) not in bindings or bindings[str(f)]==h, 'no source override of inherited identity')
        bindings[str(f)]=h
    require(shape_census()['fits'] and shape_census()['final_upper_including_recovery_bytes']<=LIMITS['final_reserve_bytes'], 'full publication proof')
    p=dict(schema=NAME+'.prepare',bridge_authority=deepcopy(authority),**joined,
           policy=bp['policy'],expected_model_hash=bp['expected_model_hash'],bindings=bindings,
           profile=deepcopy(LIMITS),contract=contract(),inherited_counts=deepcopy(bs['counts']),
           inherited_physical_counts=deepcopy(bs['inherited_physical_counts']),
           upstream_fullstrict_rebuilt_here=False,**FLAGS)
    check_bindings(p)
    require(len(raw(p))<=FILE_CAPS['binding.json'], 'bounded full prepared input')
    return p


def check_bindings(p):
    for f,h in p['bindings'].items():
        require(sha(absolute(f))==h, 'unchanged bound source/asset '+f)


def verify(prepared, expected):
    p=read(prepared,expected,FILE_CAPS['binding.json'])
    require(raw(p)==raw(build(p['bridge_authority'])), 'full prepare rebuild, no old consumer rerun')
    return p


def write_new(path,value,cap=96*MiB):
    path=absolute(path); body=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and len(body)<=cap, 'new bounded own output')
    with path.open('xb') as f:
        require(f.write(body)==len(body), 'full metadata write'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(path.parent)
    return dict(path=str(path),sha256=sha(path))


class Store:
    """Own2GiB hashchain, separate reserve, no inherited8/64MiB reader cap."""
    def __init__(self,output,*,clock=time.monotonic,free=None):
        self.output=absolute(output); self.clock=clock; self.started=clock(); self.free=free or (lambda:shutil.disk_usage(self.output.parent).free)
        require(self.free()>=LIMITS['output_bytes']+LIMITS['disk_floor_bytes'], 'whole allowance plus disk floor')
        self.output.mkdir(exist_ok=False); signal.fsync_dir(self.output.parent)
        self.stream=(self.output/'events.jsonl').open('xb'); self.index=0; self.tail='0'*64
        self.bytes=0; self.recovery=False; self.broken=False; self.final_events=0; self.memory=None; self.closed=False

    def check(self,size=0,final=False):
        require(self.bytes+size<=LIMITS['output_bytes']-(0 if final else LIMITS['final_reserve_bytes']), 'output cap/reserve')
        require(self.free()>=LIMITS['disk_floor_bytes']+size, 'disk floor')
        if not final:
            require(self.clock()-self.started<=LIMITS['wall_seconds'], 'whole wall cap')
            if self.memory:
                a,r=self.memory()
                require(a<=LIMITS['cuda_allocated_bytes'] and r<=LIMITS['cuda_reserved_bytes'], 'CUDA peak caps')

    def event(self,e):
        require(not self.closed, 'open journal')
        kind=e['kind']; recovery=self.recovery
        if recovery:
            require(kind in {'resource_failure','native_call_failure','execution_failure','execution_end','closure'}, 'only failure unwind after transport refusal')
            require(self.final_events<64, 'bounded reserve unwind'); self.final_events+=1
        rec=dict(index=self.index,previous=self.tail,payload=e); rec['sha256']=digest(rec)
        data=raw(rec)+b'\n'
        require(len(data)<=(MiB if recovery else SPECIAL.get(kind,64*1024)), 'per-kind record shape bound')
        try:
            require(recovery or self.index<16000, 'predeclared record denominator bound')
            self.check(len(data),final=recovery)
            require(self.stream.write(data)==len(data), 'complete record write')
            self.stream.flush(); os.fsync(self.stream.fileno())
        except BaseException:
            self.recovery=True; self.broken=True
            raise
        self.bytes+=len(data); self.index+=1; self.tail=rec['sha256']

    def put(self,name,value,final=False):
        require(name in FILE_CAPS and not (self.output/name).exists(), 'new own artifact only')
        data=raw(value); require(len(data)<=FILE_CAPS[name], 'final/working file quota '+name)
        self.check(len(data),final)
        with (self.output/name).open('xb') as f:
            require(f.write(data)==len(data), 'complete artifact'); f.flush(); os.fsync(f.fileno())
        self.bytes+=len(data); signal.fsync_dir(self.output)
        return hashlib.sha256(data).hexdigest()

    def close(self):
        if not self.closed:
            self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close(); self.closed=True
            signal.fsync_dir(self.output)


def endpoint(task, nominal, actions):
    c=deepcopy(task['root_context']); a=np.asarray(actions,np.float32)
    token=((a-np.asarray(task['action_mean'],np.float32))/np.asarray(task['action_std'],np.float32)).reshape(10)
    c['latent_history']=deepcopy(c['latent_history'][1:])+[list(nominal)]
    c['action_history']=[deepcopy(c['action_history'][-1]),token.tolist()]
    core.checked_path(task,task['root_context'],a.tolist(),[c]); return c


class Calls:
    def __init__(self,factory,store,guard,kind):
        self.factory,self.store,self.guard,self.kind=factory,store,guard,kind; self.index=0

    def invoke(self,task,method,request,ledger):
        self.guard(); idx=self.index; self.index+=1
        self.store.event(dict(kind='native_call_attempt',call_id=idx,backend=self.kind,
                              task=deepcopy(task),method=method,request=deepcopy(request)))
        try:
            backend=self.factory(task,self.store)
            require(backend.kind==self.kind, 'no implicit RAM backend')
            if method=='signal': result=backend.signal(request,ledger)
            else: result=backend.replay(request['parent_context'],request['actions'],request['segmented'],ledger)
            self.guard()
            self.store.event(dict(kind='native_call_return',call_id=idx,result=deepcopy(result)))
            return result
        except BaseException as e:
            try:self.store.event(dict(kind='native_call_failure',call_id=idx,error=dict(type=type(e).__name__,message=str(e))))
            except BaseException: self.store.broken=True
            raise


def execute(p,factory,store,*,permit_ram=False,guard=lambda:None):
    """The same fixed-loop control flow drives HOST and explicit RAM tests."""
    signals=[dict(index=i,status='not_requested',response=None,solve=None,nominal_context=None) for i in range(7)]
    probes=[dict(index=i,status='not_requested',raw=None,whole=None,segmented=None) for i in range(5)]
    metrics=[]; errors=[]; seal=None
    ledger=core.Ledger(caps(),sink=store.event)
    calls=Calls(factory,store,guard,'RAM_FIXTURE' if permit_ram else 'HOST_CUDA')
    store.event(dict(kind='denominator',signals=7,probes=5,metric_slots=22,known_roots=2,backend=calls.kind))
    try:
        require(len(p['signals'])==7 and len(p['endpoints'])==5, 'fixed seven/five inputs')
        for i,(spec,row) in enumerate(zip(p['signals'],signals,strict=True)):
            task=core.validate_task(spec['task']); require(spec['index']==i and digest(task)==spec['task_sha256'], 'task exact input digest')
            ledger.scope=dict(stage='signal',index=i,task_sha256=spec['task_sha256'])
            row['status']='attempted'; store.event(dict(kind='signal_request',index=i,task_sha256=spec['task_sha256']))
            req=core.request(task,dict(context=task['root_context']),operation='compile_finite_horizon_J',horizon=1)
            response=ledger.call('signal_compile',lambda:calls.invoke(task,'signal',req,ledger))
            core.finite(response['nominal'],(192,),True); core.finite(response['jacobian'],(192,10),True)
            require(response['binding_sha256']==req['binding_sha256'] and response['parent_context_sha256']==req['parent_context_sha256']
                    and response['horizon']==1 and response['producer_sha256']==sha(signal.__file__), 'original native signal binding')
            j=np.asarray(response['jacobian'],np.float64); residual=np.asarray(task['goal'],np.float64)-np.asarray(response['nominal'],np.float64)
            solve=ledger.call('solve',lambda:ledger.call('array_svd',lambda:components.kernel.original.bounded_ridge(j,residual)))
            require(len(solve['params'])==len(solve['singular_values'])==10, 'complete original solve/spectrum')
            row.update(response=response,solve=solve,nominal_context=endpoint(task,response['nominal'],[[0.,0.]]*5),status='complete')
            store.event(dict(kind='signal_return',row=deepcopy(row)))
        for i,(spec,row) in enumerate(zip(p['endpoints'],probes,strict=True)):
            guard(); task=spec['task']; require(task==p['signals'][i+2]['task'], 'same cp10 for U')
            ledger.scope=dict(stage='probe',index=i,task_sha256=spec['task_sha256'])
            row['status']='attempted'; store.event(dict(kind='probe_request',index=i,signal_index=i+2))
            row['raw']=ledger.call('program_compile',lambda:components.suffix_raw(signals[i+2]['solve']['params']))
            for field,segmented in (('whole',False),('segmented',True)):
                req=dict(parent_context=task['root_context'],actions=row['raw'],segmented=segmented)
                row[field]=ledger.call('replay',lambda:calls.invoke(task,'replay',req,ledger))
                core.checked_path(task,task['root_context'],row['raw'],row[field])
            require(raw(row['whole'])==raw(row['segmented']), 'independent dual H1 exact full native context')
            row['status']='complete'; store.event(dict(kind='probe_return',row=deepcopy(row)))
        # No metric callback before this fsynced file + journal publication.
        seal=store.put('native_seal.json',dict(signals=signals,probes=probes),final=True)
        store.event(dict(kind='native_sealed',sha256=seal,signals=7,probes=5,replays=10))
        inputs=[('actual',i,e['task'],e['task']['root_context']) for i,e in enumerate(p['endpoints'])]
        inputs += [('nominal',i,s['task'],signals[i]['nominal_context']) for i,s in enumerate(p['signals'])]
        inputs += [(field,i,e['task'],probes[i][field][-1]) for i,e in enumerate(p['endpoints']) for field in ('whole','segmented')]
        for kind,i,task,c in inputs:
            guard(); ledger.scope=dict(stage='Q',kind=kind,index=i)
            def metric():
                core.context(c)
                r=np.asarray(task['projection'],np.float64)@(np.asarray(c['latent_history'][-1],np.float64)-np.asarray(task['goal'],np.float64))
                return float(np.mean(r**2))
            value=ledger.call('metric',metric); require(np.isfinite(value), 'original finite Q')
            metrics.append(dict(kind=kind,index=i,Q=value))
        store.event(dict(kind='metrics',rows=deepcopy(metrics)))
        require(ledger.attempted==ledger.completed==caps(), 'complete actual vector, no refunds')
    except BaseException as e:
        errors.append(dict(type=type(e).__name__,message=str(e)))
        for row in signals+probes:
            if row['status']=='attempted':row['status']='failed'
            elif row['status']=='not_requested':row['status']='blocked_after_failure'
        try:store.event(dict(kind='execution_failure',error=errors[-1]))
        except BaseException:store.broken=True
    status='complete_NOT_admitted' if not errors else 'partial_or_failed'
    store.event(dict(kind='execution_end',status=status,signal_status=[s['status'] for s in signals],probe_status=[x['status'] for x in probes]))
    return dict(status=status,signals=signals,probes=probes,metrics=metrics,errors=errors,
                native_seal_sha256=seal,resource_receipt=ledger.receipt(),backend=calls.kind)


def journal(output):
    events=[]; tail='0'*64; size=0
    path=absolute(output)/'events.jsonl'
    require(path.stat().st_size<=LIMITS['output_bytes'], 'own bounded journal')
    with path.open('rb') as f:
        for i,line in enumerate(f):
            require(i<16064 and len(line)<=MiB and line.endswith(b'\n'), 'complete bounded record')
            x=json.loads(line,object_pairs_hook=signal.unique_pairs); h=x.pop('sha256')
            core.keys(x,('index','previous','payload'))
            require(x['index']==i and type(x['index']) is int and x['previous']==tail and digest(x)==h, 'journal hashchain/order')
            tail=h; events.append(x['payload']); size+=len(line)
    return events,tail


class Memory:
    def __init__(self):self.events=[];self.files={};self.broken=False
    def event(self,e):self.events.append(deepcopy(e))
    def put(self,n,v,final=False):self.files[n]=deepcopy(v);return digest(v)


def split_calls(events):
    calls=[]; active=None
    for e in events:
        kind=e['kind']
        require(kind in TOP|ARRAYS|RES|{'direction'}, 'unknown event, including after seal')
        if kind=='native_call_attempt':
            require(active is None and e['call_id']==len(calls), 'serial invocation order')
            active=[e]
        elif kind in ('native_call_return','native_call_failure'):
            require(active is not None and e['call_id']==len(calls), 'actual matching callback disposition')
            active.append(e); calls.append(active); active=None
        elif active is not None: active.append(e)
        elif kind in ARRAYS|{'direction'}: raise ValueError('native IO outside invocation')
    require(active is None, 'no fabricated closure for unfinished callback')
    return calls


def resource_census(receipt,events):
    """Original complete census, plus explicit non-success native failure prefix.

    A native output can already be durably saved when its resource-return
    publication is refused. Both bookends then exist, but the operation remains
    FAILED, never refunded or relabelled returned.
    """
    core.audit_ledger(receipt)
    actual=[e for e in events if e['kind'] in RES]; expected=[]; stack=[]
    linked_by_index={}
    for e in events:
        if e.get('operation_index') is not None:linked_by_index.setdefault(e['operation_index'],[]).append(e)
    for e in actual:
        op=e['entry']
        if e['kind']=='resource_attempt':stack.append(op['index'])
        elif e['kind']!='resource_blocked':require(stack and stack.pop()==op['index'], 'proper native failure unwind')
    require(not stack, 'no undisposed resource intent')
    for op in receipt['events']:
        if op['status']=='blocked_before_dispatch':expected.append(dict(kind='resource_blocked',entry=op))
        else:
            expected += [dict(kind='resource_attempt',entry=dict(op,status='attempted',ended_ns=None,error=None)),
                         dict(kind='resource_return' if op['status']=='returned' else 'resource_failure',entry=op)]
        linked=linked_by_index.get(op['index'],[])
        names=['io_input','io_return'] if op['resource'] in ('predict','action_encoder','predictor','pred_proj','grad') else ['raw_backward'] if op['resource']=='raw_backward' else []
        if op['status']=='returned':require([e['kind'] for e in linked]==names, 'all paid native buffers')
        else:require([e['kind'] for e in linked] in [names[:i] for i in range(len(names)+1)], 'only available failure-prefix bookends')
        require(all(e['scope']==op['scope'] for e in linked), 'resource and native IO scope')
    require(sorted(map(raw,actual))==sorted(map(raw,expected)), 'all actual attempts/returns/failures, no refunds')
    valid_indices={e['index'] for e in receipt['events']}
    for e in events:
        if e.get('operation_index') is not None:require(e['operation_index'] in valid_indices, 'no orphan native IO')
        if 'arrays' in e:
            if e['kind'] not in ('io_input','io_return'):
                core.keys(e['arrays'],{'origin':('latent','history','raw_flat','mean','std'),
                    'packet':('raw','token','outgoing'),'boundary':('latent','history'),
                    'nominal':('value',),'AD_terminal':('value',),'jacobian':('value',),
                    'raw_backward':('value',),'program_packet':('latent','history','raw','mean','std','token','outgoing'),
                    'program_boundary':('latent','history')}[e['kind']])
            else:
                op=receipt['events'][e['operation_index']]; resource=op['resource']
                keys={'predict':('latent','act_emb'),'action_encoder':('0',),
                      'predictor':('0','1'),'pred_proj':('0',),'grad':('output','raw','basis')}[resource]
                core.keys(e['arrays'],(resource+':output',) if e['kind']=='io_return' else tuple(resource+':'+k for k in keys))
            for b in e['arrays'].values():
                a=signal.unblob(b)
                require(a.dtype==np.float32 and list(a.shape) in ([1,3,192],[1,2,10],[1,3,10],[3,192],[5,2],[2],[10],[192],[192,10]), 'native full H1 shape and original precision')


class SavedNative:
    def __init__(self,task,state,kind):self.task=task;self.state=state;self.kind=kind
    def invoke(self,method,req,ledger):
        require(self.state['index']<len(self.state['calls']), 'no invented saved callback')
        call=self.state['calls'][self.state['index']]; self.state['index']+=1; a=call[0]
        require(a['task']==self.task and a['method']==method and raw(a['request'])==raw(req), 'exact saved call context/raw/method')
        native.replay_resources(call[1:-1],ledger)
        if call[-1]['kind']=='native_call_failure': raise native.exception_from(call[-1]['error'])
        return deepcopy(call[-1]['result'])
    def signal(self,req,ledger):return self.invoke('signal',req,ledger)
    def replay(self,c,actions,segmented,ledger):return self.invoke('replay',dict(parent_context=c,actions=actions,segmented=segmented),ledger)


def validate(p,result,events,seal,*,permit_fixture=False):
    require(result['backend']==('RAM_FIXTURE' if permit_fixture else 'HOST_CUDA'), 'explicit RAM vs actual HOST authority')
    core.audit_ledger(result['resource_receipt'])
    calls=split_calls(events)
    resource_census(result['resource_receipt'],events)
    for call in calls:
        require(call[0]['backend']==result['backend'], 'per-call backend binding')
        if call[-1]['kind']=='native_call_return':require(native.audit_native(call) is True, 'full saved native buffers')
        else:require(native.audit_native(call) is False, 'failure prefix never full native closure')
    state=dict(calls=calls,index=0); sink=Memory()
    rebuilt=execute(p,lambda t,s:SavedNative(t,state,result['backend']),sink,permit_ram=permit_fixture)
    require(state['index']==len(calls), 'all actual calls consumed exactly once')
    def stripped(e):return native.stripped(e)
    filtered=lambda es:[stripped(e) for e in es if e['kind'] not in ARRAYS|{'direction','closure'}]
    require(raw(filtered(events))==raw(filtered(sink.events)), 'exact fixed-loop chronology and resource calls')
    # Wall receipts are measured responses, not deterministic algebra outputs.
    for k in result:
        if k!='resource_receipt':require(raw(result[k])==raw(rebuilt[k]), 'independent saved field '+k)
    if result['status']=='complete_NOT_admitted':
        require(seal==dict(signals=result['signals'],probes=result['probes']) and digest(seal)==result['native_seal_sha256'], 'all7/5 durable seal before Q')
        require(len(calls)==17 and len(result['metrics'])==22, 'complete all-call/metric denominator')
        require(result['resource_receipt']['attempted']==result['resource_receipt']['completed']==caps(), 'all actual resources')
    else:require(result['errors'], 'partial cannot masquerade as valid full probes')
    return dict(native_calls=len(calls),saved_SVD_calls=sum(s['solve'] is not None for s in rebuilt['signals']),
                saved_new_neural_calls=0,saved_new_physics=0,full_native_closure=result['status']=='complete_NOT_admitted')


def publish(store,report):
    store.close(); report['journal_tail']=store.tail
    report['artifact_sha256']={p.name:sha(p) for p in store.output.iterdir()}
    report['transport_complete']=not store.broken
    store.put('report.json',report,final=True)
    report_sha=sha(store.output/'report.json')
    completion=dict(schema=NAME+'.completion',report_sha256=report_sha,
                    prepared=report['prepared'],prepare_sha256=report['prepare_sha256'],
                    inclusive_wall_seconds=store.clock()-store.started,
                    wall_scope='through report fsync, directory fsync and report hash; completion self-write excluded',
                    artifact_sha256={p.name:sha(p) for p in store.output.iterdir()},
                    bytes_before_completion=sum(p.stat().st_size for p in store.output.iterdir()),**FLAGS)
    store.put('completion.json',completion,final=True)
    # Actual guard after the final completion fsync. Failure is persisted and
    # accepted by no strict consumer; it never becomes a scientific/model failure.
    inclusive=store.clock()-store.started
    if inclusive>LIMITS['wall_seconds']:
        store.put('publication_failure.json',dict(stage='after_completion_fsync',elapsed_seconds=inclusive),final=True)
        raise ValueError('publication wall exceeded; report and failure preserved')
    return dict(output=str(store.output),report_sha256=report_sha,completion_sha256=sha(store.output/'completion.json'),inclusive_after_completion_fsync_seconds=inclusive)


def probe_plans(p,result,authority):
    require(result['status']=='complete_NOT_admitted', 'no partial probe plan publication')
    plans=[]
    for i,(spec,probe) in enumerate(zip(p['endpoints'],result['probes'],strict=True)):
        sig=result['signals'][i+2]; task=spec['task']
        values={(x['kind'],x['index']):x['Q'] for x in result['metrics']}
        a=np.asarray(probe['raw'],'<f4')
        plan=dict(schema=NAME+'.probe_plan',index=i,root_slot=spec['root'],role=spec['role'],
            checkpoint=10,probe_checkpoint=15,task=deepcopy(task),task_sha256=digest(task),
            endpoint_identity=deepcopy(spec['identity']),source=deepcopy(spec['source']),base_task_id=spec['base_task_id'],
            causal_raw10=deepcopy(spec['causal_raw10']),causal_raw10_bits=spec['causal_raw10_bits'],
            raw5=probe['raw'],raw_float32_bits=a.tobytes().hex(),raw_float32_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            params_float64=sig['solve']['params'],signal=deepcopy(sig['response']),solve=deepcopy(sig['solve']),
            nominal_context=deepcopy(sig['nominal_context']),whole=deepcopy(probe['whole']),segmented=deepcopy(probe['segmented']),
            Q=dict(actual_cp10=values['actual',i],nominal=values['nominal',i+2],
                   whole=values['whole',i],segmented=values['segmented',i]),
            native_call_ids=dict(signal=i+2,whole=7+2*i,segmented=8+2*i),
            authority=deepcopy(authority),bridge_authority=deepcopy(p['bridge_authority']),
            native_seal_sha256=result['native_seal_sha256'],**FLAGS)
        plan['witness_sha256']=digest(plan);plans.append(plan)
    return plans


def run(prepared,expected,output,authorization):
    require(authorization==AUTH, 'MAIN explicit source review and CUDA scheduling required')
    started=time.monotonic(); p=verify(prepared,expected); store=Store(output);store.started=started
    model=None; before=after=host=classes=classes_after=None; errors=[]; guards=[]; result=None; source_ok=False
    try:
        store.put('binding.json',dict(prepared=str(absolute(prepared)),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        store.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        store.check(); model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'], 'same frozen tensors')
        classes=native.model_sources(model,dict(sources=p['bindings']))
        def guard():
            torch.cuda.synchronize(device);store.check();native.checked_precision(torch)
        forwards=[(m,'forward' in m.__dict__,m.__dict__.get('forward')) for m in model.modules()]
        with torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False),sdpa_kernel(SDPBackend.MATH):
            guards.append(native.checked_precision(torch));guard()
            result=execute(p,lambda t,s:native.NativeTree(model,device,t,s),store,guard=guard)
            guard();guards.append(native.checked_precision(torch))
        require(all(('forward' in m.__dict__)==had and m.__dict__.get('forward') is old for m,had,old in forwards), 'instance hooks restored')
    except BaseException as e:errors.append(dict(stage='run',type=type(e).__name__,message=str(e)))
    finally:
        if model is not None:
            try:
                after=signal.model_hash(model);classes_after=native.model_sources(model,dict(sources=p['bindings']))
                require(before==after and classes==classes_after and all(not m.training for m in model.modules())
                        and all(not v.requires_grad for v in model.parameters()), 'all model weights/buffers/freeze unchanged')
            except BaseException as e:errors.append(dict(stage='model_final',message=str(e)))
        try:
            check_bindings(p);require(sha(prepared)==expected, 'prepare final identity');store.check(final=True);source_ok=True
        except BaseException as e:errors.append(dict(stage='source_final',message=str(e)))
        if result is None:errors.append(dict(stage='execution_absent',message='no completed execute return; no probe release'))
        store.put('classes.json',dict(before=classes,after=classes_after),final=True)
        store.event(dict(kind='closure',model_before=before,model_after=after,sources_verified=source_ok))
        report=dict(schema=NAME+'.report',status='complete_NOT_admitted' if result is not None and result['status']=='complete_NOT_admitted' and not errors else 'partial_or_failed',
                    prepared=str(absolute(prepared)),prepare_sha256=expected,result=result,errors=errors,
                    model_before=before,model_after=after,host=host,precision=guards,sources_verified=source_ok,
                    bindings_sha256=digest(p['bindings']),CUDA_peak_bytes=list(store.memory()) if store.memory else None,
                    elapsed_seconds=store.clock()-store.started,contract=contract(),**FLAGS)
        publication=publish(store,report)
    require(report['status']=='complete_NOT_admitted' and not store.broken, 'partial retained, no retry or physics release')
    return publication


def _consume(output,report_sha,completion_sha,prepared,prepare_sha,*,permit_fixture=False):
    output=absolute(output); files=list(output.iterdir())
    require({f.name for f in files}<=(set(FILE_CAPS)|{'events.jsonl'}), 'no unknown artifact')
    require(all(p.is_file() and not p.is_symlink() for p in files) and sum(p.stat().st_size for p in files)<=LIMITS['output_bytes'], 'all output files bounded')
    require(not (output/'publication_failure.json').exists(), 'failed final publication, no complete receipt')
    r=read(output/'report.json',report_sha);c=read(output/'completion.json',completion_sha,FILE_CAPS['completion.json'])
    p=verify(prepared,prepare_sha)
    require(r['schema']==NAME+'.report' and c['schema']==NAME+'.completion' and r['contract']==p['contract']==contract(), 'own report/contract')
    require(r['prepared']==c['prepared']==str(absolute(prepared)) and r['prepare_sha256']==c['prepare_sha256']==prepare_sha
            and c['report_sha256']==report_sha, 'explicit prepare/report/completion identity')
    require(all(r[k] is c[k] is False for k in FLAGS), 'all scientific flags remain false')
    require(c['wall_scope']=='through report fsync, directory fsync and report hash; completion self-write excluded'
            and 0<=r['elapsed_seconds']<=c['inclusive_wall_seconds']<=LIMITS['wall_seconds'], 'exact recorded wall boundary')
    require(c['artifact_sha256']=={f.name:sha(f) for f in files if f.name!='completion.json'}
            and c['bytes_before_completion']==sum(f.stat().st_size for f in files if f.name!='completion.json'), 'full publication SHA and bytes')
    require(set(r['artifact_sha256'])|{'report.json','completion.json'}=={f.name for f in files}
            and all(sha(output/n)==h for n,h in r['artifact_sha256'].items()), 'no ignored output artifacts')
    require(read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha), 'durable initial binding')
    require(r['transport_complete'] is r['sources_verified'] is True and r['bindings_sha256']==digest(p['bindings']), 'no dropped transport/source events')
    require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash'], 'frozen actual model')
    classes=read(output/'classes.json',cap=FILE_CAPS['classes.json'])
    require(classes['before']==classes['after'] and classes['before'], 'all installed model classes')
    for cl in classes['before']:require(p['bindings'][cl['path']]==cl['sha256'], 'installed class sealed source')
    require(r['precision']==[native.precision_contract()]*2, 'observed precision flags')
    if permit_fixture:
        require(r['host']==dict(device='RAM_FIXTURE_NOT_CUDA'), 'explicit inert fixture report')
    else:
        require(r['host']['device']=='cuda:0' and r['host']['allocation_sum']==4. and
                r['host']['visible_nvidia_nodes'], 'actual CUDA preflight')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=LIMITS[k]
            for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)), 'memory caps')
    events,tail=journal(output);require(tail==r['journal_tail'], 'report journal tail')
    require([e for e in events if e['kind']=='closure']==[dict(kind='closure',model_before=r['model_before'],model_after=r['model_after'],sources_verified=True)], 'final closure event')
    seal=read(output/'native_seal.json',cap=FILE_CAPS['native_seal.json']) if (output/'native_seal.json').exists() else None
    require(r['result'] is not None, 'no verified execution; metadata partial only')
    verification=validate(p,r['result'],events,seal,permit_fixture=permit_fixture)
    complete=r['result']['status']=='complete_NOT_admitted' and r['errors']==[]
    require((r['status']=='complete_NOT_admitted')==complete, 'complete vs partial exact')
    check_bindings(p);require(sha(output/'report.json')==report_sha and sha(output/'completion.json')==completion_sha, 'stable authority bytes after consume')
    return dict(schema=NAME+'.consume',status='RAM_fixture_only_NOT_evidence' if permit_fixture else 'saved_complete_NOT_admitted' if complete else 'saved_partial_NOT_admitted',
                prepared=str(absolute(prepared)),prepare_sha256=prepare_sha,report_sha256=report_sha,completion_sha256=completion_sha,
                signals=r['result']['signals'],probes=r['result']['probes'] if complete else [None]*5,metrics=r['result']['metrics'],
                resource_receipt=r['result']['resource_receipt'],verification=verification,
                fixed_signals=7,fixed_probes=5,plans=probe_plans(p,r['result'],dict(prepared=r['prepared'],prepare_sha256=prepare_sha,
                    report=str(output/'report.json'),report_sha256=report_sha,completion=str(output/'completion.json'),completion_sha256=completion_sha)) if complete else [None]*5,
                bridge_authority=p['bridge_authority'],inherited_counts=p['inherited_counts'],
                inherited_physical_counts=p['inherited_physical_counts'],upstream_fullstrict_rebuilt_here=False,
                unknowns=dict(FLOPs=None,backward_engine_calls=None,cuda_ns=None),**FLAGS)


def consume(output,report_sha,completion_sha,prepared,prepare_sha):
    return _consume(output,report_sha,completion_sha,prepared,prepare_sha)


def _witness(receipt,receipt_sha,index,*,permit_fixture=False):
    """Export only from a MAIN SHA-authenticated complete own saved receipt.

    This inherits that receipt's native IO verification; no second fullstrict,
    model replay or permission to deploy is implied by the exported plan.
    """
    r=read(receipt,receipt_sha)
    require(r['schema']==NAME+'.consume' and r['status']==('RAM_fixture_only_NOT_evidence' if permit_fixture else 'saved_complete_NOT_admitted')
            and r['fixed_signals']==7 and r['fixed_probes']==5 and len(r['plans'])==5
            and r['verification']['full_native_closure'] is True and r['verification']['native_calls']==17,
            'actual complete own receipt, never RAM/partial/status alone')
    require(all(r[k] is False for k in FLAGS), 'unchanged scientific flags')
    core.audit_ledger(r['resource_receipt'])
    require(r['resource_receipt']['attempted']==r['resource_receipt']['completed']==caps(), 'full original paid vector')
    core.integer(index);require(index<5, 'fixed probe index')
    plan=deepcopy(r['plans'][index]); h=plan.pop('witness_sha256')
    require(digest(plan)==h and plan['schema']==NAME+'.probe_plan' and plan['index']==index, 'full witness content hash')
    task=core.validate_task(plan['task'])
    require(digest(task)==plan['task_sha256'] and plan['whole']==plan['segmented'], 'full task/double replay identity')
    core.checked_path(task,task['root_context'],plan['raw5'],plan['whole'])
    require(np.asarray(plan['raw5'],'<f4').tobytes().hex()==plan['raw_float32_bits'] and
            np.asarray(components.suffix_raw(plan['params_float64']),'<f4').tobytes().hex()==plan['raw_float32_bits'], 'literal original U chart')
    a=plan['authority']
    require(a['prepare_sha256']==r['prepare_sha256'] and a['report_sha256']==r['report_sha256']
            and a['completion_sha256']==r['completion_sha256'], 'receipt authority domains')
    for key in ('report','completion'):
        require(sha(absolute(a[key]))==a[key+'_sha256'], 'actual witness artifact identity')
    require(sha(absolute(a['prepared']))==a['prepare_sha256'], 'actual witness prepared identity')
    plan['witness_sha256']=h
    return dict(schema=NAME+'.witness',receipt=str(absolute(receipt)),receipt_sha256=receipt_sha,plan=plan,
                evidence_kind='RAM_fixture_NOT_evidence' if permit_fixture else 'MAIN_SHA_bound_saved_receipt',**FLAGS)


def witness(receipt,receipt_sha,index):
    return _witness(receipt,receipt_sha,index)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True)
    for k in ('prepare','report','completion','receipt','source'):
        q.add_argument('--bridge-'+k,required=True);q.add_argument('--bridge-'+k+'-sha',required=True)
    q=sub.add_parser('witness');q.add_argument('--receipt',required=True);q.add_argument('--receipt-sha',required=True)
    q.add_argument('--index',required=True,type=int);q.add_argument('--output',required=True)
    for cmd in ('verify','run','consume'):
        q=sub.add_parser(cmd);q.add_argument('--prepared',required=True);q.add_argument('--prepare-sha',required=True)
        if cmd in ('run','consume'):q.add_argument('--output',required=True)
        if cmd=='run':q.add_argument('--authorization',required=True)
        if cmd=='consume':
            q.add_argument('--report-sha',required=True);q.add_argument('--completion-sha',required=True);q.add_argument('--receipt',required=True)
    a=ap.parse_args(argv)
    if a.cmd=='prepare':
        authority={k:dict(path=getattr(a,'bridge_'+k),sha256=getattr(a,'bridge_'+k+'_sha')) for k in ('prepare','report','completion','receipt','source')}
        value=write_new(a.output,build(authority),FILE_CAPS['binding.json'])
    elif a.cmd=='witness':value=write_new(a.output,witness(a.receipt,a.receipt_sha,a.index))
    elif a.cmd=='verify':verify(a.prepared,a.prepare_sha);value=dict(status='verified_not_executed',signals=7,probes=5)
    elif a.cmd=='run':value=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:value=write_new(a.receipt,consume(a.output,a.report_sha,a.completion_sha,a.prepared,a.prepare_sha))
    print(raw(value).decode(),flush=True)


if __name__=='__main__':main()
