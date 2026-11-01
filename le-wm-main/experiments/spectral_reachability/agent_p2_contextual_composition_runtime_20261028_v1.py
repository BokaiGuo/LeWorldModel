"""Known-moving four-slot frontier transplant. MAIN HOST CUDA only; no admission.

The original frontier kernels run unchanged on the original off/on root10
contexts, not on observed teacher15 selector contexts. No physical labels enter.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time
import zlib

import numpy as np

from experiments.spectral_reachability import agent_allmode_prefix_20261028_v2 as assets
from experiments.spectral_reachability import agent_p2_equal144_baseline_cuda_20261028_v1 as loader
from experiments.spectral_reachability import p2_composition_predict_frontier_cuda_20261027_v2 as frontier

NAME = 'agent_p2_contextual_composition_runtime_20261028_v1'
ROOT, overlay = assets.ROOT, assets.overlay
require, sha, canonical, digest = overlay.require, loader.sha, overlay.canonical, frontier.digest
ARMS, ORDER = frontier.ARMS, tuple(assets.ORDER)
CAP, FINAL, FLOOR, WALL = 128*1024**2, 8*1024**2, 200*1024**2, 900
PREP_CAP, EXPANDED = 2*1024**2, 256*1024**2
OP_KINDS = ('jacobian', 'bounded_ridge_svd', 'mode_svd', 'suffix_jacobian', 'suffix_solve')


def protocol():
    return dict(schema=NAME+'.protocol', known_roots=2, scientific_slots=4,
        order=[list(x) for x in ORDER], arms=list(ARMS), programs_per_arm=[1, 2],
        candidates_per_slot=[80, 40], candidate_denominator=480, plan_denominator=8,
        start='original observed root10, z0/z5/z10 and actual warm10',
        suffix_start='own model-predicted complete mid15 context, never observed teacher15',
        goal='original same-condition task goal; original identity192 terminal MSE',
        selector='stable minimum(cost,index), all candidates, no physical input',
        shared_predict=4, exclusive_predict_per_arm=[164, 164],
        standalone_predict_per_arm=[168, 168], paired_unique_predict_per_slot=332,
        total_predict=1328, action_encoder=1328, predictor=1328, pred_proj=1328,
        root_J_per_slot=1, suffix_J_per_slot=40, mathematical_VJP_per_slot=7872,
        shared_SVD_per_slot=2, suffix_SVD_solve_cap_per_slot=40,
        suffix_SVD_zero_radius='0 calls; nominal and Jacobian still executed',
        cost_cap='fixed original nonlinear Q evaluations: 120/slot; no pruning or retry',
        bank='original frontier40 + seeded top9 mixtures40, seed20261027+scientific_slot',
        action_domain='original frontier joint20 logit ball .75*sqrt(10), NumPy tanh(float64)->float32',
        not_equal144_domain='not independent packet-ball projection; no equality claim with packet-mapped baseline',
        damping=.002, raw_steps=10, packets=2, steps_per_packet=5,
        replay='each arm fresh segmented2 plus whole2, included in164',
        inherited_encoding_once=dict(encoder=16, projector=16),
        inherited_search_costs=None, upstream_acquisition_costs=None, total_FLOPs=None,
        backward_engine_calls=None, equal_FLOPs=False, equal_total_resources=False,
        new_encoder=0, new_projector=0, new_physics=0, training=False, Qfit=False,
        P1_admission=False, P2_admission=False, fresh=False, physical_release=False,
        numerics=dict(CUDA_only=True, MATH=True, TF32=False, autocast=False, batch=1),
        retries=0, wall_seconds=WALL, output_cap_bytes=CAP,
        final_reserve_bytes=FINAL, disk_floor_bytes=FLOOR)


def source_files():
    """Bind project import closure, including unchanged dispatch and loader."""
    todo=[Path(__file__).resolve(), Path(assets.__file__), Path(loader.__file__), Path(frontier.__file__)]
    seen=set(overlay.source_files())
    while todo:
        p=todo.pop().resolve()
        if p in seen:
            continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text())):
            names=[]
            if isinstance(n, ast.Import):
                names=[a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                names=[n.module]+[n.module+'.'+a.name for a in n.names]
            for name in names:
                if name.startswith('experiments.'):
                    child=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                    if child.is_file() and child not in seen:
                        todo.append(child)
    seen.update((ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'), loader.sampling.OFFICIAL))
    return sorted(seen)


def task_bindings(slot):
    e=overlay.read(slot['task_file'], PREP_CAP)
    require(sha(slot['task_file'])==slot['task_sha256'], 'actual task bytes')
    interface=overlay.task_interface(e); t=e['task']; c=e['task_contract']
    require(e['task_id']==slot['task_id'] and t['root_context']==slot['packet']['root_context'] and
        t['goal']==slot['packet']['goal'] and (c['root_slot'],c['condition'])==ORDER[slot['index']] and
        np.array_equal(interface['q'], np.eye(192)), 'exact root/condition/context/goal/Q')
    require(np.array_equal(t['action_mean'], overlay.moving.normalization()['mean']) and
        np.array_equal(t['action_std'], overlay.moving.normalization()['std']) and
        np.array_equal(t['root_context']['action_history'],
            overlay.core.normalized_tokens(t, np.asarray(t['root_raw_action_history'], np.float32))),
        'original raw warm10 and one normalization')
    root=dict(root_id=e['task_id'], root_rgb_sha256=c['pixel_sha256'][2],
        goal_rgb_sha256=c['pixel_sha256'][3], context=deepcopy(t['root_context']))
    domain=dict(name='raw', units='native PushT raw float32, no second /100',
        normalization_sha256=digest(dict(mean=t['action_mean'],std=t['action_std'])),
        action_block=5, low=[-1.,-1.], high=[1.,1.])
    compiled=dict(goal_vectors=[deepcopy(t['goal'])], scoring=dict(metric=dict(projection=deepcopy(t['projection']))))
    return dict(root=root, action_domain=domain), compiled


def build():
    # Existing complete native consumer authenticates the16 encodes, full
    # tensor chain, original blind NPZ and four exports; no new model work.
    p=assets.build()
    r=overlay.read(assets.PARENT/'report.json', 4*1024**2)
    parent=overlay.verify(r['prepared'], assets.PARENT_PREP_SHA)
    require([(s['root_slot'],s['condition']) for s in p['slots']]==list(ORDER), 'all four original slots')
    bindings=dict(parent['bindings']); bindings.update(p['bindings'])
    for f in source_files():
        bindings[str(f.resolve())]=sha(f)
    for slot in p['slots']:
        task_bindings(slot)
    check_files(dict(bindings_sha256=bindings))
    return dict(schema=NAME+'.prepare', contract=protocol(), slots=p['slots'],
        policy=p['policy'], expected_model_hash=p['expected_model_hash'], bindings_sha256=bindings,
        source_report_sha256=overlay.SOURCE_SHA, inherited_report_sha256=assets.PARENT_SHA,
        group_denominator=[dict(index=s['index'], root_slot=s['root_slot'],condition=s['condition'],
            task_id=s['task_id'], arm=a, candidates=80 if a==ARMS[0] else 40) for s in p['slots'] for a in ARMS])


def check_files(p):
    for path,h in p['bindings_sha256'].items():
        require(Path(path).is_file() and not Path(path).is_symlink() and sha(path)==h, 'file closure '+path)


def prepare(output):
    p=build(); raw=canonical(p); path=Path(output).absolute()
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and
        len(raw)<=PREP_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(raw), 'exclusive bounded prepare')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw), 'short prepare'); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(path), prepare_sha256=sha(path), status='engineering_prepared_not_admitted')


def verify(path, expected):
    require(sha(path)==expected, 'external prepare SHA')
    p=overlay.read(path, PREP_CAP)
    require(p==build(), 'full own preparation reconstruction')
    return p


def expanded_bank(base, slot):
    """Same fixed frontier recipe, explicit moving slot seed; no fake old root ID."""
    require(type(slot) is int and slot in range(4), 'four scientific slot seed')
    u,right=frontier.bridge.validate_base(base)
    rng=np.random.Generator(np.random.PCG64(frontier.bridge.SEED_BASE+slot))
    tail=[]; coefficients=[]
    for _ in range(10):
        c=rng.normal(size=9); require(np.linalg.norm(c)>0, 'no redraw')
        c=c/np.linalg.norm(c); coefficients.append(c.tolist())
        for radius in frontier.bridge.RADII:
            for sign in (-1,1):
                tail.append(sign*radius*(c@right[:9]))
    logits=np.concatenate((u,np.asarray(tail)))
    return dict(logits=logits.tolist(),commands=np.tanh(logits).astype(np.float32).reshape(80,2,5,2).tolist(),
        seed=frontier.bridge.SEED_BASE+slot, coefficients=coefficients, base_sha256=digest(base))


def provenance(p, slot, bank, arm):
    return dict(checkpoint_sha256=p['bindings_sha256'][str(Path(p['policy'])/'weights.pt')],
        encoder_sha256=slot['task_sha256'], source_sha256=overlay.SOURCE_SHA,
        protocol_sha256=digest(protocol()), metric_sha256=digest(dict(goal=slot['packet']['goal'],q=slot['q'])),
        candidate_bank_sha256=digest(dict(bank=bank,arm=arm)))


class Store(overlay.Store):
    def __init__(self, output):
        self.started=time.monotonic(); self.output=Path(output).absolute()
        require(self.output.name.startswith(NAME+'_') and not self.output.exists() and
            not self.output.is_symlink() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,
            'exclusive128MiB plus200MiB floor')
        self.output.mkdir(); self.comp=zlib.compressobj(6,zlib.DEFLATED,31)
        self.closed=self.broken=False; self._write('events.jsonl.gz',b'')

    def check(self):
        require(time.monotonic()-self.started<=WALL, '900s boundary wall; no native call preemption')

    def _write(self, name, raw, append=False, final=False):
        if not final:
            self.check()
        path=overlay.flat(self.output,name)
        require(not path.is_symlink(), 'regular output')
        used=sum(f.stat().st_size for f in self.output.iterdir())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and
            shutil.disk_usage(self.output).free>=FLOOR+len(raw), 'output cap/final reserve/floor')
        with path.open('ab' if append else 'xb') as f:
            require(f.write(raw)==len(raw), 'short output'); f.flush(); os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def event(self, e):
        require(not self.closed and not self.broken, 'journal cannot resume')
        try:
            raw=self.comp.compress(canonical(e)+b'\n')+self.comp.flush(zlib.Z_SYNC_FLUSH)
            self._write('events.jsonl.gz',raw,append=True)
        except BaseException:
            self.broken=True
            raise


class Gate:
    """Intent is not execution. Ordered native entry/return and operation caps."""
    def __init__(self, sink, check=lambda:None):
        self.sink,self.check=sink,check
        self.entries=[]; self.operations=[]; self.pending=None; self.failed=False
        self.arrays=[]
        self.specs=frontier.expected_entries(80); self.scopes=frontier.expected_scopes()
        self.op_specs=[('jacobian',None),('bounded_ridge_svd',None),('mode_svd',None)]
        self.op_specs += [(k,i) for i in range(40) for k in ('suffix_jacobian','suffix_solve')]

    def __call__(self,e):
        self.check(); require(not self.failed, 'poisoned dispatch')
        k=e.get('kind')
        try:
            if k=='entry_attempt':
                x=e['entry']; i=len(self.entries)
                require(i<len(self.specs) and x['index']==i and x['kind']==self.specs[i]['kind'] and
                    x['scope']==self.scopes[i] and x['samples']==1 and x['context_tokens']==3 and
                    x['completed'] is False and all(z['completed'] for z in self.entries), 'exact native predispatch schedule')
                scope=x['scope']; expected_pending=None
                if scope['phase']=='compilation' and scope['path']==['joint_jacobian']:
                    expected_pending=('jacobian',None)
                if scope['phase']=='suffix_compilation' and scope['path'][1]=='jacobian':
                    expected_pending=('suffix_jacobian',scope['path'][0])
                require(self.pending==expected_pending, 'forward must be inside its actual AD operation, or no pending math')
                self.sink(dict(kind='reserve_intent',resource=x['kind'],scope=x['scope'],upper=1,actual_completed=False))
                self.entries.append(deepcopy(x))
            elif k=='entry_terminal':
                x=e['entry']; require(self.entries and self.entries[-1]==dict(x,completed=False) and x['completed'] is True,
                    'native return must complete current entry')
                self.entries[-1]=deepcopy(x)
            elif k in OP_KINDS:
                key=(k,e.get('index'))
                request=e['completed'] is False and 'wall_time_s' not in e
                expected_entries=(4 if request else 8) if k=='jacobian' else 8
                if k in ('suffix_jacobian','suffix_solve'):
                    expected_entries=336+8*e['index']+(4 if k=='suffix_jacobian' and request else 6)
                require(len(self.entries)==expected_entries and all(x['completed'] for x in self.entries),
                    'operation joined to exact completed forward boundary')
                if request:
                    i=len(self.operations)//2
                    require(self.pending is None and i<len(self.op_specs) and key==self.op_specs[i], 'ordered operation cap')
                    if k in ('jacobian','suffix_jacobian'):
                        require(e['shape']==[192,20 if k=='jacobian' else 10] and e['vectorize'] is True, 'AD contract')
                    else:
                        require(e['svd_calls_requested'] in (0,1) and
                            (k=='suffix_solve' or e['svd_calls_requested']==1), 'solve request cap')
                    self.sink(dict(kind='reserve_intent',resource=k,index=e.get('index'),upper=1,actual_completed=False))
                    self.pending=key
                else:
                    require(self.pending==key and e['completed'] is True, 'operation return or failure')
                    self.pending=None
                self.operations.append(deepcopy(e))
            elif k=='array_saved':
                i=len(self.arrays)
                expected_count=(4 if i==0 else 8) if i<2 else 340+8*((i-2)//2)+2*((i-2)%2)
                require(i<82 and len(self.entries)==expected_count and all(x['completed'] for x in self.entries),
                    'durable derivative array before next operation/forward')
                self.arrays.append(deepcopy(e['receipt']))
            self.sink(dict(kind='native_event',event=deepcopy(e)))
        except BaseException:
            self.failed=True
            raise

    def complete(self,results):
        require(not self.failed and self.pending is None and len(self.arrays)==82, 'no failure/pending/missing array work')
        budget=frontier.validate_actual(self.entries)
        math=frontier.validate_operations(self.operations,results)
        return dict(budget=budget,math=math,actual_forward_entries=len(self.entries),
            reserved_intent_is_computation=False, total_FLOPs=None, backward_engine_calls=None)


class Metric:
    """Fixed nonlinear-Q evaluation cap; no physical metric or learned score."""
    def __init__(self, compiled, sink, check=lambda:None):
        self.value=frontier.legacy().full20.metric_for(compiled)
        self.sink,self.check=sink,check; self.entries=[]; self.failed=False

    def __call__(self,arm,context):
        self.check(); cap=80 if arm==ARMS[0] else 40
        require(not self.failed and arm in ARMS and sum(x['arm']==arm for x in self.entries)<cap,
            'predeclared nonlinear cost evaluation cap')
        ticket=dict(index=len(self.entries),arm=arm,context_sha256=digest(context),completed=False)
        try:
            self.sink(dict(kind='cost_attempt',receipt=deepcopy(ticket)))
            self.entries.append(ticket)
            value=self.value(deepcopy(context))
            ticket.update(completed=True,value=value)
            self.sink(dict(kind='cost_return',receipt=deepcopy(ticket)))
            return value
        except BaseException:
            self.failed=True
            raise


@contextmanager
def module_trace(model, replay, gate, sink):
    """Real tensor bridges for ALL forwards, including AD; no synthetic receipts."""
    pending={}; tickets=[]; hooks=[]
    resources=('action_encoder','predictor','pred_proj')
    def before(resource):
        def enter(module,args):
            require(module not in pending and len(tickets)<996 and resource==resources[len(tickets)%3], 'native module order/cap')
            shapes=[tuple(x.shape) for x in args]; loader.sampling.module_shapes(resource,shapes)
            require(all(x.device.type=='cuda' for x in args), 'CUDA tensor inputs only')
            gate.check()
            ticket=dict(index=len(tickets),resource=resource,scope=deepcopy(replay._p2_scope),
                inputs=[loader.sampling.tensor_receipt(x) for x in args],completed=False)
            sink(dict(kind='module_attempt',receipt=deepcopy(ticket)))
            pending[module]=ticket; tickets.append(ticket)
        return enter
    def returned(module,args,output):
        ticket=pending.pop(module,None)
        if ticket is not None:
            require(output is not None, 'module failed')
            ticket.update(completed=True,output=loader.sampling.tensor_receipt(output))
            sink(dict(kind='module_return',receipt=deepcopy(ticket)))
    def forbid(module,args):
        raise ValueError('new image encoder/projector forbidden')
    try:
        for m,r in ((model.action_encoder,'action_encoder'),(model.predictor,'predictor'),(model.pred_proj,'pred_proj')):
            hooks.extend((m.register_forward_pre_hook(before(r)),m.register_forward_hook(returned,always_call=True)))
        for m in (model.encoder,model.projector): hooks.append(m.register_forward_pre_hook(forbid))
        yield tickets
        require(not pending and len(tickets)==996 and all(x['completed'] for x in tickets), 'all actual module returns')
    finally:
        for h in reversed(hooks): h.remove()


def execute_slot(p,slot,model,device,store):
    """Unchanged production frontier kernels, not fixture=True or selector reuse."""
    ready,compiled=task_bindings(slot); faults=frontier.Faults(); stages=[]; arrays=[]; results=[]
    row=dict(index=slot['index'],ready=ready,compiled=compiled,results=results,arrays=arrays,
        derivative_stages=stages,status='attempted',candidate_denominator=[80,40])
    def sink(e): store.event(dict(slot=slot['index'],payload=e))
    gate=Gate(sink,store.check); metric=Metric(compiled,sink,store.check)
    replay=frontier.legacy().full20.RootAnchoredCudaReplay(model,device)
    def save(name,value,binding):
        a=np.asarray(value,np.float64)
        require(a.shape in ((192,),(192,20),(192,10)) and np.isfinite(a).all(), 'finite derivative array')
        body=dict(name=name,binding=binding,shape=list(a.shape),dtype='float64',value=a.tolist(),
            array_sha256=hashlib.sha256(a.tobytes()).hexdigest())
        file=f'slot{slot["index"]}_{name}.json.gz'; store.put(file,body)
        receipt=dict(file=file,sha256=sha(store.output/file)); arrays.append(receipt)
        gate(dict(kind='array_saved',receipt=receipt)); return a
    def shared_save(stage,value): return save('shared_'+stage,value,dict(task_id=slot['task_id']))
    def suffix(context,prefix,i):
        binding=dict(task_id=slot['task_id'],context_sha256=digest(context),prefix_sha256=digest(prefix.tolist()),index=i)
        return frontier.suffix_cuda(model,device,replay,context,prefix,compiled,i,gate,
            lambda stage,value:save(f'suffix{i}_'+stage,value,binding),faults,stages)
    try:
        with module_trace(model,replay,gate,sink) as modules, frontier.guarded_entries(model,replay,gate,faults):
            base=frontier.compile_cuda(model,device,ready,compiled,replay,gate,shared_save,faults,stages)
            bank=expanded_bank(base,slot['index']); row.update(base=base,bank=bank)
            store.put(f'slot{slot["index"]}_bank.json.gz',dict(base=base,bank=bank))
            gate(dict(kind='bank_frozen',bank_sha256=digest(bank)))
            for arm in ARMS:
                result=frontier.run_arm(arm=arm,bank=bank,root=ready['root'],
                    provenance=provenance(p,slot,bank,arm),action_domain=ready['action_domain'],
                    target_metric=lambda c:metric(arm,c),replay=replay,
                    suffix=suffix,journal=gate,faults=faults)
                results.append(result)
                require(result['complete'], 'incomplete arm, no retry/refill')
        require(len(metric.entries)==120 and all(x['completed'] for x in metric.entries), 'all120 original-Q evaluations')
        row.update(resources=gate.complete(results),status='complete')
    except BaseException as exc:
        faults.capture('slot_execution',exc)
    finally:
        row.update(actual_model_entries=gate.entries,operations=gate.operations,
            modules=locals().get('modules',[]),costs=metric.entries,faults=faults.receipt())
        # Even incomplete slot keeps every reached candidate, pending ticket and
        # full denominator; later slots stay not_started in the terminal report.
        faults.attempt('slot_result_persistence',lambda:store.put(f'slot{slot["index"]}_result.json.gz',row))
    if faults.primary is not None:
        for e in faults.receipt()['secondary']:
            faults.primary.add_note(str(e))
        raise faults.primary
    return row


def run_cuda(prepared, expected, output):
    p=verify(prepared,expected); store=Store(output)
    rows=[dict(index=s['index'],root_slot=s['root_slot'],condition=s['condition'],
        status='not_started',candidate_denominator=[80,40]) for s in p['slots']]
    errors=[]; model=None; before=after=None; source_ok=False; host=None
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        store.check(); model,before=loader.loaded_model(p,torch,device,store,expected)
        with sdpa_kernel(SDPBackend.MATH):
            for slot,row in zip(p['slots'],rows,strict=True):
                row['status']='attempted'
                result=execute_slot(p,slot,model,device,store)
                row.update(status='complete',resources=result['resources'],result_file=f'slot{slot["index"]}_result.json.gz')
    except BaseException as exc:
        errors.append(loader.enc.error('execution',exc))
    finally:
        for row in rows:
            if row['status']=='attempted': row['status']='partial_or_failed'
        if model is not None:
            try: loader.enc.frozen(model); after=loader.enc.tensor_hash(model)
            except BaseException as exc: errors.append(loader.enc.error('final_model',exc))
        try: verify(prepared,expected); source_ok=True
        except BaseException as exc: errors.append(loader.enc.error('final_source',exc))
        try: store.close()
        except BaseException as exc: errors.append(loader.enc.error('journal_close',exc))
        complete=not errors and source_ok and before==after==p['expected_model_hash'] and all(x['status']=='complete' for x in rows)
        report=dict(schema=NAME+'.report',status='known_contextual_plans_sealed' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=protocol(),rows=rows,errors=errors,
            artifact_sha256={f.name:sha(f) for f in store.output.iterdir() if f.is_file()},
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before else None,host=host,elapsed_seconds=time.monotonic()-store.started,
            P1_admission=False,P2_admission=False,physical_release=False)
        store.put('report.json',report,final=True)
    require(complete, 'partial output retained; no retry '+str(errors))
    return report


def module_array(receipt):
    a=np.asarray(receipt['values'],np.float32)
    require(receipt['dtype']=='float32' and list(a.shape)==receipt['shape'] and np.isfinite(a).all() and
        np.array_equal(a.astype(float),receipt['values']), 'literal float32 tensor receipt')
    return a


def check_tensor_chain(slot,row):
    """Independently join all332 packets to raw commands and rolling contexts."""
    initial=slot['packet']['root_context']; results={r['arm']:r for r in row['results']}
    modules=row['modules']; require(len(modules)==996, 'all native module entries')
    scopes=frontier.expected_scopes()[::2]; prior=None; before=None
    for j,scope in enumerate(scopes):
        arm,phase,path=scope['arm'],scope['phase'],scope['path']
        if phase=='compilation':
            # Consecutive nominal/J forwards each run two packets from root.
            before=deepcopy(initial) if j%2==0 else prior
            raw=np.zeros((5,2),np.float32)
        else:
            result=results[arm]; n=80 if arm==ARMS[0] else 40
            candidate=result['attempts'][path[0]] if phase!='replay' else next(
                x for x in result['attempts'] if x['candidate_id']==result['winner']['candidate_id'])
            actions=candidate['actions']
            if phase=='suffix_compilation':
                before=candidate['witness']['edges'][0]['output_context']; raw=np.zeros((5,2),np.float32)
            elif phase=='planning':
                before=deepcopy(initial) if path[1]==0 else (
                    candidate['witness']['edges'][0]['output_context'] if arm==ARMS[1] else prior)
                raw=np.asarray(actions[5*path[1]:5*path[1]+5],np.float32)
            else:
                k=0 if path[0]==n or (path[0]==n+2 and scopes[j-1]!=scope) else 1
                before=deepcopy(initial) if k==0 else prior
                raw=np.asarray(actions[5*k:5*k+5],np.float32)
        ae,pred,proj=modules[3*j:3*j+3]
        require([x['resource'] for x in (ae,pred,proj)]==['action_encoder','predictor','pred_proj'] and
            all(x['index']==3*j+k and x['scope']==scope and x['completed'] is True
                for k,x in enumerate((ae,pred,proj))), 'module identities/scopes/returns')
        ain=module_array(ae['inputs'][0]); aout=module_array(ae['output'])
        pin=module_array(pred['inputs'][0]); action=module_array(pred['inputs'][1])
        pout=module_array(pred['output']); qin=module_array(proj['inputs'][0]); qout=module_array(proj['output'])
        outgoing=np.concatenate((np.asarray(before['action_history'],np.float32),
            np.asarray([frontier._normalized_token(raw)],np.float32)))
        require(ain.shape==(1,3,10) and np.array_equal(ain[0],outgoing) and pin.shape==(1,3,192) and
            np.array_equal(pin[0],before['latent_history']) and np.array_equal(action,aout) and
            pout.shape==(1,3,192) and qin.shape==qout.shape==(3,192) and np.array_equal(qin,pout.reshape(3,192)),
            'actual raw->normalized history->action embedding->predictor->pred_proj seam')
        prior=dict(before,latent_history=deepcopy(before['latent_history'][1:])+[qout[-1].astype(float).tolist()],
            action_history=outgoing[1:].astype(float).tolist())
        if phase=='planning' and path[1]==1:
            require(prior==candidate['witness']['receipt']['terminal_context'], 'all candidate tensor terminals')
        if phase=='planning' and path[1]==0 and arm==ARMS[1]:
            require(prior==candidate['witness']['edges'][0]['output_context'], 'actual own-prefix tensor context')
        if phase=='suffix_compilation':
            require(prior['latent_history'][-1]==candidate['suffix_compilation']['nominal'], 'suffix nominal/AD graph endpoint')
        if phase=='compilation' and j%2==1:
            require(prior['latent_history'][-1]==row['base']['nominal'], 'root nominal/AD graph endpoint')
        if phase=='replay':
            expected=result['winner']['segmented_contexts'][1 if k==0 else 2]
            require(prior==expected, 'independent winner tensor replay')


def check_module_event_order(events):
    """Join hooks to guarded original entry boundaries, not just hash lists."""
    actual=[]
    for e in events:
        if e['kind'] in ('module_attempt','module_return'):
            x=e['receipt']; actual.append((e['kind'],x['resource'],x['index'],x['scope']))
        elif e['kind']=='native_event' and e['event']['kind'] in ('entry_attempt','entry_terminal'):
            x=e['event']['entry']; actual.append((e['event']['kind'],x['kind'],x['index'],x['scope']))
    expected=[]
    for j,scope in enumerate(frontier.expected_scopes()[::2]):
        expected.extend((k,r,i,scope) for k,r,i in [
            ('module_attempt','action_encoder',3*j),('entry_attempt','action_encoder',2*j),
            ('entry_terminal','action_encoder',2*j),('module_return','action_encoder',3*j),
            ('entry_attempt','predict',2*j+1),('module_attempt','predictor',3*j+1),
            ('module_return','predictor',3*j+1),('module_attempt','pred_proj',3*j+2),
            ('module_return','pred_proj',3*j+2),('entry_terminal','predict',2*j+1)])
    require(actual==expected, 'actual module hooks enclosed by native entry/return boundaries')


def consume(output, report_sha):
    output=Path(output).absolute(); require(sha(output/'report.json')==report_sha, 'external report SHA')
    r=overlay.read(output/'report.json', 8*1024**2); p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='known_contextual_plans_sealed' and r['errors']==[] and
        r['contract']==protocol() and r['source_verified'] is r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['P1_admission'] is r['P2_admission'] is r['physical_release'] is False, 'terminal complete only, no admission')
    require(len(r['rows'])==4 and [(x['index'],x['root_slot'],x['condition'],x['status']) for x in r['rows']]==
        [(i,*ORDER[i],'complete') for i in range(4)], 'fixed four-slot terminal denominator')
    names={'binding.json','loader.json','events.jsonl.gz'}
    for i in range(4):
        names.update((f'slot{i}_result.json.gz',f'slot{i}_bank.json.gz'))
        names.update(f'slot{i}_{s}.json.gz' for s in ['shared_nominal','shared_jacobian']+
            [f'suffix{j}_{stage}' for j in range(40) for stage in ('nominal','jacobian')])
    require(set(r['artifact_sha256'])==names and {x.name for x in output.iterdir()}==names|{'report.json'}, 'exact complete artifacts')
    for name,h in r['artifact_sha256'].items():
        require(not (output/name).is_symlink() and sha(output/name)==h, 'artifact hash')
    require(overlay.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']), 'execution binding')
    loader.validate_loader(overlay.read(output/'loader.json'),p,r['prepare_sha256'])
    raw,eof=overlay.blind.inflate((output/'events.jsonl.gz').read_bytes(),EXPANDED)
    require(eof and raw.endswith(b'\n'), 'complete bounded journal')
    events=[overlay.blind.decode(x) for x in raw.splitlines()]
    load=overlay.read(output/'loader.json')
    expected_load=[e for f in load['files'] for e in [dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']),
        dict(kind='consumed_buffer_frozen',receipt=f)]]
    require(events[:4]==expected_load, 'actual immutable loader buffers')
    plans=[]; offset=4
    for slot,rr in zip(p['slots'],r['rows'],strict=True):
        require(rr['result_file']==f'slot{slot["index"]}_result.json.gz' and rr['candidate_denominator']==[80,40],
            'fixed own result path and full failure denominator')
        row=overlay.read(output/rr['result_file'],EXPANDED); ready,compiled=task_bindings(slot)
        require(row['index']==slot['index'] and row['ready']==ready and row['compiled']==compiled and row['status']=='complete'
            and row['candidate_denominator']==[80,40] and row['faults']==dict(primary=None,secondary=[]), 'complete own task row')
        emitted=[]; gate=Gate(emitted.append); own=[]
        while offset<len(events) and events[offset].get('slot')==slot['index']:
            own.append(events[offset]['payload']); offset+=1
        require(own, 'all slot journals')
        for e in own:
            if e['kind']=='native_event': gate(e['event'])
        require([x for x in own if x['kind'] in ('native_event','reserve_intent')]==emitted and
            gate.entries==row['actual_model_entries'] and gate.operations==row['operations'] and
            gate.complete(row['results'])==row['resources']==rr['resources'], 'intent/entry/return/actual count reconciliation')
        requests=[x['receipt'] for x in own if x['kind']=='module_attempt']
        returns=[x['receipt'] for x in own if x['kind']=='module_return']
        require(returns==row['modules'] and requests==[{k:v for k,v in x.items() if k!='output'}|{'completed':False}
            for x in returns], 'real module request/return receipts')
        check_module_event_order(own)
        values={}
        array_names=['shared_nominal','shared_jacobian']+[f'suffix{i}_{stage}' for i in range(40) for stage in ('nominal','jacobian')]
        require([x['file'] for x in row['arrays']]==[f'slot{slot["index"]}_{n}.json.gz' for n in array_names],
            'all82 derivative arrays, exact order and own flat paths')
        require(row['derivative_stages']==[dict(kind=kind,index=i,stage=stage,returned=True,converted=True,persisted=True)
            for kind,i in [('jacobian',None)]+[('suffix_jacobian',i) for i in range(40)]
            for stage in ('nominal','jacobian')], 'all derivative stage durable completions')
        require([e['event']['receipt'] for e in emitted if e['kind']=='native_event' and e['event']['kind']=='array_saved']==row['arrays'],
            'durable derivative receipt order')
        for name,receipt in zip(array_names,row['arrays'],strict=True):
            require(sha(output/receipt['file'])==receipt['sha256'], 'derivative file SHA')
            body=overlay.read(output/receipt['file'],PREP_CAP); a=np.asarray(body['value'],np.float64)
            require(body['name']==name and list(a.shape)==body['shape'] and body['dtype']=='float64' and np.isfinite(a).all() and
                hashlib.sha256(a.tobytes()).hexdigest()==body['array_sha256'], 'original derivative array bytes')
            values[body['name']]=body
        base=frontier.legacy().sra.spectral_bank(values['shared_jacobian']['value'],values['shared_nominal']['value'],
            compiled['goal_vectors'][0],compiled['scoring']['metric']['projection'])
        require(row['base']==base and row['bank']==expanded_bank(base,slot['index']) and
            overlay.read(output/f'slot{slot["index"]}_bank.json.gz',PREP_CAP)==dict(base=base,bank=row['bank']), 'root derivatives/bank/all80')
        require(all(values['shared_'+stage]['binding']==dict(task_id=slot['task_id']) for stage in ('nominal','jacobian')), 'root derivative identity')
        for i,candidate in enumerate(row['results'][1]['attempts']):
            binding=dict(task_id=slot['task_id'],context_sha256=digest(candidate['witness']['edges'][0]['output_context']),
                prefix_sha256=digest(row['bank']['logits'][i][:10]),index=i)
            for stage in ('nominal','jacobian'):
                body=values[f'suffix{i}_{stage}']
                require(body['binding']==binding and body['value']==candidate['suffix_compilation'][stage], 'own-context suffix derivative identity')
        check_tensor_chain(slot,row)
        require([x['arm'] for x in row['results']]==list(ARMS), 'both arms ordered')
        costs=[dict(index=j,arm=result['arm'],context_sha256=digest(c['witness']['receipt']['terminal_context']),
            completed=True,value=c['cost']) for j,(result,c) in enumerate(
            (result,c) for result in row['results'] for c in result['attempts'])]
        require(row['costs']==costs and [e for e in own if e['kind'] in ('cost_attempt','cost_return')]==
            [e for ticket in costs for e in [dict(kind='cost_attempt',receipt={k:v for k,v in ticket.items() if k!='value'}|{'completed':False}),
                dict(kind='cost_return',receipt=ticket)]], 'all120 fixed nonlinear cost callback entries/returns')
        for arm,result in zip(ARMS,row['results'],strict=True):
            prov=provenance(p,slot,row['bank'],arm)
            best=frontier.validate_result(result,arm=arm,bank=row['bank'],ready=ready,prov=prov,compiled=compiled)
            plan=dict(root_slot=slot['root_slot'],condition=slot['condition'],arm=arm,task_id=slot['task_id'],
                commands=best['actions'],commands_dtype='<f4',raw_sha256=hashlib.sha256(np.asarray(best['actions'],'<f4').tobytes()).hexdigest(),
                root_context=ready['root']['context'],goal=compiled['goal_vectors'][0],witness=best['witness'],
                original_task_file=slot['task_file'],original_task_sha256=slot['task_sha256'],
                original_blind_sha256=slot['blind_sha256'],candidate_bank_sha256=digest(row['bank']),
                predicted_Q=best['cost'],prepare_sha256=r['prepare_sha256'],planner_report_sha256=report_sha,
                engineering_executable=True,P1_admission=False,P2_admission=False,physical_release=False)
            plan['selection_digest']=digest(plan); plans.append(plan)
    require(offset==len(events) and len(plans)==8 and sha(output/'report.json')==report_sha, 'no unscoped/missing slots')
    check_files(p)
    return dict(schema=NAME+'.consume',status='engineering_verified_science_blocked',plans=plans,
        candidate_denominator=480,known_roots=2,scientific_slots=4,total_predict=1328,total_FLOPs=None,
        P1_admission=False,P2_admission=False,physical_release=False,
        blockers=['independent P1 not admitted','no independent fresh cohort','no new physical efficacy evidence'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run_cuda','consume'))
    parser.add_argument('--prepared',type=Path); parser.add_argument('--prepare-sha')
    parser.add_argument('--output',type=Path); parser.add_argument('--report-sha')
    a=parser.parse_args()
    if a.command=='prepare': out=prepare(a.output)
    elif a.command=='verify':
        p=verify(a.prepared,a.prepare_sha); out=dict(status='engineering_prepared_not_admitted',contract=p['contract'])
    elif a.command=='run_cuda': out=run_cuda(a.prepared,a.prepare_sha,a.output)
    else: out=consume(a.output,a.report_sha)
    print(canonical(out).decode(),flush=True)


if __name__=='__main__': main()
