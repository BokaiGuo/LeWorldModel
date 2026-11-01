"""Durable dispatch adapter for sealed frontier-v2, never a P1/P2 admission.

Known-root CUDA execution is MAIN-only. Import/prepare/tests need no torch or
simulator. Native numeric kernels are called unchanged, not monkey-patched.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from experiments.spectral_reachability import agent_p2_resume_20261027_next as contract
from experiments.spectral_reachability import p2_composition_predict_frontier_cuda_20261027_v2 as native

NAME = 'agent_p2_runtime_guard_20261028_v1'
ROOT = contract.ROOT
C = contract.cost
require = C.require
SHA_NATIVE = 'bf8ae74d7453cbf19f2420c8c83701f8e5c893f08f37293e7960cd84ac2ad061'
NEXT = ROOT/'results/agent_p2_resume_20261027_next_prepared01.json'
NEXT_SHA = 'b3ba3450b7833d872ee3992134c8382e343bbdd58170fb2d6edeb69cfa7b7bcb'
OP_KINDS = ('jacobian','suffix_jacobian','bounded_ridge_svd','mode_svd','suffix_solve')


def encode(value): return contract.prior.graph.encode(value)
def file_sha(path): return contract.prior.file_sha(path)


def write_metadata(path,value):
    """Own-prefix exclusive canonical metadata; never call prior's OLD-prefix writer."""
    path=Path(path).absolute(); data=encode(value)
    require(path.name.startswith(NAME) and path.parent.resolve().is_relative_to(ROOT/'results'),
            'exclusive runtime-guard prefix under results required')
    require(len(data)<=32*1024**2 and shutil.disk_usage(path.parent).free>len(data)+200*1024**2,
            'bounded metadata and 200MiB disk floor')
    with path.open('xb') as stream:
        require(stream.write(data)==len(data), 'short metadata write')
        stream.flush(); os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


class DurableJournal:
    """Exclusive append, hash chain, flush+fsync before returning to a caller.

    A failed/partial append poisons permanently. No truncation, resume or retry.
    Caller must close in finally. Parent is fsynced before first dispatch.
    """
    def __init__(self, path, binding, *, max_bytes=16*1024**2, wall_s=900, floor_bytes=200*1024**2):
        self.path = Path(path); self.binding = deepcopy(binding)
        self.max_bytes, self.wall_s, self.floor_bytes = max_bytes, wall_s, floor_bytes
        require(type(max_bytes) is int and max_bytes > 0 and 0 < wall_s <= 900, 'bounded journal/wall')
        self.started=time.monotonic(); self.seq=0; self.prev='0'*64; self.bytes=0; self.poisoned=False
        self.stream=self.path.open('xb')
        try:
            fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            self.append('header',dict(binding=self.binding, computation_observed=False))
        except BaseException:
            self.stream.close(); raise

    def append(self, kind, payload):
        require(not self.poisoned, 'journal poisoned: no subsequent dispatch')
        try:
            require(time.monotonic()-self.started <= self.wall_s, 'boundary wall cap exceeded')
            require(shutil.disk_usage(self.path.parent).free > self.floor_bytes, 'disk floor before dispatch/write')
            body=dict(schema=NAME+'.event',seq=self.seq,previous_sha256=self.prev,kind=kind,
                      elapsed_ns=time.monotonic_ns(),payload=deepcopy(payload))
            h=hashlib.sha256(encode(body)).hexdigest(); raw=encode(dict(body,event_sha256=h))+b'\n'
            require(self.bytes+len(raw) <= self.max_bytes, 'journal byte cap exceeded')
            require(self.stream.write(raw)==len(raw), 'short journal append')
            self.stream.flush(); os.fsync(self.stream.fileno())
            self.seq+=1; self.prev=h; self.bytes+=len(raw)
        except BaseException:
            self.poisoned=True; raise

    def close(self): self.stream.close()


def read_journal(path):
    """Read-only strict verification. An incomplete tail fails, never repaired."""
    raw=Path(path).read_bytes()
    require(raw.endswith(b'\n'), 'incomplete journal tail')
    out=[]; prev='0'*64
    for line in raw.splitlines():
        row=contract.prior.graph.parse(line); h=row.pop('event_sha256')
        require(row['schema']==NAME+'.event' and row['seq']==len(out) and row['previous_sha256']==prev
                and hashlib.sha256(encode(row)).hexdigest()==h, 'journal hash-chain/order mismatch')
        prev=h; out.append(dict(row,event_sha256=h))
    require(out and out[0]['kind']=='header', 'journal header required')
    return out


def forward_entry(kind):
    require(kind in ('predict','action_encoder'), 'native forward entry only')
    if kind=='predict': return C.vector(predict_calls=1,predict_samples=1,predict_context_tokens=3)
    return C.vector(action_encoder_calls=1,action_encoder_samples=1,action_encoder_context_tokens=3)


def operation_work(e, terminal=False):
    kind=e['kind']; require(kind in OP_KINDS, 'known native operation')
    if kind in ('jacobian','suffix_jacobian'):
        width=20 if kind=='jacobian' else 10
        require(e['shape']==[192,width] and e['vectorize'] is True, 'exact native AD shape/backend')
        return C.vector(**{f'jacobian_192x{width}_calls':1,f'math_vjp_192x{width}_directions':192,
                           'jacobian_scalar_entries':192*width})
    if kind=='suffix_solve':
        n=e['svd_calls'] if terminal else e['svd_calls_requested']
        require(type(n) is int and n in (0,1), 'zero-radius exact solve count')
        if terminal: require(e['bounded_solve_calls']==n, 'SVD/solve count closure')
        return C.vector(svd_192x10_calls=n,bounded_solve_10_calls=n)
    require(e['svd_calls_requested']==1, 'native root SVD request')
    return C.vector(svd_192x20_calls=1,bounded_solve_20_calls=int(kind=='bounded_ridge_svd'))


class DispatchGuard:
    """Real native event sink: predispatch intent and observed returns SEPARATE.

    Group reservations pay once. Primitive reservations are subdivisions, never
    added again to standalone budget. Native guarded_entries performs actual
    model method wrapping; native operation requests bracket AD/solve call sites.
    """
    def __init__(self, journal, root_slot, root_sha256):
        require(type(root_slot) is int and root_slot in range(4) and
                root_sha256==native.ROOTS[root_slot], 'fixed known-root slot')
        self.journal=journal; self.root_slot=root_slot; self.root_sha256=root_sha256
        self.resources=contract.allocation(); self.groups={}; self.pending={}; self.seen_tokens=set(); self.serial=0
        self.current=None; self.poisoned=False; self.entry_count=0; self.failed_returns=[]
        self.reserved={a:C.vector() for a in native.ARMS}
        self.candidates={a:[dict(index=i,status='not_started',error=None) for i in range(contract.COUNTS[a])]
                         for a in native.ARMS}

    def _emit(self, kind, payload):
        require(not self.poisoned, 'dispatch guard poisoned')
        try: self.journal.append(kind,dict(root_slot=self.root_slot,root_sha256=self.root_sha256,**payload))
        except BaseException:
            self.poisoned=True; raise

    def group(self, key, arm, upper):
        require(key not in self.groups, 'group cannot retry/refund')
        contract.guard.supported_vector(upper)
        arms=native.ARMS if arm=='shared' else (arm,)
        require(all(a in native.ARMS for a in arms), 'fixed group arm')
        totals={a:C.add(self.reserved[a],upper) for a in arms}; cap=self.resources['common_standalone_cap']
        require(all(totals[a][k] <= cap[k] for a in arms for k in C.KNOWN), 'standalone group cap exceeded')
        self._emit('group_reserve_intent',dict(group=key,arm=arm,upper=upper,computation_observed=False))
        for a in arms: self.reserved[a]=totals[a]
        self.groups[key]=dict(arm=arm,upper=deepcopy(upper),subreserved=C.vector(),returned=C.vector())
        self.current=key

    def begin_shared(self): self.group('base','shared',self.resources['graph_base_shared_upper'])

    def reserve(self, token, work, evidence):
        require(token not in self.seen_tokens and self.current in self.groups, 'unique operation/current paid group')
        contract.guard.supported_vector(work)
        g=self.groups[self.current]; total=C.add(g['subreserved'],work)
        require(all(total[k]<=g['upper'][k] for k in C.KNOWN), 'primitive allocation exceeds base/edge group')
        self._emit('operation_reserve_intent',dict(token=token,group=self.current,upper=work,
            native_evidence=evidence,computation_observed=False))
        g['subreserved']=total
        self.seen_tokens.add(token)
        self.pending[token]=dict(group=self.current,upper=deepcopy(work),entry=False)
        # entry denotes host dispatch boundary, NOT CUDA synchronization or kernel count.
        self._emit('entry_boundary',dict(token=token,group=self.current,
            observation='native API/operation host-call boundary; not completion',actual_completed=None))
        self.pending[token]['entry']=True

    def returned(self, token, actual, evidence, success=True):
        require(token in self.pending, 'exactly one terminal per operation')
        p=self.pending[token]
        contract.guard.supported_vector(actual)
        require(all(actual[k]<=p['upper'][k] for k in C.KNOWN), 'observed work exceeds intent')
        self._emit('return_observed',dict(token=token,group=p['group'],success=success,
            observed_completed=actual if success else None, native_evidence=evidence,
            observation='host callable returned; GPU completion requires external synchronize'))
        if success: self.groups[p['group']]['returned']=C.add(self.groups[p['group']]['returned'],actual)
        del self.pending[token]
        if not success:
            self.failed_returns.append(token); self.poisoned=True

    def __call__(self, event):
        try: return self._event(deepcopy(event))
        except BaseException:
            self.poisoned=True; raise

    def _event(self,e):
        require(not self.poisoned, 'dispatch guard poisoned')
        kind=e.get('kind')
        if kind=='candidate_request':
            a,i=e['arm'],e['index']; require(a in native.ARMS and type(i) is int and i in range(contract.COUNTS[a]), 'fixed candidate')
            row=self.candidates[a][i]; require(row['status']=='not_started', 'no candidate retry')
            self.group(f'{a}.candidate.{i}',a,self.resources['arms'][a]['per_candidate_upper'])
            row['status']='attempted'
        elif kind in ('entry_attempt','entry_terminal'):
            x=e['entry']; token='forward.'+str(x['index']); scope=x['scope']
            if kind=='entry_attempt':
                require(x['index']==self.entry_count and x['samples']==1 and x['context_tokens']==3, 'exact actual entry metadata')
                specs=native.expected_entries(80); scopes=native.expected_scopes()
                require(x['index']<len(specs) and x['kind']==specs[x['index']]['kind'] and scope==scopes[x['index']], 'entry order/scope')
                if scope['phase']=='replay':
                    key=scope['arm']+'.replay'
                    if key not in self.groups: self.group(key,scope['arm'],self.resources['winner_replay_upper'])
                    self.current=key
                work=forward_entry(x['kind']); self.reserve(token,work,e); self.entry_count+=1
            else:
                self.returned(token,forward_entry(x['kind']),e,success=x['completed'] is True and not x.get('error'))
        elif kind in OP_KINDS:
            token=kind+'.'+str(e.get('index'))
            if e['completed'] is False and token not in self.pending:
                self.reserve(token,operation_work(e),e)
            else:
                self.returned(token,operation_work(e,True),e,success=e['completed'] is True)
        elif 'candidate_id' in e and 'index' in e and 'completed' in e:
            a=e['candidate_id'].rsplit('_',1)[0]; i=e['index']; row=self.candidates[a][i]
            require(row['status']=='attempted', 'candidate terminal without attempt')
            self._emit('candidate_terminal',dict(arm=a,index=i,success=e['completed'] is True,native_event_sha256=C.digest(e)))
            row.update(status='completed' if e['completed'] is True else 'failed',error=e.get('error'))
        elif kind=='arm_failure':
            self._emit('native_failure',dict(event=e)); self.poisoned=True
        else:
            # Native derivative artifacts have their own full hashes in this event.
            self._emit('native_evidence',dict(event=e))

    def metric(self, callback):
        def measured(context):
            token='metric.'+str(self.serial); self.serial+=1
            self.reserve(token,C.vector(selection_evaluations=1),dict(kind='metric_call'))
            try: value=callback(context)
            except BaseException as exc:
                self.returned(token,C.vector(),dict(error_type=type(exc).__name__,error=str(exc)),False); raise
            self.returned(token,C.vector(selection_evaluations=1),dict(kind='metric_return'))
            return value
        return measured

    def snapshot(self, error=None):
        rows=deepcopy(self.candidates)
        if error is not None:
            for records in rows.values():
                for r in records:
                    if r['status']=='not_started': r.update(status='skipped',error=error)
                    elif r['status']=='attempted': r.update(status='failed_or_pending',error=error)
        shared=self.groups.get('base',{}).get('returned',C.vector())
        observed={a:C.add(shared,*[g['returned'] for k,g in self.groups.items() if k!='base' and g['arm']==a]) for a in native.ARMS}
        unique=C.add(*[g['returned'] for g in self.groups.values()]) if self.groups else C.vector()
        return dict(root_slot=self.root_slot,root_sha256=self.root_sha256,candidates=rows,
            reservation_intent_standalone=deepcopy(self.reserved),observed_returned_lower_bound_standalone=observed,
            observed_returned_lower_bound_unique=unique,pending_operations=deepcopy(self.pending),groups=deepcopy(self.groups),
            failed_returns=deepcopy(self.failed_returns),
            unknown_completion=bool(self.pending or self.failed_returns),
            actual_complete_totals=None if self.pending or self.failed_returns else unique,
            physics_and_visual_endpoint='NOT_DISPATCHED; future cap room retained, not actual work',
            upstream_acquisition_cost=None,flops=None,backward_engine_api_calls=None,
            evidence_scope='native host entry/return and mathematical AD/solve counters, not synchronized kernel profiler',
            P2_admission=False,execution_authorized=False)


def prepare(next_path=NEXT,next_sha=NEXT_SHA,native_dir=None,native_sha=None):
    require(file_sha(native.__file__)==SHA_NATIVE, 'sealed native runner SHA')
    next_path=Path(next_path).resolve(); q=contract.verify(contract.prior.read(next_path),next_sha)
    require(native_dir is not None and native_sha is not None, 'externally bound native prepared required')
    directory,p,s=native.load_prepared(native_dir,native_sha)
    policy=Path(p['context']['policy'])
    require(policy.is_absolute() and (policy/'weights.pt').is_file() and (policy/'config.json').is_file(), 'local-only checkpoint assets required')
    require(native.read(policy/'config.json')['encoder']['pretrained'] is False, 'no pretrained constructor download')
    for i,case in enumerate(p['context']['cases']):
        slot=q['plan']['slots'][2*i]; ident=contract.prior.identity(case)
        require(slot['identity']==ident and ident['root_sha256']==native.ROOTS[i], 'native/next full identity seam')
    bindings=dict(q['bindings']); bindings[str(next_path)]=next_sha
    for f in [Path(__file__),ROOT/'tests'/(NAME+'_tests.py'),ROOT/'innerdocs'/(NAME+'.md'),
              Path(native.__file__),directory/'protocol.json',directory/'prepared.json',*map(Path,p['source_paths'])]:
        bindings[str(f.resolve())]=file_sha(f)
    return dict(schema=NAME+'.prepare',next_path=str(next_path),next_sha256=next_sha,
        native_directory=str(directory),native_prepared_sha256=native_sha,bindings=bindings,
        root_order=list(native.ROOTS),resources=contract.allocation(),
        known_root_engineering_only=True,confirmatory=False,P1_admission=False,P2_admission=False,
        execution_requires='MAIN review + explicit execute-known invocation',new_neural_calls=0,new_physics_steps=0)


def verify(value,expected):
    require(C.digest(value)==expected and value['schema']==NAME+'.prepare', 'external wrapper prepare SHA/schema')
    for path,h in value['bindings'].items(): require(file_sha(path)==h, 'wrapper input/source drift')
    require(prepare(value['next_path'],value['next_sha256'],value['native_directory'],value['native_prepared_sha256'])==value,
            'wrapper contract/source closure drift')
    return value


def run_known_root(output,p,s,prepared_sha,ready,compiled,model,device,synchronize):
    """Direct calls to original compile/suffix/run_arm and genuine entry guard.

    Not native.run_root: only orchestration is adapted to inject our event sink.
    Mathematical kernels, candidate generator, action selection and replay stay native.
    """
    output=Path(output); old=native.legacy(); faults=native.Faults(); i=ready['root_index']
    require(getattr(device,'type',None)=='cuda', 'CUDA only')
    journal=DurableJournal(output/f'root_{i}_guard.jsonl',dict(root=ready['root_sha256'],native_prepare_sha256=prepared_sha,
        ready_sha256=C.digest(ready),wrapper_source_sha256=file_sha(__file__),resource_contract_sha256=C.digest(contract.allocation())))
    guard=DispatchGuard(journal,i,ready['root_sha256']); results=[]; operations=[]; arrays=[]; stages=[]; entries=[]; bank=None
    def emit(e):
        guard(e)
        if e.get('kind') in OP_KINDS: operations.append(deepcopy(e))
    sink=faults.journal(emit)
    try:
        replay=old.full20.RootAnchoredCudaReplay(model,device)
        binding=native.array_binding(p,ready,compiled,prepared_sha)
        def shared_save(stage,value): return native.persist_array(output,f'root_{i}_shared_{stage}',value,binding,arrays,sink)
        def suffix(context,prefix,index):
            local=dict(binding,input_context_sha256=C.digest(context),prefix_logits_sha256=C.digest(prefix.tolist()),index=index)
            def save(stage,value): return native.persist_array(output,f'root_{i}_suffix_{index}_{stage}',value,local,arrays,sink)
            return native.suffix_cuda(model,device,replay,context,prefix,compiled,index,sink,save,faults,stages)
        with native.guarded_entries(model,replay,sink,faults) as entries:
            guard.begin_shared()
            base=native.compile_cuda(model,device,ready,compiled,replay,sink,shared_save,faults,stages)
            bank=native.candidate_bank(base,ready['root_sha256'])
            native.write(output/f'root_{i}_bank.json',dict(base=base,expanded=bank))
            for arm in native.ARMS:
                result=native.run_arm(arm=arm,bank=bank,root=ready['root'],
                    provenance=native.provenance(p,s,ready,bank,arm),action_domain=ready['action_domain'],
                    target_metric=guard.metric(old.full20.metric_for(compiled)),replay=replay,suffix=suffix,
                    journal=sink,faults=faults)
                results.append(result)
                require(result['complete'] and not guard.poisoned, 'native arm incomplete/guard poisoned')
        native.validate_actual(entries); native.validate_operations(operations,results)
    except BaseException as exc: faults.capture('guarded_root',exc)
    finally:
        faults.attempt('guarded_root.synchronize',synchronize)
        error=None if faults.primary is None else str(faults.primary)
        snapshot=guard.snapshot(error)
        # A poisoned sink cannot be retried, even for finalization.
        if not journal.poisoned and not guard.poisoned:
            faults.attempt('guarded_root.final_journal',lambda:journal.append('root_terminal',snapshot))
        faults.attempt('guarded_root.close',journal.close)
    return dict(root_slot=i,root_sha256=ready['root_sha256'],complete=faults.primary is None,
        faults=faults.receipt(),guard=snapshot,results=results,actual_model_entries=entries,
        operations=operations,array_receipts=arrays,derivative_stages=stages,
        journal_sha256=file_sha(journal.path),new_physics_steps=0,P2_admission=False)


def execute_known(prepared,expected,output,*,main_review_sha256,confirmatory=False):
    if confirmatory: raise ValueError('confirmatory prohibited: independent P1/freshness not admitted')
    contract.prior.graph.digest(main_review_sha256)
    v=verify(prepared,expected); _,p,s=native.load_prepared(v['native_directory'],v['native_prepared_sha256'])
    output=Path(output).resolve()
    require(output.name.startswith(NAME) and output.parent==ROOT/'results' and not output.exists(), 'exclusive wrapper output')
    require(shutil.disk_usage(output.parent).free>400*1024**2, '200MiB output allowance +200MiB reserve')
    output.mkdir(); (output/'arrays').mkdir()
    native.write(output/'execution.json',dict(wrapper_prepare_sha256=expected,main_review_sha256=main_review_sha256,
        mode='known_root_engineering_NOT_confirmatory',P2_admission=False))
    rows=[]; failure=None; before=None; after=None; model=None; unchanged=None; source_unchanged=None; secondary=[]
    try:
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
        base.require_cuda_allocation('cuda')
        import torch
        device=torch.device('cuda:0'); torch.backends.cuda.matmul.allow_tf32=False
        model=base.load_compatible_pretrained(Path(p['context']['policy']),device=device).eval().requires_grad_(False)
        before=checked.tensor_state_hash(model)
        source=native.read(p['context']['compilation']); compiled={r['root_sha256']:r['compiled'] for r in source['rows']}
        for ready in p['context']['cases']:
            row=run_known_root(output,p,s,v['native_prepared_sha256'],ready,compiled[ready['root_sha256']],
                               model,device,lambda:torch.cuda.synchronize(device))
            rows.append(row)  # Preserve used work/census even if the following write fails.
            native.write(output/f'root_{ready["root_index"]}.json',row)
            if not row['complete']: raise ValueError('root failed; remaining fixed slots retained, no retry')
    except BaseException as exc: failure=dict(type=type(exc).__name__,error=str(exc))
    finally:
        if model is not None and before is not None:
            try:
                after=checked.tensor_state_hash(model); unchanged=before==after
                require(unchanged and all(not m.training for m in model.modules()) and
                        all(not t.requires_grad for t in model.parameters()),'model frozen/eval/state changed')
            except BaseException as exc:
                if failure is None: failure=dict(type=type(exc).__name__,error=str(exc))
                else: secondary.append(dict(stage='model.final_check',type=type(exc).__name__,error=str(exc)))
        try: verify(v,expected); source_unchanged=True
        except BaseException as exc:
            source_unchanged=False
            if failure is None: failure=dict(type=type(exc).__name__,error=str(exc))
            else: secondary.append(dict(stage='source.final_check',type=type(exc).__name__,error=str(exc)))
    census=[]
    by_slot={r['root_slot']:r for r in rows}
    for i,root in enumerate(native.ROOTS):
        for arm in native.ARMS:
            records=by_slot[i]['guard']['candidates'][arm] if i in by_slot else [
                dict(index=j,status='skipped',error='earlier root/runtime failure') for j in range(contract.COUNTS[arm])]
            census.append(dict(root_slot=i,root_sha256=root,arm=arm,candidate_denominator=contract.COUNTS[arm],candidates=records))
    report=dict(schema=NAME+'.report',status='known_root_completed' if failure is None else 'failed_closed',
        wrapper_prepare_sha256=expected,main_review_sha256=main_review_sha256,root_denominator=4,
        arm_denominator=8,candidate_denominator=480,census=census,
        rows=[dict(root_slot=r['root_slot'],complete=r['complete'],file=f'root_{r["root_slot"]}.json',
                   sha256=file_sha(output/f'root_{r["root_slot"]}.json') if (output/f'root_{r["root_slot"]}.json').is_file() else None) for r in rows],
        error=failure,secondary_errors=secondary,model_unchanged=unchanged,source_unchanged=source_unchanged,
        model_before=before,model_after=after,
        new_physics_steps=0,P1_admission=False,P2_admission=False,
        actual_dispatch_scope='native host calls; no FLOPs/backward-engine/kernel profiler',
        physics_endpoint_cost=None,upstream_acquisition_cost=None)
    native.write(output/'report.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','audit-journal','execute-known'))
    parser.add_argument('--native-prepared'); parser.add_argument('--native-sha')
    parser.add_argument('--prepared'); parser.add_argument('--prepare-sha'); parser.add_argument('--output')
    parser.add_argument('--journal'); parser.add_argument('--main-review-sha'); parser.add_argument('--confirmatory',action='store_true')
    a=parser.parse_args(argv)
    if a.confirmatory: raise ValueError('confirmatory prohibited: independent P1/freshness not admitted')
    if a.command=='prepare':
        value=prepare(native_dir=a.native_prepared,native_sha=a.native_sha)
        require(Path(a.output).name.startswith(NAME), 'exclusive wrapper prepare stem')
        write_metadata(a.output,value)
        out=dict(prepare_sha256=C.digest(value),new_neural_calls=0,new_physics_steps=0)
    elif a.command=='verify':
        verify(contract.prior.read(a.prepared),a.prepare_sha); out=dict(status='verified_no_dispatch')
    elif a.command=='audit-journal':
        records=read_journal(a.journal); out=dict(events=len(records),last_sha256=records[-1]['event_sha256'],P2_admission=False)
    else:
        out=execute_known(contract.prior.read(a.prepared),a.prepare_sha,a.output,main_review_sha256=a.main_review_sha)
        out={k:v for k,v in out.items() if k not in ('census','rows')}
    print(encode(out).decode()); return 0


if __name__=='__main__': raise SystemExit(main())
