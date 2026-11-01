"""Exception-durable frontier v2; identical mathematical protocol, not efficacy evidence."""
from __future__ import annotations
import argparse
from copy import deepcopy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from experiments.spectral_reachability import p2_composition_predict_frontier_20261027_v1 as bridge
from experiments.spectral_reachability.p2_composition_predict_frontier_20261027_v1 import (
    PROJECT, ROOTS, ARMS, CAP, array, require, digest, candidate_bank,
    bound_candidate_bank, expected_entries, validate_entries, legacy_source_contract, summary)
NAME='p2_composition_predict_frontier_cuda_20261027_v2'


def contract():
    out=bridge.contract()
    out.update(status='runner_awaiting_main_CUDA_review',runner=NAME,
        exporter='NEW all-slot80/40 exporter; never old40 whitelist',
        exception_protocol='R1 first exception retained; secondary journal/restore/finalization failures separate',
        derivative_durability='R2 nominal and Jacobian fsynced individually before next AD/solve stage',
        efficacy_interpretation='exception hardening only; no explanation of v1 control outcomes',
        cached_preparation_cost='inherited unknown, not zero; saved-data verification is additional CPU work')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bank-json',type=Path);parser.add_argument('--root-sha256');parser.add_argument('--count',type=int,default=80)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--verify',action='store_true');mode.add_argument('--execute-cuda',action='store_true')
    mode.add_argument('--export-selected',action='store_true')
    parser.add_argument('--prepared',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--prepared-sha256');parser.add_argument('--reference',type=Path,default=PROJECT/'results/p2_composition_paired_20261027_v2_prepared01')
    a=parser.parse_args()
    if a.prepare:r=prepare(a.reference,a.output)
    elif a.verify:
        _,p,s=load_prepared(a.prepared,a.prepared_sha256);r=dict(status='verified',protocol_sha256=digest(p),prepared_sha256=digest(s))
    elif a.execute_cuda:r=cuda(a.prepared,a.output,a.prepared_sha256)
    elif a.export_selected:
        require(a.output is not None,'NEW export output JSON required');r=selected_actions(a.prepared);write(a.output,r)
    elif a.bank_json:
        require(a.root_sha256 is not None,'fixed root identity');require(a.bank_json.stat().st_size<=4*1024**2,'small supplied source bank only')
        r=bound_candidate_bank(json.loads(a.bank_json.read_text()),a.root_sha256,a.count)
    else:r=summary()
    print(json.dumps(r,sort_keys=True,indent=2,allow_nan=False))


def legacy():
    from experiments.spectral_reachability import p2_composition_paired_20261027_v2 as old
    return old


def error_record(stage,exc):
    try:message=str(exc)
    except BaseException:message='<exception text unavailable>'
    return dict(stage=stage,error_type=type(exc).__name__,error=message)


class Faults:
    """Keep the actual FIRST exception object, never replace it during cleanup."""
    def __init__(self):
        self.exceptions=[];self.records=[]

    @property
    def primary(self):return self.exceptions[0] if self.exceptions else None

    def capture(self,stage,exc):
        if not any(exc is e for e in self.exceptions):
            self.exceptions.append(exc);self.records.append(error_record(stage,exc))
            for child_stage,child in getattr(exc,'_frontier_secondary',()):self.capture(child_stage,child)

    def receipt(self):
        return dict(primary=self.records[0] if self.records else None,secondary=self.records[1:])

    def attempt(self,stage,callback):
        try:return callback()
        except BaseException as exc:self.capture(stage,exc);return None

    def journal(self,sink):
        """Record a pending kernel/solve exception BEFORE a failing finally sink.

        With no active exception a journal failure propagates and stops work.
        With one active, sink failure is secondary and the original propagation
        continues. This also protects the inherited mathematical SVD finally.
        """
        def emit(event):
            active=sys.exception()
            if active is not None:self.capture('active_during_journal',active)
            try:return sink(event)
            except BaseException as exc:
                self.capture('journal.'+str(event.get('kind','candidate')),exc)
                if active is None:raise
        return emit


def write(path,value):
    """Exclusive durable writes; failures leave evidence, never overwrite/retry."""
    faults=Faults();stream=None;fd=None
    try:
        stream=Path(path).open('x')
        json.dump(value,stream,allow_nan=False,separators=(',',':'));stream.flush();os.fsync(stream.fileno())
    except BaseException as exc:faults.capture('write.file',exc)
    finally:
        if stream is not None:faults.attempt('write.close_file',stream.close)
    if faults.primary is None:
        try:
            fd=os.open(str(Path(path).parent),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd)
        except BaseException as exc:faults.capture('write.parent_fsync',exc)
        finally:
            if fd is not None:faults.attempt('write.close_parent',lambda:os.close(fd))
    if faults.primary is not None:
        try:faults.primary._frontier_secondary=[(r['stage'],e) for r,e in zip(faults.records[1:],faults.exceptions[1:])]
        except BaseException:raise faults.primary
        raise faults.primary


def read(path,limit=96*1024**2):
    path=Path(path);require(not path.is_symlink() and path.is_file() and path.stat().st_size<=limit,'bounded regular saved JSON')
    def unique(pairs):
        d={}
        for k,v in pairs:
            require(k not in d,'duplicate JSON field');d[k]=v
        return d
    return json.loads(path.read_text(),object_pairs_hook=unique,parse_constant=lambda s:require(False,'nonfinite JSON'))


def provenance(p,s,ready,bank,arm):
    old=legacy();prov=old.sra.provenance_for(p,s,ready)
    prov['candidate_bank_sha256']=digest(dict(bank=bank,arm=arm,contract=contract()))
    return prov


def run_arm(*,arm,bank,root,provenance,action_domain,target_metric,replay,suffix=None,
            fixture=False,journal=lambda e:None,faults=None):
    """Actual dispatch: joint80 x2; conditional40 x(1+2AD-overhead+1)."""
    faults=Faults() if faults is None else faults;journal=faults.journal(journal)
    old=legacy();n=80 if arm==ARMS[0] else 40
    require(arm in ARMS and (fixture or type(replay) is old.full20.RootAnchoredCudaReplay),'supported arm and actual CUDA adapter required')
    require(arm!=ARMS[1] or callable(suffix),'own-context suffix compiler required')
    u=array(bank['logits'],(80,20));commands=array(bank['commands'],(80,2,5,2))
    require(np.array_equal(commands,np.tanh(u).astype(np.float32).reshape(80,2,5,2)) and np.max(np.linalg.norm(u,axis=1))<=CAP*(1+1e-12),'bank domain/mapping')
    require(action_domain['low']==[-1.,-1.] and action_domain['high']==[1.,1.] and action_domain['action_block']==5,'original action domain')
    initial=old.full20._validated_context(root['context']);ledger=old.full20._Ledger(replay,arm);attempts=[]
    old.saved_witness(root,provenance,action_domain,[initial,initial],
        [commands[0].reshape(10,2).tolist()],'metadata_predispatch')
    start=time.perf_counter();winner=None;error=None
    try:
        for i in range(n):
            event=dict(candidate_id=f'{arm}_{i:02d}',index=i,completed=False,metric_attempted=False)
            attempts.append(event);journal(dict(kind='candidate_request',arm=arm,index=i,completed=False))
            prefix=commands[i,0].tolist();mid=ledger.call(initial,prefix,'planning',[i,0])
            local=suffix(deepcopy(mid),u[i,:10].copy(),i) if arm==ARMS[1] else None
            v=array(local['suffix_logits'],(10,)) if local else u[i,10:]
            if local:
                require(local['input_context_sha256']==old.digest(mid),'suffix context identity');event['suffix_compilation']=local
            joint=np.concatenate((u[i,:10],v));require(np.linalg.norm(joint)<=CAP*(1+1e-12),'joint ball')
            second=np.tanh(v).astype(np.float32).reshape(5,2).tolist();actions=prefix+second
            end=ledger.call(mid,second,'planning',[i,1])
            contexts,chunks=([initial,mid,end],[prefix,second]) if local else ([initial,end],[actions])
            witness=old.saved_witness(root,provenance,action_domain,contexts,chunks,event['candidate_id'])
            event.update(actions=actions,logits=joint.tolist(),witness=witness,metric_attempted=True)
            metric_start=time.perf_counter();cost=old.full20._cost(target_metric(deepcopy(end)))
            event.update(cost=cost,selection_wall_time_s=time.perf_counter()-metric_start,completed=True);journal(deepcopy(event))
        best=min(attempts,key=lambda e:e['cost']);actions=best['actions']
        mid=ledger.call(initial,actions[:5],'replay',[n]);end=ledger.call(mid,actions[5:],'replay',[n+1])
        whole=ledger.call(initial,actions,'replay',[n+2])
        require(end==whole==best['witness']['receipt']['terminal_context'],'double replay/search exactness')
        if arm==ARMS[1]:require(mid==best['witness']['edges'][0]['output_context'],'conditional prefix replay exact')
        winner=dict(candidate_id=best['candidate_id'],cost=best['cost'],witness=deepcopy(best['witness']),
            independent_full_replay_exact=True,segmented_contexts=[initial,mid,end],whole_context=whole)
    except BaseException as exc:
        faults.capture('arm.'+arm,exc)
        record=error_record('arm',exc);error={k:record[k] for k in ('error_type','error')};journal(dict(kind='arm_failure',arm=arm,**error))
    return dict(arm=arm,fixture_only=fixture,attempts=attempts,ledger=ledger.events,winner=winner,error=error,
        complete=winner is not None,selection_evaluations=sum(e['metric_attempted'] for e in attempts),
        nested40_minimum=min(e['cost'] for e in attempts[:40]) if len(attempts)>=40 and all(e['completed'] for e in attempts[:40]) else None,
        wall_time_s=time.perf_counter()-start,faults=faults.receipt())


def expected_scopes():
    old=legacy();out=[]
    def packet(scope):out.extend([deepcopy(scope),deepcopy(scope)])
    for stage in ('nominal','joint_jacobian'):
        for _ in range(2):packet(dict(arm=old.sra.ARM,phase='compilation',path=[stage]))
    for arm,n in zip(ARMS,(80,40)):
        for i in range(n):
            packet(dict(arm=arm,phase='planning',path=[i,0]))
            if arm==ARMS[1]:
                for stage in ('nominal','jacobian'):packet(dict(arm=arm,phase='suffix_compilation',path=[i,stage]))
            packet(dict(arm=arm,phase='planning',path=[i,1]))
        for k in (n,n+1,n+2,n+2):packet(dict(arm=arm,phase='replay',path=[k]))
    return out


@contextmanager
def guarded_entries(model,replay,journal,faults=None):
    """Predispatch ordered dual-entry cap, nonrefundable durable attempt ledger.

    Metadata only: no tensor arithmetic. No extra encoder is allowed. Failure
    poisons this root; later calls are blocked, not used to fill the budget.
    """
    faults=Faults() if faults is None else faults;journal=faults.journal(journal)
    specs=expected_entries(80);scopes=expected_scopes();events=[];restored=[];poisoned=False
    try:
        targets=[(model,'predict','predict'),(model.action_encoder,'forward','action_encoder')]
        if hasattr(model,'encoder'):targets.append((model.encoder,'forward','forbidden_visual_encoder'))
        for owner,name,kind in targets:
            original=getattr(owner,name);restored.append((owner,name,name in vars(owner),vars(owner).get(name)))
            def wrap(original,kind):
                def counted(*args,**kw):
                    nonlocal poisoned
                    index=len(events);scope=deepcopy(getattr(replay,'_p2_scope',None))
                    meta=dict(kind=kind,scope=scope,completed=False,index=index)
                    try:
                        require(not poisoned and index<len(specs),'poisoned or exhausted forward cap')
                        spec=specs[index];x=args[0] if args else kw.get('x')
                        shape=tuple(x.shape);meta.update(samples=shape[0],context_tokens=shape[0]*shape[1])
                        require(kind==spec['kind'] and scope==scopes[index],'unexpected forward scope/order')
                        require(shape==(1,3,192 if kind=='predict' else 10) and x.device.type=='cuda','single sample original CUDA context required')
                    except BaseException as exc:
                        faults.capture('guard.metadata',exc)
                        poisoned=True;meta.update(blocked=True,error=str(exc));journal(dict(kind='blocked_entry',entry=meta));raise
                    events.append(meta)
                    try:journal(dict(kind='entry_attempt',entry=deepcopy(meta)))
                    except BaseException:
                        poisoned=True;raise
                    try:
                        result=original(*args,**kw);meta['completed']=True;return result
                    except BaseException as exc:
                        faults.capture('guard.forward.'+kind,exc)
                        poisoned=True;meta.update(error_type=type(exc).__name__,error=str(exc));raise
                    finally:
                        try:journal(dict(kind='entry_terminal',entry=deepcopy(meta)))
                        except BaseException:
                            poisoned=True;raise
                return counted
            setattr(owner,name,wrap(original,kind))
        yield events
    except BaseException as exc:
        faults.capture('guard.body',exc);raise
    finally:
        active=sys.exception();restore_first=None
        if active is not None:faults.capture('guard.unwind',active)
        for owner,name,existed,value in reversed(restored):
            try:
                if existed:setattr(owner,name,value)
                else:delattr(owner,name)
            except BaseException as exc:
                if restore_first is None:restore_first=exc
                faults.capture('guard.restore.'+name,exc)
        if active is None and restore_first is not None and faults.primary is restore_first:raise restore_first


def validate_actual(events):
    require(len(events)==664,'all actual dual-entry slots required')
    expected=expected_entries(80);scopes=expected_scopes()
    for i,e in enumerate(events):
        require(e['index']==i and e['kind']==expected[i]['kind'] and e['scope']==scopes[i] and e['completed'] is True
            and e['samples']==1 and e['context_tokens']==3 and not e.get('error'),'actual entry census drift')
    return validate_entries(expected,80)


def validate_operations(operations,results):
    """Each requested AD/SVD/solve has one completed terminal receipt."""
    old=legacy();specs=[('jacobian',None),('bounded_ridge_svd',None),('mode_svd',None)]
    specs += [(k,i) for i in range(40) for k in ('suffix_jacobian','suffix_solve')]
    require(len(operations)==2*len(specs),'all operation requests and terminals required')
    for j,(kind,index) in enumerate(specs):
        request,terminal=operations[2*j:2*j+2]
        require(request['kind']==terminal['kind']==kind and request.get('index')==terminal.get('index')==index
            and request['completed'] is False and terminal['completed'] is True,'operation dispatch/completion order')
        if kind in ('jacobian','suffix_jacobian'):
            require(request['shape']==terminal['shape']==[192,20 if kind=='jacobian' else 10]
                and request['vectorize'] is True and terminal['vectorize'] is True,'actual derivative contract')
    counts=old.operation_counts(operations)
    locals_=[e['suffix_compilation'] for r in results if r['arm']==ARMS[1] for e in r['attempts']]
    require(len(locals_)==40 and counts==dict(jacobian_calls=41,scalar_output_vjp_directions=7872,
        svd_calls=2+sum(e['svd_calls'] for e in locals_),bounded_solve_calls=1+sum(e['bounded_solve_calls'] for e in locals_)),'AD/SVD/solve actual total')
    for i,local in enumerate(locals_):
        request,terminal=operations[6+4*i+2:6+4*i+4]
        require(request['svd_calls_requested']==terminal['svd_calls']==local['svd_calls']
            and terminal['bounded_solve_calls']==local['bounded_solve_calls'],'individual suffix operation counts')
    return counts


def persist_array(output,name,value,binding,receipts,journal):
    """Persist a completed array before the NEXT expensive computation.

    Float64 is a storage conversion only, not a new model evaluation. The file
    remains even when receipt journaling or a subsequent SVD/solve fails.
    """
    a=np.asarray(value,dtype=np.float64)
    require(a.shape in ((192,),(192,20),(192,10)) and np.isfinite(a).all(),'finite derivative array')
    require(name.replace('_','').isalnum(),'fixed array identifier')
    path=Path(output)/'arrays'/f'{name}.json'
    body=dict(schema=NAME+'.array',name=name,shape=list(a.shape),binding=deepcopy(binding),value=a.tolist())
    write(path,body)
    receipt=dict(name=name,shape=list(a.shape),binding=deepcopy(binding),sha256=file_sha(path))
    receipts.append(receipt);journal(dict(kind='array_saved',receipt=deepcopy(receipt)))
    return a


def derivative_stages(*,nominal,jacobian,convert,persist,solve,journal,faults,stages,kind,index=None):
    """Pure orchestration seam used by real CUDA AND NumPy fault tests."""
    journal=faults.journal(journal)
    def save(stage,value):
        event=dict(kind=kind,index=index,stage=stage,returned=True,converted=False,persisted=False)
        stages.append(event)
        a=convert(value);event['converted']=True
        saved=persist(stage,a);event['persisted']=True
        return saved
    n=save('nominal',nominal())  # Must finish fsync BEFORE calling Jacobian.
    request=dict(kind=kind,completed=False,shape=[192,20 if kind=='jacobian' else 10],vectorize=True)
    if index is not None:request['index']=index
    if kind=='jacobian':request['scalar_output_vjp_directions_requested']=192
    journal(deepcopy(request))
    j=jacobian()
    # Save before terminal journaling too: a broken journal must not discard J.
    j=save('jacobian',j);request['completed']=True;journal(deepcopy(request))
    return solve(j,n)


@contextmanager
def scoped(replay,value,faults):
    previous=getattr(replay,'_p2_scope',None)
    try:
        replay._p2_scope=value;yield
    except BaseException as exc:
        faults.capture('primitive.'+str(value),exc);raise
    finally:
        active=sys.exception()
        if active is not None:faults.capture('primitive.unwind',active)
        try:replay._p2_scope=previous
        except BaseException as exc:
            faults.capture('primitive.restore_scope',exc)
            if active is None:raise


def compile_cuda(model,device,ready,compiled,replay,journal,save,faults,stages):
    """Identical joint-H2 math to v1; arrays durable before old spectral solve."""
    require(getattr(device,'type',None)=='cuda','host CUDA only')
    import torch
    old=legacy();transition=old.compiler.make_transition(model,device);c=ready['root']['context']
    emb=torch.tensor(c['latent_history'],dtype=torch.float32,device=device)[None]
    history=torch.tensor(c['action_history'],dtype=torch.float32,device=device)[None];zero=emb.new_zeros(20)
    def terminal(u):
        state,between=emb,history;raw=torch.tanh(u).reshape(2,5,2)
        for packet in range(2):state,between=transition(state,between,raw[packet])
        return state[0,-1]
    def nominal():
        with scoped(replay,dict(arm=old.sra.ARM,phase='compilation',path=['nominal']),faults),torch.no_grad():return terminal(zero)
    def jacobian():
        with scoped(replay,dict(arm=old.sra.ARM,phase='compilation',path=['joint_jacobian']),faults):
            return torch.autograd.functional.jacobian(terminal,zero.requires_grad_(True),vectorize=True)
    return derivative_stages(nominal=nominal,jacobian=jacobian,convert=lambda t:t.detach().cpu().numpy().astype(np.float64),
        persist=save,solve=lambda j,n:old.sra.spectral_bank(j,n,compiled['goal_vectors'][0],compiled['scoring']['metric']['projection'],journal=faults.journal(journal)),
        journal=journal,faults=faults,stages=stages,kind='jacobian')


def suffix_cuda(model,device,replay,context,prefix,compiled,index,journal,save,faults,stages):
    """Same 192x10 own-prefix residual/cap and ridge as v1; earlier persistence."""
    require(getattr(device,'type',None)=='cuda','host CUDA only')
    import torch
    old=legacy();transition=old.compiler.make_transition(model,device)
    emb=torch.tensor(context['latent_history'],dtype=torch.float32,device=device)[None]
    history=torch.tensor(context['action_history'],dtype=torch.float32,device=device)[None];zero=emb.new_zeros(10)
    def terminal(u):
        state,_=transition(emb,history,torch.tanh(u).reshape(5,2));return state[0,-1]
    def nominal():
        with scoped(replay,dict(arm=ARMS[1],phase='suffix_compilation',path=[index,'nominal']),faults),torch.no_grad():return terminal(zero)
    def jacobian():
        with scoped(replay,dict(arm=ARMS[1],phase='suffix_compilation',path=[index,'jacobian']),faults):
            return torch.autograd.functional.jacobian(terminal,zero.requires_grad_(True),vectorize=True)
    def solve(j,n):
        emit=faults.journal(journal);op=dict(kind='suffix_solve',index=index,completed=False,svd_calls_requested=int(CAP**2-float(prefix@prefix)>0))
        emit(deepcopy(op));start=time.perf_counter()
        result=old.suffix_solution(prefix,j,n,compiled['goal_vectors'][0],compiled['scoring']['metric']['projection'])
        result['input_context_sha256']=old.digest(context)
        op.update(completed=True,svd_calls=result['svd_calls'],bounded_solve_calls=result['bounded_solve_calls'],cpu_transfer_and_solve_wall_time_s=time.perf_counter()-start)
        emit(op);return result
    return derivative_stages(nominal=nominal,jacobian=jacobian,convert=lambda t:t.detach().cpu().numpy().astype(np.float64),persist=save,
        solve=solve,journal=journal,faults=faults,stages=stages,kind='suffix_jacobian',index=index)


def prepare(reference,output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal
    require(output is not None,'new prepare output required');output=Path(output).resolve()
    require(NAME in output.name and not output.exists(),'new exclusive tagged prepare path')
    old=legacy();ref,p,s=old.load_prepared(reference);legacy_source_contract()
    require(tuple(c['root_sha256'] for c in p['context']['cases'])==ROOTS,'original root order/identities')
    files=set(map(Path,p['source_paths']))|{Path(__file__),Path(bridge.__file__),
        PROJECT/'tests'/f'test_{NAME}.py',PROJECT/'innerdocs'/f'{NAME}.md',
        PROJECT/'tests'/f'test_{bridge.NAME}.py',PROJECT/'innerdocs'/f'{bridge.NAME}.md',
        PROJECT/'tests/test_p2_composition_predict_frontier_cuda_20261027_v1.py',
        PROJECT/'experiments/spectral_reachability/p2_composition_predict_frontier_cuda_20261027_v1.py',
        PROJECT/'innerdocs/p2_composition_predict_frontier_cuda_20261027_v1.md'}
    # The inherited closure contains the seal, model, transition and CUDA loader.
    files=sorted(f.resolve() for f in files)
    require(sum(f.stat().st_size for f in files)<16*1024**2,'small source closure')
    require(shutil.disk_usage(output.parent).free>220*1024**2,'retain 200MiB finalization reserve plus prepare')
    protocol=dict(schema=NAME+'.protocol',contract=contract(),context=p['context'],reference=str(ref),
        reference_protocol_sha256=old.digest(p),reference_prepared_sha256=old.digest(s),
        source_paths=list(map(str,files)),bindings=p['bindings'],frozen_before_new_frontier_outcomes=True)
    output.mkdir();write(output/'protocol.json',protocol)
    seals=[create_seal(files=files[i:i+40],args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
        output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
    prepared=dict(schema=NAME+'.prepared',protocol_sha256=digest(protocol),source_seals=seals,status='prepared_not_executed',new_neural_calls=0,P2_admission=False)
    write(output/'prepared.json',prepared);load_prepared(output,digest(prepared))
    return dict(status='prepared_not_executed',prepared=str(output),prepared_sha256=digest(prepared),protocol_sha256=digest(protocol))


def load_prepared(output,expected_sha=None):
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    require(output is not None,'prepared path required');output=Path(output).resolve();p=read(output/'protocol.json');s=read(output/'prepared.json')
    require(p['schema']==NAME+'.protocol' and s['schema']==NAME+'.prepared' and s['protocol_sha256']==digest(p)
        and p['contract']==contract() and p['frozen_before_new_frontier_outcomes'] is True,'protocol drift')
    if expected_sha is not None:require(digest(s)==expected_sha,'external prepared SHA drift')
    paths=[]
    for seal in s['source_seals']:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        directory=Path(seal['output_dir']);require(read(directory/'args.json')==dict(protocol_sha256=digest(p)),'seal protocol binding')
        paths.extend(e['source'] for e in read(directory/'manifest.json')['files'])
    require(sorted(paths)==p['source_paths'] and str(Path(__file__).resolve()) in paths,'source closure')
    old=legacy();_,rp,rs=old.load_prepared(p['reference'])
    require(old.digest(rp)==p['reference_protocol_sha256'] and old.digest(rs)==p['reference_prepared_sha256'] and rp['context']==p['context'],'inherited context/source changed')
    require(tuple(c['root_sha256'] for c in p['context']['cases'])==ROOTS,'root census');legacy_source_contract()
    return output,p,s


def finalized_run(output,binding,work,source_check,model_check,writer=None):
    """Post-directory envelope: every recoverable failure reaches finalization.

    Source and model checks AND their artifact writes are independent. There is
    no durability guarantee under disk/device failure or SIGKILL; report writing
    is attempted once, with a distinct emergency receipt if that write fails.
    """
    writer=write if writer is None else writer;faults=Faults()
    state=dict(rows=[],row_cache=[],host=None,model_load_wall_time_s=None)
    try:
        writer(Path(output)/'execution.json',binding);work(state,faults)
    except BaseException as exc:faults.capture(state.get('execution_stage','execution'),exc)
    if faults.primary is None and (len(state['rows'])!=4 or not all(r['complete'] for r in state['rows'])):
        faults.capture('execution.incomplete',ValueError('four complete root receipts required'))
    checks={}
    for name,callback in (('source',source_check),('model',model_check)):
        try:
            value=callback(state);require(value is not None,'finalization unavailable')
            require((value.get('source_unchanged') is True) if name=='source' else
                (value.get('model_unchanged') is True and value.get('frozen_eval') is True),'finalization check returned false')
            receipt=dict(status='passed',result=value)
        except BaseException as exc:
            faults.capture('finalization.'+name,exc);receipt=dict(status='failed',error=error_record(name,exc))
        checks[name]=receipt
        faults.attempt('finalization.write.'+name,lambda n=name,r=receipt:writer(Path(output)/(n+'_finalization.json'),r))
    try:report=build_report(binding,state,checks,faults)
    except BaseException as exc:
        faults.capture('finalization.report_build',exc)
        report=dict(schema=NAME+'.report',status='incomplete_or_failed',**binding,rows=state['rows'],
            faults=faults.receipt(),finalization=checks,resource_totals=None,
            summary_error='resource/report aggregation failed; no fabricated zero budget',
            P1_admission=False,P2_admission=False,total_resource_matched=False)
    try:writer(Path(output)/'report.json',report)
    except BaseException as exc:
        faults.capture('finalization.write.report',exc)
    if faults.primary is None:
        faults.attempt('finalization.commit',lambda:writer(Path(output)/'completion.json',
            dict(schema=NAME+'.completion',report_sha256=file_sha(Path(output)/'report.json'),binding=binding)))
    if faults.primary is not None:
        faults.attempt('finalization.write.emergency',lambda:writer(Path(output)/'finalization_failure.json',
            dict(schema=NAME+'.finalization_failure',status='incomplete_or_failed',faults=faults.receipt(),binding=binding)))
    if faults.primary is not None:raise faults.primary
    return report


def build_report(binding,state,checks,faults):
    rows=state['row_cache']
    totals={k:sum(r['completed_operations'][k] for r in rows) for k in
        ('jacobian_calls','scalar_output_vjp_directions','svd_calls','bounded_solve_calls')}
    complete=(faults.primary is None and len(state['rows'])==4 and all(r['complete'] for r in state['rows'])
        and all(c['status']=='passed' for c in checks.values()))
    model=checks['model'].get('result',{})
    report=dict(schema=NAME+'.report',status='cuda_completed' if complete else 'incomplete_or_failed',
        **binding,rows=state['rows'],host=state['host'],finalization=checks,faults=faults.receipt(),
        model_unchanged=model.get('model_unchanged'),frozen_eval=model.get('frozen_eval'),
        model_load_wall_time_s=state['model_load_wall_time_s'],
        actual_unique_predict_calls=sum(r['resource_receipt']['unique']['predict_calls'] for r in rows),
        actual_unique_action_encoder_calls=sum(r['resource_receipt']['unique']['action_encoder_calls'] for r in rows),
        completed_operations=totals,completed_selection_evaluations=sum(sum(x['selection_evaluations'] for x in r['results']) for r in rows),
        P1_admission=False,P2_admission=False,total_resource_matched=False,
        context_cost='inherited RGB/goal/Q/context costs unknown, not zero; fresh root compile charged',
        validation_cost='saved-array export checks/SVD/solves plus new immediate persistence CPU/I/O, not inference credit',
        interpretation='exception durability only; no inferred improvement of v1 normal control performance')
    return report


def array_binding(p,ready,compiled,prepared_sha):
    old=legacy()
    return dict(root_sha256=ready['root_sha256'],root_index=ready['root_index'],
        root_context_sha256=old.digest(ready['root']['context']),goal_sha256=old.digest(compiled['goal_vectors'][0]),
        projection_sha256=old.digest(compiled['scoring']['metric']['projection']),
        prepared_sha256=prepared_sha,protocol_sha256=digest(p))


def commit_row(output,row,state,faults):
    """Row storage/hash failure cannot prevent outer source/model finalization."""
    path=Path(output)/f'root_{row["root_index"]}.json'
    faults.attempt('row.write',lambda:write(path,row))
    canonical=faults.attempt('row.hash',lambda:digest(row))
    raw=faults.attempt('row.file_hash',lambda:file_sha(path))
    state['row_cache'].append(row)
    state['rows'].append(dict(root_sha256=row['root_sha256'],root_index=row['root_index'],
        row_sha256=canonical,row_file_sha256=raw,complete=row['complete']))


def run_root(output,p,s,prepared_sha,ready,compiled,model,device,synchronize,faults):
    old=legacy();start=time.perf_counter();index=ready['root_index'];operations=[];results=[];entries=[]
    arrays=[];stages=[];compile_s=None;generation_s=None;stream=None;bank=None
    def raw_journal(event):
        if event.get('kind') in ('jacobian','suffix_jacobian','bounded_ridge_svd','mode_svd','suffix_solve'):
            operations.append(deepcopy(event))
        stream.write(json.dumps(event,allow_nan=False,separators=(',',':'))+'\n')
        stream.flush();os.fsync(stream.fileno())
    journal=faults.journal(raw_journal)
    try:
        stream=(Path(output)/f'root_{index}_journal.jsonl').open('x')
        replay=old.full20.RootAnchoredCudaReplay(model,device)
        binding=array_binding(p,ready,compiled,prepared_sha)
        def shared_save(stage,value):
            return persist_array(output,f'root_{index}_shared_{stage}',value,binding,arrays,journal)
        def suffix(context,prefix,i):
            local=dict(binding,input_context_sha256=old.digest(context),prefix_logits_sha256=old.digest(prefix.tolist()),index=i)
            def save(stage,value):
                return persist_array(output,f'root_{index}_suffix_{i}_{stage}',value,local,arrays,journal)
            return suffix_cuda(model,device,replay,context,prefix,compiled,i,journal,save,faults,stages)
        with guarded_entries(model,replay,journal,faults) as entries:
            t=time.perf_counter();basebank=compile_cuda(model,device,ready,compiled,replay,journal,shared_save,faults,stages)
            compile_s=time.perf_counter()-t
            t=time.perf_counter();bank=candidate_bank(basebank,ready['root_sha256']);generation_s=time.perf_counter()-t
            write(Path(output)/f'root_{index}_bank.json',dict(base=basebank,expanded=bank))
            for arm in ARMS:
                result=run_arm(arm=arm,bank=bank,root=ready['root'],provenance=provenance(p,s,ready,bank,arm),
                    action_domain=ready['action_domain'],target_metric=old.full20.metric_for(compiled),replay=replay,
                    suffix=suffix,journal=journal,faults=faults)
                results.append(result)
                if not result['complete']:
                    if faults.primary is not None:raise faults.primary
                    raise ValueError('incomplete arm')
        validate_actual(entries);validate_operations(operations,results)
    except BaseException as exc:faults.capture('root.'+str(index),exc)
    finally:
        if stream is not None:faults.attempt('root.journal.close',stream.close)
    faults.attempt('root.synchronize',synchronize)
    journal_sha=faults.attempt('root.journal.hash',lambda:file_sha(Path(output)/f'root_{index}_journal.jsonl'))
    bank_sha=faults.attempt('root.bank.hash',lambda:file_sha(Path(output)/f'root_{index}_bank.json')) if bank is not None else None
    row=dict(root_sha256=ready['root_sha256'],root_index=index,results=results,actual_model_entries=entries,
        operations=operations,completed_operations=old.operation_counts(operations),error=faults.receipt()['primary'],
        faults=faults.receipt(),array_receipts=arrays,derivative_stages=stages,
        resource_receipt=old.resource_receipt(entries,results,operations),journal_sha256=journal_sha,bank_sha256=bank_sha,
        shared_compile_wall_time_s=compile_s,generation_wall_time_s=generation_s,
        synchronized_root_wall_time_s=time.perf_counter()-start,
        complete=faults.primary is None and len(results)==2 and all(r['complete'] for r in results))
    return row


def cuda(prepared,output,expected_sha):
    """Main only. No CPU neural fallback, retry, or cross-version reseal."""
    require(expected_sha is not None,'external prepared SHA required')
    directory,p,s=load_prepared(prepared,expected_sha);require(output is not None,'NEW execution output required')
    output=Path(output).resolve();require(NAME in output.name and not output.exists(),'exclusive execution path')
    require(shutil.disk_usage(output.parent).free>310*1024**2,'110MiB array/artifact allowance +200MiB reserve')
    output.mkdir()
    binding=dict(prepared=str(directory),prepared_sha256=expected_sha,protocol_sha256=digest(p))
    runtime={}
    def work(state,faults):
        state['execution_stage']='runtime.initialize'
        (output/'arrays').mkdir()
        from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
        from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
        state['host']=base.require_cuda_allocation('cuda')
        import torch
        device=torch.device('cuda:0');torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
        state['execution_stage']='model.load'
        model=base.load_compatible_pretrained(Path(p['context']['policy']),device=device).eval().requires_grad_(False)
        state['model_load_wall_time_s']=time.perf_counter()-start
        state['execution_stage']='model.initial_hash'
        runtime.update(model=model,checked=checked,before=checked.tensor_state_hash(model))
        state['execution_stage']='source.compilation_read'
        source=read(p['context']['compilation']);compiled={r['root_sha256']:r['compiled'] for r in source['rows']}
        for ready in p['context']['cases']:
            state['execution_stage']='root.'+str(ready['root_index'])
            row=run_root(output,p,s,expected_sha,ready,compiled[ready['root_sha256']],model,device,
                lambda:torch.cuda.synchronize(device),faults)
            commit_row(output,row,state,faults)
            if faults.primary is not None:break
    def source_check(state):
        load_prepared(directory,expected_sha);return dict(source_unchanged=True,protocol_sha256=digest(p))
    def model_check(state):
        require('model' in runtime and 'before' in runtime,'model or initial model hash unavailable')
        model=runtime['model']
        unchanged=runtime['before']==runtime['checked'].tensor_state_hash(model)
        frozen=all(not m.training for m in model.modules()) and all(not t.requires_grad for t in model.parameters())
        require(unchanged and frozen,'model/frozen-eval finalization failed')
        return dict(model_unchanged=unchanged,frozen_eval=frozen)
    return finalized_run(output,binding,work,source_check,model_check)


def file_sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_result(result,*,arm,bank,ready,prov,compiled):
    """Validate EVERY evaluated slot and both saved winner replay paths.

    Saved transition bindings are not another neural replay. The independent
    CUDA replay is executed by run_arm and its saved exact contexts checked.
    """
    old=legacy();n=80 if arm==ARMS[0] else 40
    require(result['arm']==arm and result['complete'] is True and result['fixture_only'] is False and result['error'] is None,'complete actual arm only')
    require(result['faults']==dict(primary=None,secondary=[]),'no arm diagnostic fault may be exported')
    require(len(result['attempts'])==n and result['selection_evaluations']==n,'all genuine candidate evaluations')
    initial=ready['root']['context'];metric=old.full20.metric_for(compiled)
    for i,event in enumerate(result['attempts']):
        require(event['index']==i and event['candidate_id']==f'{arm}_{i:02d}' and event['completed'] is True and event['metric_attempted'] is True,'slot identity/completion')
        u=array(bank['logits'][i],(20,));w=event['witness'];edges=w['edges']
        require(len(edges)==(1 if arm==ARMS[0] else 2),'arm witness edge count')
        if arm==ARMS[1]:
            local=event['suffix_compilation'];mid=edges[0]['output_context']
            require(local['input_context_sha256']==old.digest(mid),'actual conditional compile context binding')
            expected=old.suffix_solution(u[:10],local['jacobian'],local['nominal'],compiled['goal_vectors'][0],compiled['scoring']['metric']['projection'])
            expected['input_context_sha256']=old.digest(mid)
            require(local==expected,'all suffix saved derivatives/solve/cap binding')
            u=np.concatenate((u[:10],np.asarray(local['suffix_logits'])))
        actions=np.tanh(u).astype(np.float32).reshape(10,2).tolist()
        require(event['logits']==u.tolist() and event['actions']==actions,'full10 selected raw action mapping')
        terminal=w['receipt']['terminal_context']
        require(terminal['action_history']==[_normalized_token(actions[:5]),_normalized_token(actions[5:])]
            and terminal['latent_history'][0]==initial['latent_history'][-1],'candidate H2 boundary history')
        if arm==ARMS[1]:
            mid=edges[0]['output_context']
            require(mid['latent_history'][:2]==initial['latent_history'][1:] and mid['action_history']==initial['action_history'][1:]+[_normalized_token(actions[:5])]
                and terminal['latent_history'][:2]==mid['latent_history'][1:],'candidate conditional prefix/full boundary history')
        contexts,chunks=([initial,edges[0]['output_context'],terminal],[actions[:5],actions[5:]]) if arm==ARMS[1] else ([initial,terminal],[actions])
        require(w==old.saved_witness(ready['root'],prov,ready['action_domain'],contexts,chunks,event['candidate_id']),'all-slot witness provenance/context/action exactness')
        require(event['cost']==old.full20._cost(metric(deepcopy(terminal))),'all-slot original nonlinear predicted Q')
    best=min(result['attempts'],key=lambda e:e['cost']);winner=result['winner']
    require(winner['candidate_id']==best['candidate_id'] and winner['cost']==best['cost'] and winner['witness']==best['witness']
        and winner['independent_full_replay_exact'] is True,'stable all-candidate minimum/winner')
    contexts=winner['segmented_contexts'];terminal=best['witness']['receipt']['terminal_context']
    require(len(contexts)==3 and contexts[0]==initial and contexts[2]==terminal==winner['whole_context'],'saved double replay full contexts')
    old.full20._validated_context(contexts[1],initial)
    require(contexts[1]['latent_history'][:2]==initial['latent_history'][1:]
        and contexts[2]['latent_history'][:2]==contexts[1]['latent_history'][1:]
        and contexts[1]['action_history'][:1]==initial['action_history'][1:],'winner full recurrence history')
    if arm==ARMS[1]:require(contexts[1]==best['witness']['edges'][0]['output_context'],'saved conditional prefix replay')
    require(contexts[1]['action_history'][-1]==_normalized_token(best['actions'][:5]) and contexts[2]['action_history'][-1]==_normalized_token(best['actions'][5:]),'winner normalized incoming action history')
    ledger=result['ledger'];require(len(ledger)==2*n+4,'packet ledger count')
    expected=[dict(phase='planning',path=[i,p],packet=0) for i in range(n) for p in range(2)]
    expected += [dict(phase='replay',path=[n],packet=0),dict(phase='replay',path=[n+1],packet=0),dict(phase='replay',path=[n+2],packet=0),dict(phase='replay',path=[n+2],packet=1)]
    for event,spec in zip(ledger,expected):
        require(all(event[k]==v for k,v in spec.items()) and event['completed'] is True and event['predict_calls']==event['predict_samples']==event['action_encoder_calls']==1,'ledger path/sample exactness')
    require(result['nested40_minimum']==min(e['cost'] for e in result['attempts'][:40]),'nested40 diagnostic only')
    return best


def _normalized_token(actions):
    # Match public packets.normalize_live_packet float32 arithmetic (saved-data
    # only); no encoder, predictor, physics, or physical labels are involved.
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    raw=np.asarray(actions,dtype=np.float32)
    mean=np.asarray(p0.PUBLIC_ACTION_MEAN,dtype=np.float32)
    std=np.asarray(p0.PUBLIC_ACTION_STD,dtype=np.float32)
    if mean.shape==(2,):mean=np.tile(mean,5)
    if std.shape==(2,):std=np.tile(std,5)
    return ((raw.reshape(10)-mean)/std).tolist()


def validate_array_artifacts(output,p,prepared_sha,ready,compiled,saved,row):
    old=legacy();i=ready['root_index'];binding=array_binding(p,ready,compiled,prepared_sha)
    expected=[(f'root_{i}_shared_nominal',saved['base']['nominal'],binding),
        (f'root_{i}_shared_jacobian',saved['base']['jacobian'],binding)]
    stages=[dict(kind='jacobian',index=None,stage=s,returned=True,converted=True,persisted=True) for s in ('nominal','jacobian')]
    require([r['arm'] for r in row['results']]==list(ARMS),'ordered actual arms')
    for index,event in enumerate(row['results'][1]['attempts']):
        local=event['suffix_compilation'];prefix=saved['expanded']['logits'][index][:10]
        context=event['witness']['edges'][0]['output_context']
        b=dict(binding,input_context_sha256=old.digest(context),prefix_logits_sha256=old.digest(prefix),index=index)
        for stage in ('nominal','jacobian'):
            expected.append((f'root_{i}_suffix_{index}_{stage}',local[stage],b))
            stages.append(dict(kind='suffix_jacobian',index=index,stage=stage,returned=True,converted=True,persisted=True))
    require(len(expected)==82 and len(row['array_receipts'])==82 and row['derivative_stages']==stages,'82 immediate derivative stages/root')
    for receipt,(name,value,b) in zip(row['array_receipts'],expected):
        path=Path(output)/'arrays'/f'{name}.json';body=read(path,512*1024)
        shape=list(np.asarray(value).shape)
        require(receipt==dict(name=name,shape=shape,binding=b,sha256=file_sha(path))
            and body==dict(schema=NAME+'.array',name=name,shape=shape,binding=b,value=value),'root/goal/Q/context-bound immediate derivative array')


def expected_array_progress():
    """Array save journal boundaries relative to actual entries/AD/solve ops."""
    out=[(4,0),(8,1)];entries=expected_entries(80)
    for i in range(40):
        for role,ops in (('suffix_nominal',6+4*i),('suffix_Jacforward',7+4*i)):
            end=max(j+1 for j,e in enumerate(entries) if e['scope']==ARMS[1] and e['index']==i and e['role']==role)
            out.append((end,ops))
    return out


def selected_actions(output):
    """New80/40 exporter, not the old40 whitelist. All eight arms or refuse."""
    started=time.perf_counter();output=Path(output).resolve();report=read(output/'report.json');execution=read(output/'execution.json')
    require(not (output/'finalization_failure.json').exists(),'failed finalization retained; no export')
    require(read(output/'completion.json')==dict(schema=NAME+'.completion',report_sha256=file_sha(output/'report.json'),binding=execution),'final report commit binding')
    require(report['schema']==NAME+'.report' and report['status']=='cuda_completed' and report['model_unchanged'] is True
        and report['frozen_eval'] is True and report['P1_admission'] is False and report['P2_admission'] is False
        and report['total_resource_matched'] is False,'completed source-bound actual run required')
    require(report['faults']==dict(primary=None,secondary=[]),'exception-bearing run cannot export')
    require(report['finalization']['source']['status']==report['finalization']['model']['status']=='passed'
        and report['finalization']['source']['result']==dict(source_unchanged=True,protocol_sha256=report['protocol_sha256'])
        and report['finalization']['model']['result']==dict(model_unchanged=True,frozen_eval=True),'separate successful finalization required')
    for name in ('source','model'):
        require(read(output/(name+'_finalization.json'))==report['finalization'][name],'independent finalization artifact binding')
    require(execution=={k:report[k] for k in ('prepared','prepared_sha256','protocol_sha256')},'execution/prepared binding')
    _,p,s=load_prepared(report['prepared'],report['prepared_sha256']);require(report['protocol_sha256']==digest(p),'report protocol binding')
    require(len(report['rows'])==4,'four root report census')
    old=legacy();source=read(p['context']['compilation']);compiled={r['root_sha256']:r['compiled'] for r in source['rows']};plans=[];counts=[]
    for ready,receipt in zip(p['context']['cases'],report['rows']):
        i=ready['root_index'];row=read(output/f'root_{i}.json')
        require(receipt==dict(root_sha256=ready['root_sha256'],root_index=i,row_sha256=digest(row),row_file_sha256=file_sha(output/f'root_{i}.json'),complete=True)
            and row['root_sha256']==ready['root_sha256'] and row['root_index']==i and row['complete'] is True and row['error'] is None,'stable root row binding/census')
        require(row['faults']==dict(primary=None,secondary=[]),'exception-bearing row cannot export')
        budget=validate_actual(row['actual_model_entries']);counts.append(row['completed_operations'])
        require(file_sha(output/f'root_{i}_bank.json')==row['bank_sha256'] and file_sha(output/f'root_{i}_journal.jsonl')==row['journal_sha256'],'bank/journal byte binding')
        saved=read(output/f'root_{i}_bank.json');c=compiled[ready['root_sha256']]
        regenerated=old.sra.spectral_bank(saved['base']['jacobian'],saved['base']['nominal'],c['goal_vectors'][0],c['scoring']['metric']['projection'])
        require(saved['base']==regenerated and saved['expanded']==candidate_bank(regenerated,ready['root_sha256']),'fresh root saved Jacobian/goal/Q/all80 generation')
        validate_array_artifacts(output,p,report['prepared_sha256'],ready,c,saved,row)
        # Bounded streaming journal verification, no unbounded duplicate read.
        terminals=[];attempts=[];operations=[];candidate_events=[];array_events=[];array_progress=[]
        with (output/f'root_{i}_journal.jsonl').open() as stream:
            for line in stream:
                require(len(line)<4*1024**2,'bounded journal line');e=json.loads(line)
                if e.get('kind')=='entry_terminal':terminals.append(e['entry'])
                elif e.get('kind')=='entry_attempt':attempts.append(e['entry'])
                elif e.get('kind') in ('jacobian','suffix_jacobian','bounded_ridge_svd','mode_svd','suffix_solve'):operations.append(e)
                elif e.get('kind')=='array_saved':
                    array_events.append(e['receipt']);array_progress.append((len(terminals),len(operations)))
                elif 'candidate_id' in e:candidate_events.append(e)
                require(e.get('kind') not in ('arm_failure','blocked_entry'),'failed/blocked journal cannot export')
        require(terminals==row['actual_model_entries'] and len(attempts)==664 and operations==row['operations'],'durable actual entry/operation completeness')
        require(array_events==row['array_receipts'],'all immediate array receipts durable in journal')
        require(array_progress==expected_array_progress(),'array persistence must precede next AD/solve dispatch')
        require(all(a==dict(t,completed=False) for a,t in zip(attempts,terminals)),'durable attempts not refunded')
        require([r['arm'] for r in row['results']]==list(ARMS),'eight root-arm identities')
        require(candidate_events==[e for r in row['results'] for e in r['attempts']],'all-slot durable witness journal')
        require(row['completed_operations']==validate_operations(operations,row['results']) and row['completed_operations']['jacobian_calls']==41
            and row['completed_operations']['scalar_output_vjp_directions']==7872,'AD operation accounting')
        require(row['resource_receipt']==old.resource_receipt(terminals,row['results'],operations),'actual resource receipt integrity')
        for r,arm in zip(row['results'],ARMS):
            prov=provenance(p,s,ready,saved['expanded'],arm);best=validate_result(r,arm=arm,bank=saved['expanded'],ready=ready,prov=prov,compiled=c)
            plans.append(dict(root_index=i,root_sha256=ready['root_sha256'],arm=arm,horizon=2,raw_commands=best['actions'],
                root=deepcopy(ready['root']),action_domain=ready['action_domain'],provenance=prov,witness=best['witness'],
                predicted_Q=best['cost'],metric_sha256=ready['metric_sha256'],candidate_bank_sha256=saved['expanded']['candidate_bank_sha256'],
                report_sha256=file_sha(output/'report.json'),row_sha256=receipt['row_sha256'],resources=budget))
    require(report['actual_unique_predict_calls']==report['actual_unique_action_encoder_calls']==1328,'full paired actual census')
    require(report['completed_operations']=={k:sum(c[k] for c in counts) for k in ('jacobian_calls','scalar_output_vjp_directions','svd_calls','bounded_solve_calls')}
        and report['completed_selection_evaluations']==480,'actual total operations and selection census')
    return dict(schema=NAME+'.selected_actions',plans=plans,report_sha256=file_sha(output/'report.json'),
        source_bound=True,saved_data_validation=True,independent_neural_replay_performed_by_exporter=False,
        CPU_validation_wall_time_s=time.perf_counter()-started,CPU_validation_cost='8 root SVD +160 suffix validations (0..160 SVD) +4 root/0..160 suffix ridge solves, all-slot witnesses/Q/minima',
        completed_operations=counts,P1_admission=False,P2_admission=False,total_resource_matched=False)


if __name__=='__main__':main()
