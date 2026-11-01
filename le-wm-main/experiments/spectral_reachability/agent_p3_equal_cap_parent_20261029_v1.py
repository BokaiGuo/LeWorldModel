"""MAIN-only equal-cap parent trees; own schema, original policy/native algebra.

No actual-resource matching/FLOPs claim. Import/prepare/RAM paths load no model.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as base

NAME='agent_p3_equal_cap_parent_20261029_v1'
ROOT=base.ROOT
parent,core,signal=base.parent,base.core,base.signal
require,sha,digest,canonical=base.require,base.sha,base.digest,base.canonical
PROTOCOL=ROOT/'innerdocs/agent_p3_equal_budget_protocol_20261029_v1.md'
PROTOCOL_SHA='36dc4e70eefd1a361ca4c7bc8a8181ea587c381640b85679921fa86bccb46861'
BASE_SHA='eeea5710d4171b7c43f99d1f2b5ad2412e03a25ba14e8fc5589ea5ef73f57b3b'
FLAGS=dict(base.FLAGS,equal_actual_resources=False,equal_total_compute=False,equal_wall=False)
TIMING='equal_cap_sync_probe'


def profile():
    b=parent.default_caps('adaptive')
    return dict(base.profile(),wall_seconds=18000,arm_wall_seconds=3600,
        resource_caps={m:deepcopy(b) for m in ('fixed','adaptive')},
        comparison='equal_cap_NOT_equal_actual_NOT_equalFLOPs',matching_axes=[],
        timer_scope='host_synchronized_arm_elapsed_including_policy_IO_excluding_load_and_source_checks',
        stop_contract='original core R/cap no plan; deadline/failure no release and later arms blocked')


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build():
    require(sha(base.__file__)==BASE_SHA and sha(PROTOCOL)==PROTOCOL_SHA,'frozen native helper/protocol')
    upstream=base.build()
    files=set(map(Path,upstream['sources']))|set(own_files())|{Path(base.__file__).resolve(),PROTOCOL}
    p=dict(schema=NAME+'.prepare',tasks=upstream['tasks'],policy=upstream['policy'],expected_model_hash=upstream['expected_model_hash'],
        sources={str(f.resolve()):sha(f) for f in sorted(files)},authority=deepcopy(upstream['authority']),
        roster=base.roster(),profile=profile(),precision=base.precision_contract(),rules=deepcopy(upstream['rules']),
        protocol_sha256=PROTOCOL_SHA,fixed_denominator=dict(arms=4,requests_per_arm=144,total_requests=576),**FLAGS)
    validate_prepare(p);return p


def validate_prepare(p):
    require(p['schema']==NAME+'.prepare' and p['profile']==profile() and p['roster']==base.roster(), 'own equal-cap ABI/fixed B W roster')
    require(p['fixed_denominator']==dict(arms=4,requests_per_arm=144,total_requests=576),'full576 denominator')
    require(len(p['tasks'])==2 and p['precision']==base.precision_contract() and p['protocol_sha256']==PROTOCOL_SHA,'known2/precision/protocol')
    for t in p['tasks']:core.validate_task(t)
    require(all(p[k] is False for k in FLAGS),'no admission/equal-actual claims')


def exclusive(path):
    p=Path(path).absolute()
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists() and not p.is_symlink(),'new own result path')
    return p


def write_new(path,obj,cap=128*1024**2):
    p=exclusive(path);body=canonical(obj)
    require(len(body)<=cap and shutil.disk_usage(p.parent).free>=profile()['disk_floor_bytes']+len(body),'metadata cap/floor')
    with p.open('xb') as f:require(f.write(body)==len(body),'whole write');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p))


def prepare(output):return write_new(output,build(),16*1024**2)


def verify(prepared,expected):
    core.hash_value(expected);path=base.regular(prepared,16*1024**2)
    require(sha(path)==expected,'actual own prepare SHA');p=signal.read(path)
    require(canonical(p)==canonical(build()) and sha(path)==expected,'complete own source/input/cap closure')
    return p


class Store(signal.Store):
    """Durable original-format hashchain; allow bounded failure unwind reserve.

    Wall/memory checks happen at guards, never after a committed return record.
    File/disk caps remain enforced even for recovery events.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.recovery=False
    def check(self,size=0,final=False):
        if self.recovery and not final:
            raise RuntimeError('equal-cap journal incomplete; recovery only')
        return super().check(size,final)
    def event(self,value):
        require(not self.closed,'open own journal')
        if self.recovery:
            kind=value.get('kind')
            terminal=kind in ('arm_return','source_after','model_after','precision_after')
            terminal=terminal or (kind==TIMING and value.get('status')=='failed')
            terminal=terminal or (kind=='arm_event' and value.get('payload',{}).get('kind') in
                ('resource_failure','resource_return','native_call_failure'))
            require(terminal,'recovery journal only failure unwind/final evidence; no dispatch')
        record=dict(index=self.index,previous=self.tail,payload=deepcopy(value));record['sha256']=digest(record)
        body=canonical(record)+b'\n'
        limit=self.limits['output_bytes']-(0 if self.recovery else self.limits['final_reserve_bytes'])
        if self.bytes+len(body)>limit:
            self.recovery=True;raise RuntimeError('equal-cap normal journal allowance exhausted')
        require(self.bytes+len(body)<=self.limits['output_bytes'] and self.free()>=self.limits['disk_floor_bytes']+len(body),'hard output/disk cap')
        require(self.stream.write(body)==len(body),'whole journal record');self.stream.flush();os.fsync(self.stream.fileno())
        self.bytes+=len(body);self.tail=record['sha256'];self.index+=1


class Timer:
    def __init__(self,store,slot,sync,guard,clock=time.monotonic_ns):
        self.store,self.slot,self.sync,self.guard,self.clock=store,slot,sync,guard,clock
        self.start=None;self.probes=[];self.failed=False
    def probe(self,stage,*,call=None,operation_index=None,raise_error=True):
        entered=self.clock();error=None;guard=None;status='returned';synced=False
        try:
            self.sync();synced=True;guard=self.guard()
        except Exception as exc:error=dict(type=type(exc).__name__,message=str(exc));status='failed'
        ended=self.clock()
        if self.start is None and synced:self.start=ended
        elapsed=None if self.start is None else ended-self.start
        overrun=None if elapsed is None else max(0,elapsed-int(profile()['arm_wall_seconds']*1e9))
        deadline=overrun is not None and overrun>0
        if deadline and error is None:error=dict(type='RuntimeError',message='equal-cap arm deadline');status='failed'
        self.failed=self.failed or status=='failed'
        probe=dict(kind=TIMING,slot=self.slot,index=len(self.probes),stage=stage,call=call,operation_index=operation_index,
            entered_ns=entered,ended_ns=ended,arm_start_ns=self.start,elapsed_ns=elapsed,overrun_ns=overrun,
            status=status,sync_returned=synced,guard=deepcopy(guard),error=error)
        self.probes.append(probe);self.store.event(probe)
        if error is not None and raise_error:
            # Original saved exception reconstruction supports RuntimeError.
            raise RuntimeError(error['message'])
        return probe
    def receipt(self):
        return dict(scope=profile()['timer_scope'],cap_seconds=3600,probes=deepcopy(self.probes),
            failed=self.failed,elapsed_ns=None if not self.probes else self.probes[-1]['elapsed_ns'],
            synchronization_attempts=len(self.probes),synchronization_returned=sum(p['sync_returned'] for p in self.probes))


class ArmSink(base.ArmSink):
    def __init__(self,store,slot,timer):super().__init__(store,slot);self.timer=timer
    def event(self,payload):
        super().event(payload)
        # Only fail at attempts, never after a persisted resource_return.
        # Native inner operations are checked by TimedLedger, outside call().
        if payload['kind']=='resource_attempt' and self.call is None:
            self.timer.probe('core_predispatch',operation_index=payload['entry']['index'])


class TimedLedger:
    def __init__(self,ledger,timer,sink):self.original,self.timer,self.sink=ledger,timer,sink
    def __getattr__(self,name):return getattr(self.original,name)
    def call(self,resource,fn,*,shape=None,directions=None):
        index=len(self.original.events)
        self.timer.probe('native_predispatch',call=self.sink.call,operation_index=index)
        out=self.original.call(resource,fn,shape=shape,directions=directions)
        self.timer.probe('native_postreturn',call=self.sink.call,operation_index=index)
        return out


class Backend:
    def __init__(self,task,sink,factory,kind):
        self.task,self.sink,self.factory,self.kind=deepcopy(task),sink,factory,kind;self.backend=None
    def invoke(self,method,req,ledger):
        require(self.sink.call is None,'nonrecursive native invocation')
        cid=self.sink.next_call;self.sink.next_call+=1;self.sink.call=cid
        try:
            self.sink.event(dict(kind='native_call_attempt',method=method,task=deepcopy(self.task),request=deepcopy(req)))
            self.sink.timer.probe('native_entry',call=cid)
            if self.backend is None:
                self.backend=self.factory(self.task,self.sink);require(self.backend.kind==self.kind,'actual backend kind')
            proxy=TimedLedger(ledger,self.sink.timer,self.sink)
            if method=='signal':out=self.backend.signal(req,proxy)
            else:out=self.backend.replay(req['parent_context'],req['actions'],req['segmented'],proxy)
            self.sink.timer.probe('native_exit',call=cid)
            self.sink.event(dict(kind='native_call_return',result=deepcopy(out)));return out
        except Exception as exc:
            self.sink.event(dict(kind='native_call_failure',error=dict(type=type(exc).__name__,message=str(exc))));raise
        finally:self.sink.call=None
    def signal(self,req,ledger):return self.invoke('signal',req,ledger)
    def replay(self,context,actions,segmented,ledger):
        return self.invoke('replay',dict(parent_context=context,actions=actions,segmented=segmented),ledger)


def initial_rows():
    return [dict(**x,status='not_requested',result=None,timing=None,plan_available=False,released_plan=None) for x in base.roster()]


def execute(p,factory,store,*,sync,guard,clock=time.monotonic_ns,permit_ram=False):
    validate_prepare(p);rows=initial_rows();failed=False;store.event(dict(kind='denominator',rows=deepcopy(rows)))
    for row in rows:
        if failed:row['status']='blocked_after_runtime_failure';continue
        store.event(dict(kind='arm_attempt',identity=base.roster()[row['slot']]))
        timer=Timer(store,row['slot'],sync,guard,clock);sink=ArmSink(store,row['slot'],timer)
        try:
            timer.probe('arm_start')
            result=parent.run_core(p['tasks'][row['root']],lambda t,s:Backend(t,s,factory,'RAM_FIXTURE' if permit_ram else 'HOST_CUDA'),sink,
                mode=row['mode'],caps=p['profile']['resource_caps'][row['mode']],max_requests=144,permit_ram=permit_ram)
            row.update(status=result['status'],result=result)
        except Exception as exc:
            row.update(status='outer_runtime_failure',error=dict(type=type(exc).__name__,message=str(exc)))
        finally:
            timer.probe('arm_final',raise_error=False);row['timing']=timer.receipt()
        result=row['result']
        failed=timer.failed or result is None or bool(result['errors']) or result['status']=='failed_or_budget_exhausted'
        row['plan_available']=not failed and result['model_plan'] is not None
        row['released_plan']=deepcopy(result['model_plan']) if row['plan_available'] else None
        store.event(dict(kind='arm_return',row=deepcopy(row)))
    return rows


def run(prepared,expected,output,*,authorization):
    require(authorization=='MAIN_APPROVED_HOST_CUDA','explicit MAIN HOST CUDA authorization')
    started=time.monotonic();p=verify(prepared,expected);output=exclusive(output)
    require(shutil.disk_usage(output.parent).free>=p['profile']['output_bytes']+p['profile']['disk_floor_bytes'],'whole output allowance/floor')
    store=Store(output,p['profile']);store.started=started
    rows=initial_rows();errors=[];model=None;before=after=classes=classes_after=host=None;precision=[];source_ok=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.event(dict(kind='source_before',sha256=digest(p['sources'])))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        store.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        store.check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original frozen weights')
        classes=base.model_sources(model,p);store.event(dict(kind='model_before',hashes=before,classes=classes))
        forwards=[(m,'forward' in m.__dict__,m.__dict__.get('forward')) for m in model.modules()]
        with torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False),sdpa_kernel(SDPBackend.MATH):
            def guard():
                store.check()
                return dict(precision=base.checked_precision(torch),peak_bytes=list(store.memory()),output_bytes=store.bytes,
                    whole_job_elapsed_seconds=store.clock()-store.started)
            precision.append(base.checked_precision(torch));store.event(dict(kind='precision_before',value=precision[-1]))
            rows=execute(p,lambda t,s:base.NativeTree(model,device,t,s),store,sync=lambda:torch.cuda.synchronize(device),guard=guard)
            precision.append(base.checked_precision(torch));store.event(dict(kind='precision_after',value=precision[-1]))
        require(all(('forward' in m.__dict__)==had and m.__dict__.get('forward') is fn for m,had,fn in forwards),'native hooks restored')
    except BaseException as exc:errors.append(dict(stage='run',type=type(exc).__name__,message=str(exc)))
    finally:
        store.recovery=True
        if model is not None:
            try:
                after=signal.model_hash(model);classes_after=base.model_sources(model,p)
                require(before==after and classes==classes_after and all(not m.training for m in model.modules()) and
                    all(not v.requires_grad for v in model.parameters()),'frozen weights/buffers/classes final')
                store.event(dict(kind='model_after',hashes=after,classes=classes_after))
            except BaseException as exc:errors.append(dict(stage='model_final',type=type(exc).__name__,message=str(exc)))
        try:
            verify(prepared,expected);source_ok=True;store.event(dict(kind='source_after',sha256=digest(p['sources'])))
        except BaseException as exc:errors.append(dict(stage='source_final',type=type(exc).__name__,message=str(exc)))
        settled=all(x['result'] is not None and not x['result']['errors'] and not x['timing']['failed'] for x in rows)
        store.close()
        report=dict(schema=NAME+'.report',status='four_slots_settled_not_admitted' if settled and not errors else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,model_before=before,model_after=after,
            classes_before=classes,classes_after=classes_after,precision=precision,host=host,sources_verified=source_ok,
            source_manifest_sha256=digest(p['sources']),profile=p['profile'],fixed_denominator=p['fixed_denominator'],
            wall_seconds=store.clock()-started,CUDA_peak_bytes=store.memory() if store.memory else None,journal_tail=store.tail,
            artifact_sha256={n:sha(output/n) for n in ('binding.json','events.jsonl') if (output/n).exists()},**FLAGS)
        store.put('report.json',report)
    require(report['status']=='four_slots_settled_not_admitted','partial saved; no retry '+repr(errors))
    return dict(output=str(output),report_sha256=sha(output/'report.json'))


def audit_timer(row,probes,*,permit_fixture=False):
    t=row['timing'];require(t is not None and t['probes']==probes and probes,'all actual timing probes persisted')
    require([v['index'] for v in probes]==list(range(len(probes))) and probes[0]['stage']=='arm_start' and probes[-1]['stage']=='arm_final','timing bookends/order')
    last=-1;start=None;failed=False
    for v in probes:
        require(v['slot']==row['slot'] and type(v['entered_ns']) is type(v['ended_ns']) is int and last<=v['entered_ns']<=v['ended_ns'],'measured monotone synchronization')
        last=v['ended_ns'];require(v['stage'] in ('arm_start','arm_final','core_predispatch','native_entry','native_exit','native_predispatch','native_postreturn'),'known guard stage')
        if start is None and v['sync_returned']:start=v['ended_ns']
        require(v['arm_start_ns']==start,'actual timer origin after start sync')
        elapsed=None if start is None else v['ended_ns']-start
        overrun=None if elapsed is None else max(0,elapsed-3600*10**9)
        require(v['elapsed_ns']==elapsed and v['overrun_ns']==overrun,'independent elapsed/overrun')
        require(v['status'] in ('returned','failed') and (v['error'] is None)==(v['status']=='returned'),'sync/guard outcome')
        if overrun:require(v['status']=='failed','deadline must fail, never successful overrun')
        if not permit_fixture and v['status']=='returned':
            g=v['guard'];require(v['sync_returned'] is True and g['precision']==base.precision_contract(),'real guard precision')
            require(len(g['peak_bytes'])==2 and all(type(n) is int and 0<=n<=profile()[k] for n,k in zip(g['peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'))),'per-probe memory caps')
            require(0<=g['output_bytes']<=profile()['output_bytes'] and 0<=g['whole_job_elapsed_seconds']<=18000,'per-probe job guards')
        failed=failed or v['status']=='failed'
    require(t==dict(scope=profile()['timer_scope'],cap_seconds=3600,probes=probes,failed=failed,
        elapsed_ns=probes[-1]['elapsed_ns'],synchronization_attempts=len(probes),synchronization_returned=sum(p['sync_returned'] for p in probes)), 'exact timer summary')


def audit_guard_placement(events,slot):
    es=[e for e in events if e['kind'] in (TIMING,'arm_event') and e['slot']==slot]
    failure_seen=False
    for i,e in enumerate(es):
        before=es[i-1] if i else None;after=es[i+1] if i+1<len(es) else None
        if e['kind']=='arm_event':
            x=e['payload'];kind=x['kind']
            if kind in ('resource_attempt','resource_blocked'):
                require(not failure_seen,'no new resource dispatch after first guard failure')
                if e['call'] is None:
                    if kind=='resource_attempt':
                        require(after is not None and after['kind']==TIMING and after['stage']=='core_predispatch' and
                            after['operation_index']==x['entry']['index'], 'all core attempts immediately guarded')
                else:
                    require(before is not None and before['kind']==TIMING and before['stage']=='native_predispatch' and
                        before['call']==e['call'] and before['operation_index']==x['entry']['index'] and before['status']=='returned','all native dispatches preceded by actual guard')
            if kind=='resource_return' and e['call'] is not None:
                require(after is not None and after['kind']==TIMING and after['stage']=='native_postreturn' and
                    after['call']==e['call'] and after['operation_index']==x['entry']['index'],'all returned native work post-synchronized')
            if kind=='native_call_attempt':
                require(not failure_seen and after is not None and after['kind']==TIMING and after['stage']=='native_entry' and after['call']==e['call'],'native entry guard')
            if kind=='native_call_return':
                require(before is not None and before['kind']==TIMING and before['stage']=='native_exit' and before['status']=='returned' and before['call']==e['call'],'native exit guard')
            continue
        stage=e['stage']
        if stage=='core_predispatch':
            require(before is not None and before['kind']=='arm_event' and before['call'] is None and
                before['payload']['kind']=='resource_attempt' and before['payload']['entry']['index']==e['operation_index'],'no invented core probe')
        elif stage=='native_predispatch':
            if e['status']=='returned':
                require(after is not None and after['kind']=='arm_event' and after['call']==e['call'] and after['payload']['kind'] in ('resource_attempt','resource_blocked') and
                    after['payload']['entry']['index']==e['operation_index'],'no invented native predispatch probe')
        elif stage=='native_postreturn':
            require(before is not None and before['kind']=='arm_event' and before['payload']['kind']=='resource_return' and
                before['call']==e['call'] and before['payload']['entry']['index']==e['operation_index'],'no invented postreturn probe')
        elif stage=='native_entry':
            require(before is not None and before['kind']=='arm_event' and before['payload']['kind']=='native_call_attempt' and before['call']==e['call'],'no invented native entry')
        elif stage=='native_exit':
            require(after is not None and after['kind']=='arm_event' and after['payload']['kind'] in ('native_call_return','native_call_failure') and after['call']==e['call'],'no invented native exit')
        elif stage=='arm_start':require(i==0,'only first arm start probe')
        elif stage=='arm_final':require(i==len(es)-1,'only last arm final probe')
        failure_seen=failure_seen or e['status']=='failed'


def validate_arm(task,row,wrapped,probes,*,permit_fixture=False):
    r=row['result'];require(r is not None and r['mode']==row['mode'] and r['status']==row['status'],'original core result identity')
    events=[e['payload'] for e in wrapped];base.resource_census(r['resource_receipt'],events)
    calls=[];ids=[]
    for e in wrapped:
        cid=e['call']
        if cid is None:require(e['payload']['kind'] in base.RESOURCE|base.CORE_EVENTS,'original unscoped core events');continue
        if not ids or ids[-1]!=cid:
            require(cid==len(ids),'contiguous original invocation ids');ids.append(cid);calls.append([])
        calls[-1].append(e['payload'])
    failed={v['operation_index']:v for v in probes if v['stage']=='core_predispatch' and v['status']=='failed'}
    no_backend=r['backend_kind'] is None and not calls and bool(failed)
    require(no_backend or r['backend_kind']=='HOST_CUDA' or (permit_fixture and r['backend_kind']=='RAM_FIXTURE'),'no implicit RAM production')
    native_complete=[base.audit_native(c) for c in calls]
    # Native failures are replayed by original SavedBackend. Only unscoped
    # core-attempt guard failures need an explicit sink before core executes fn.
    used=[]
    class ReplaySink:
        def event(self,e):
            if e['kind']=='resource_attempt' and e['entry']['index'] in failed:
                i=e['entry']['index'];require(i not in used,'single deadline injection');used.append(i)
                raise RuntimeError(failed[i]['error']['message'])
    state=dict(calls=calls,index=0)
    replay=parent.run_core(task,lambda t,s:base.SavedBackend(t,state,r['backend_kind']),ReplaySink(),mode=row['mode'],
        caps=r['resource_receipt']['caps'],max_requests=144,permit_ram=permit_fixture)
    require(set(used)==set(failed) and state['index']==len(calls),'all real calls and non-native guard failures consumed')
    require(base.stripped(replay)==base.stripped(r),'original complete tree/choose/queue/native inputs/resource/failure replay')
    require([e['mapping'] for e in events if e['kind']==parent.NAME+'.signal_map']==r['signal_mappings'],'all original signal mappings')
    generated=[e for e in events if e['kind']==parent.NAME+'.generated'];jobs=[j for j in r['jobs'] if j['payload'] is not None]
    require(len(generated)==len(jobs) and [e['result'] for e in generated]==[j['payload'] for j in jobs],'all generated program events')
    # The same frozen helper independently validates exact banks and dual
    # identities when no injected unscoped guard failure changes core execution.
    if not failed:base.validate_arm(task,row,wrapped,permit_fixture=permit_fixture)
    return dict(slot=row['slot'],root=row['root'],mode=row['mode'],status=row['status'],
        plan_available=row['plan_available'],resources=r['resource_receipt'],native_calls=len(calls),
        complete_native_calls=sum(native_complete),backend_kind=r['backend_kind'],timing=row['timing'],scalar_backward_calls=None,AD_primal_predict_API=None,Jacobian_elements=None)


def validate_saved(p,r,events,*,permit_fixture=False):
    validate_prepare(p);require(r['profile']==p['profile'] and r['fixed_denominator']==p['fixed_denominator'],'fixed actual common contract')
    filtered=[e for e in events if e['kind']!=TIMING]
    base.event_census(filtered)  # Original payload schema only, not an old report.
    require(len(r['rows'])==4 and [(x['slot'],x['root'],x['mode']) for x in r['rows']]==[(x['slot'],x['root'],x['mode']) for x in base.roster()],'four fixed slots')
    require([e['rows'] for e in events if e['kind']=='denominator']==[initial_rows()],'full initial denominator')
    attempted=[x for x in r['rows'] if x['timing'] is not None]
    require([e['identity'] for e in events if e['kind']=='arm_attempt']==[base.roster()[x['slot']] for x in attempted],'ordered single attempts')
    require([e['row'] for e in events if e['kind']=='arm_return']==attempted,'all whole attempted rows sealed')
    summaries=[];failed=False;previous=next(i for i,e in enumerate(events) if e['kind']=='denominator')
    for row in r['rows']:
        wrapped=[e for e in events if e['kind']=='arm_event' and e['slot']==row['slot']]
        probes=[e for e in events if e['kind']==TIMING and e['slot']==row['slot']]
        if row['timing'] is None:
            require(failed and row['status']=='blocked_after_runtime_failure' and not wrapped and not probes and row['result'] is None and row['released_plan'] is None,'blocked is unknown/no work/no plan')
            continue
        require(not failed,'no new arm after failure')
        starts=[i for i,e in enumerate(events) if e['kind']=='arm_attempt' and e['identity']['slot']==row['slot']]
        ends=[i for i,e in enumerate(events) if e['kind']=='arm_return' and e['row']['slot']==row['slot']]
        require(len(starts)==len(ends)==1 and previous<starts[0]<ends[0] and
            all(starts[0]<i<ends[0] for i,e in enumerate(events) if (e['kind'] in ('arm_event',TIMING) and e['slot']==row['slot'])),'all timing/native inside actual arm')
        previous=ends[0];audit_timer(row,probes,permit_fixture=permit_fixture);audit_guard_placement(events,row['slot'])
        result=row['result']
        if result is None:
            require(row['status']=='outer_runtime_failure' and row['timing']['failed'] and not wrapped,'only pre-core start failure without core result')
        else:
            require(len(result['jobs'])==144 and [j['slot'] for j in result['jobs']]==list(range(144)),'all144 original slots including refused/missing')
            require(result['resource_receipt']['caps']==p['profile']['resource_caps'][row['mode']],'actual common vector cap')
            summaries.append(validate_arm(p['tasks'][row['root']],row,wrapped,probes,permit_fixture=permit_fixture))
        failed=row['timing']['failed'] or result is None or bool(result['errors']) or result['status']=='failed_or_budget_exhausted'
        available=not failed and result['model_plan'] is not None
        require(row['plan_available'] is available and row['released_plan']==(result['model_plan'] if available else None),'only complete validated under-cap plans released')
    require(all(e['slot'] in range(4) for e in events if e['kind']==TIMING),'no foreign timing slot')
    return summaries


def consume(output,report_sha,prepared,prepare_sha):
    output=Path(output).absolute();require(output.resolve()==output and not output.is_symlink(),'exact own output directory')
    p=verify(prepared,prepare_sha);rp=base.regular(output/'report.json',p['profile']['final_reserve_bytes'])
    require(sha(rp)==report_sha,'external own report SHA');r=signal.read(rp,p['profile']['final_reserve_bytes'])
    core.keys(r,('schema','status','prepared','prepare_sha256','rows','errors','model_before','model_after','classes_before','classes_after',
        'precision','host','sources_verified','source_manifest_sha256','profile','fixed_denominator','wall_seconds','CUDA_peak_bytes','journal_tail','artifact_sha256',*FLAGS))
    require(r['schema']==NAME+'.report' and r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha,'own report/prepare authority')
    require(all(r[k] is False for k in FLAGS) and not r['errors'] and r['sources_verified'] is True and r['source_manifest_sha256']==digest(p['sources']),'no science claims/source clean')
    require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash'],'frozen parameters/buffers')
    require(r['classes_before']==r['classes_after'] and r['classes_before'],'same complete model classes')
    for c in r['classes_before']:require(p['sources'].get(c['path'])==c['sha256'],'presealed installed module source')
    require(r['host']['device']=='cuda:0' and r['host']['allocation_sum']==4. and r['host']['visible_nvidia_nodes'],'actual host CUDA preflight')
    require(r['precision']==[base.precision_contract()]*2,'actual before/after precision')
    require(set(r['artifact_sha256'])=={'binding.json','events.jsonl'} and {f.name for f in output.iterdir()}=={'binding.json','events.jsonl','report.json'},'all-and-only artifacts')
    for n,h in r['artifact_sha256'].items():require(sha(base.regular(output/n,p['profile']['output_bytes']))==h,'saved artifact bytes')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'actual durable execution binding')
    events=base.load_events(output,r)
    for phase in ('before','after'):
        require([e for e in events if e['kind']=='source_'+phase]==[dict(kind='source_'+phase,sha256=digest(p['sources']))],'source event closure')
        require([e for e in events if e['kind']=='model_'+phase]==[dict(kind='model_'+phase,hashes=r['model_'+phase],classes=r['classes_'+phase])],'model event closure')
        require([e['value'] for e in events if e['kind']=='precision_'+phase]==[base.precision_contract()],'guard event closure')
    require(0<=r['wall_seconds']<=18000 and len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=p['profile'][k] for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'))),'whole job/memory cap')
    require(sum(f.stat().st_size for f in output.iterdir())<=p['profile']['output_bytes'],'whole saved output cap')
    summaries=validate_saved(p,r,events)
    settled=all(x['result'] is not None and not x['result']['errors'] and not x['timing']['failed'] for x in r['rows'])
    require(r['status']==('four_slots_settled_not_admitted' if settled else 'partial_or_failed'),'settled status exact semantics')
    require(sha(rp)==report_sha and all(sha(f)==h for f,h in p['sources'].items()) and all(sha(output/n)==h for n,h in r['artifact_sha256'].items()),'final all inputs/artifacts unchanged')
    slots=[]
    for row in r['rows']:
        result=row['result']
        slots.append(dict(slot=row['slot'],root=row['root'],mode=row['mode'],status=row['status'],
            observed_request_rows=0 if result is None else len(result['jobs']),
            unobserved_request_slots=144 if result is None else 0,
            resources=None if result is None else result['resource_receipt'],plan_available=row['plan_available']))
    differences=[]
    for root in (0,1):
        a,b=(r['rows'][2*root+i]['result'] for i in (0,1))
        differences.append(dict(root=root,direction='adaptive_minus_fixed',
            actual_resource_differences=None if a is None or b is None else {phase:{k:b['resource_receipt'][phase][k]-a['resource_receipt'][phase][k]
                for k in core.AXES} for phase in ('attempted','completed')}))
    return dict(schema=NAME+'.consume',status='saved_equal_cap_verified_not_admitted' if settled else 'saved_equal_cap_partial_verified_not_admitted',
        prepared=r['prepared'],prepare_sha256=prepare_sha,report_sha256=report_sha,fixed_denominator=p['fixed_denominator'],
        arms=summaries,slots=slots,resource_pairs=differences,slot_statuses=[x['status'] for x in r['rows']],common_budget=p['profile'],equal_cap_contract=True,
        historical_acquisition=p['authority']['history'],total_FLOPs=None,matching_axes=[],**FLAGS)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True)
    for cmd in ('verify','run','consume'):
        q=sub.add_parser(cmd);q.add_argument('--prepared',required=True);q.add_argument('--prepare-sha',required=True)
        if cmd in ('run','consume'):q.add_argument('--output',required=True)
        if cmd=='run':q.add_argument('--authorization',required=True)
        if cmd=='consume':q.add_argument('--report-sha',required=True);q.add_argument('--receipt',required=True)
    a=ap.parse_args(argv)
    if a.command=='prepare':out=prepare(a.output)
    elif a.command=='verify':out=dict(verified=True,schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.command=='run':out=run(a.prepared,a.prepare_sha,a.output,authorization=a.authorization)
    else:out=write_new(a.receipt,consume(a.output,a.report_sha,a.prepared,a.prepare_sha))
    print(canonical(out).decode())


if __name__=='__main__':main()
