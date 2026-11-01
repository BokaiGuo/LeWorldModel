"""Four original known task/method slots; parent-specific native CUDA program tree.

Only explicit MAIN run imports torch. Saved strict replays core algebra from
native call transcripts, not a model. No R generator, physics or admission.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p3_parent_native_adapter_20261029_v1 as parent
from experiments.spectral_reachability import agent_p3_native_program_20261028_v1 as native

signal=native.signal
core=parent.core
live=parent.progress
ROOT=core.ROOT
NAME='agent_p3_parent_native_cuda_20261029_v1'
require,canonical,digest,sha=signal.require,signal.canonical,signal.digest,signal.sha
FLAGS=dict(parent.FLAGS,P2_admission=False,complete_R=False)
PARENT_SHA='150a1c4189a68f9b7c71479e5aa6980cea490fdeed0d5524b7bfad0cc78c2558'
ADAPTER_FILES={ROOT/'experiments/spectral_reachability/agent_p3_parent_native_adapter_20261029_v1.py':PARENT_SHA,
    ROOT/'tests/test_agent_p3_parent_native_adapter_20261029_v1.py':'63950e433f7a8de271516d9345b51b7a034f913854a3f6b8c1bdf68d39cd01e0',
    ROOT/'innerdocs/agent_p3_parent_native_adapter_20261029_v1.md':'4ad02e80828b8e9635411d63a013f8660f3dffd48f330c851e66e4d074bd52dc'}
CALL_ARRAYS={'io_input','io_return','origin','packet','boundary','nominal','AD_terminal','jacobian',
    'raw_backward','program_packet','program_boundary'}
RESOURCE={'resource_attempt','resource_return','resource_failure','resource_blocked'}
CORE_EVENTS={parent.NAME+'.signal_map',parent.NAME+'.generated'}


def roster():return [dict(slot=2*r+i,root=r,mode=m) for r in (0,1) for i,m in enumerate(('fixed','adaptive'))]


def profile():
    return dict(wall_seconds=3600,output_bytes=1024**3,final_reserve_bytes=256*1024**2,
        disk_floor_bytes=200*1024**2,cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        resource_caps={m:parent.default_caps(m) for m in ('fixed','adaptive')},max_requests=144,
        CPU_neural=False,encoder_calls=0,physics_steps=0,no_retry=True,no_padding=True,
        stop_contract='R/cap stops that entire tree; continue next fixed slot; runtime errors stop later slots',
        depth3_max_raw45=True,model_parity_recomputed=False)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build():
    for f,h in ADAPTER_FILES.items():require(sha(f)==h,'sealed parent triple')
    # Original task/source authority, not a new cohort or outcome-selected report.
    old=signal.prepare_material()
    files=set(map(Path,old['sources']))|set(own_files())|set(ADAPTER_FILES)
    files.update(Path(m.__file__).resolve() for m in (parent,live,native,native.generator))
    files.add(ROOT/'experiments/spectral_reachability/pusht_full_packet_chart_20261027.py')
    return dict(schema=NAME+'.prepare',tasks=old['tasks'],policy=old['policy'],
        expected_model_hash=old['expected_model_hash'],sources={str(p.resolve()):sha(p) for p in sorted(files)},
        authority=dict(upstream=old['upstream'],upstream_sha256=old['upstream_sha256'],history=old['history'],
            original_input_scope=old['input_scope'],parent_source_sha256=PARENT_SHA),
        roster=roster(),profile=profile(),precision=precision_contract(),
        rules=dict(queue='original choose/request/checked_path; complete144 slots per arm',
            generator='frozen fullcommit.generate 4/8 bounded-ridge fractions',
            type='progress edges only; composition is their exact chain; R refuses all8',
            partial='no best-plan release after any unsupported/cap/failure',
            input='fixed original known2 RGB-derived root contexts/goal192/identityQ/raw normalizer; no new encoder'),**FLAGS)


def regular(path,cap):
    p=Path(path).absolute();require(not p.is_symlink() and p.is_file() and p.resolve()==p and p.stat().st_size<=cap,'bounded regular exact path')
    return p


def exclusive(path):
    p=Path(path).absolute();require(p.parent==ROOT/'results' and p.name.startswith(NAME) and
        p.resolve()==p and not p.exists() and not p.is_symlink(),'new own result path')
    return p


def write_new(path,value,cap=16*1024**2):
    p=exclusive(path);b=canonical(value)
    require(len(b)<=cap and shutil.disk_usage(p.parent).free>=profile()['disk_floor_bytes']+len(b),'metadata cap/floor')
    with p.open('xb') as f:require(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p))


def prepare(output):return write_new(output,build())


def verify(prepared,expected):
    core.hash_value(expected);p=regular(prepared,16*1024**2);require(sha(p)==expected,'external prepare SHA')
    value=signal.read(p);require(value==build() and sha(p)==expected,'full source/task/cap reconstruction')
    return value


def precision_contract():return dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False,autocast_cuda=False,autocast_cpu=False)


def checked_precision(torch):
    actual=dict(SDP='MATH' if torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled()
        and not torch.backends.cuda.mem_efficient_sdp_enabled() and not torch.backends.cuda.cudnn_sdp_enabled() else 'UNAPPROVED',
        matmul_TF32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_TF32=bool(torch.backends.cudnn.allow_tf32),
        autocast_cuda=bool(torch.is_autocast_enabled('cuda')),autocast_cpu=bool(torch.is_autocast_enabled('cpu')))
    require(actual==precision_contract(),'actual MATH/noTF32/noautocast guards');return actual


class ArmSink:
    def __init__(self,store,slot):self.store=store;self.slot=slot;self.call=None;self.next_call=0
    def event(self,payload):self.store.event(dict(kind='arm_event',slot=self.slot,call=self.call,payload=payload))


class NativeTree(native.NativeProgram):
    """Original model equations, but full root-to-best paths may contain9 packets."""
    def replay(self,parent_context,actions,segmented,ledger):
        core.context(parent_context);core.finite(actions,(len(actions),2),True)
        require(parent_context==self.task['root_context'] and len(actions)%5==0 and 5<=len(actions)<=45
            and np.max(np.abs(actions))<=1,'exact parent/full tree raw1..9 packets')
        import torch
        self.frozen();self.ledger=ledger;self.phase='segmented' if segmented else 'whole'
        state=torch.tensor(parent_context['latent_history'],dtype=torch.float32,device=self.device)[None]
        history=torch.tensor(parent_context['action_history'],dtype=torch.float32,device=self.device)[None]
        mean=torch.tensor(self.task['action_mean'],dtype=torch.float32,device=self.device)
        std=torch.tensor(self.task['action_std'],dtype=torch.float32,device=self.device);path=[]
        with torch.no_grad(),self.observe():
            for step,packet in enumerate(np.asarray(actions,np.float32).reshape(-1,5,2)):
                self.step=step
                if segmented and path:
                    state=torch.tensor(path[-1]['latent_history'],dtype=torch.float32,device=self.device)[None]
                    history=torch.tensor(path[-1]['action_history'],dtype=torch.float32,device=self.device)[None]
                raw=torch.tensor(packet,dtype=torch.float32,device=self.device)
                token=((raw-mean)/std).reshape(10);outgoing=torch.cat((history,token[None,None]),dim=1)
                self.arrays('program_packet',latent=state,history=history,raw=raw,mean=mean,std=std,token=token,outgoing=outgoing)
                encoded=self.model.action_encoder(outgoing)
                predicted=self.call('predict',lambda:self.model.predict(state,encoded),dict(latent=state,act_emb=encoded),shape=list(state.shape))
                require(tuple(predicted.shape)==(1,3,192),'native full3 prediction ABI')
                state=torch.cat((state[:,1:],predicted[:,-1:]),dim=1);history=outgoing[:,1:]
                self.arrays('program_boundary',latent=state,history=history)
                path.append(dict(parent_context,latent_history=state[0].cpu().tolist(),action_history=history[0].cpu().tolist()))
        self.frozen();core.checked_path(self.task,parent_context,actions,path);return path


class RecordingBackend:
    def __init__(self,backend,task,sink,check):
        self.backend=backend;self.task=deepcopy(task);self.sink=sink;self.check=check;self.kind=backend.kind
    def invoke(self,method,request,ledger):
        require(self.sink.call is None,'no recursive native invocation')
        self.check();cid=self.sink.next_call;self.sink.next_call+=1;self.sink.call=cid
        try:
            self.sink.event(dict(kind='native_call_attempt',method=method,task=deepcopy(self.task),request=deepcopy(request)))
            if method=='signal':out=self.backend.signal(request,ledger)
            else:out=self.backend.replay(request['parent_context'],request['actions'],request['segmented'],ledger)
            self.check();self.sink.event(dict(kind='native_call_return',result=deepcopy(out)));return out
        except Exception as exc:
            self.sink.event(dict(kind='native_call_failure',error=dict(type=type(exc).__name__,message=str(exc))));raise
        finally:self.sink.call=None
    def signal(self,request,ledger):return self.invoke('signal',request,ledger)
    def replay(self,context,actions,segmented,ledger):
        return self.invoke('replay',dict(parent_context=context,actions=actions,segmented=segmented),ledger)


def execute(p,factory,store,*,permit_ram=False,check=lambda:None):
    rows=[dict(**r,status='not_requested',result=None) for r in roster()];failed=False
    store.event(dict(kind='denominator',rows=deepcopy(rows)))
    for row in rows:
        if failed:row['status']='blocked_after_runtime_failure';continue
        sink=ArmSink(store,row['slot']);row['status']='attempted';store.event(dict(kind='arm_attempt',identity=roster()[row['slot']]))
        def make(t,s):return RecordingBackend(factory(t,s),t,s,check)
        try:
            check();result=parent.run_core(p['tasks'][row['root']],make,sink,mode=row['mode'],
                caps=p['profile']['resource_caps'][row['mode']],permit_ram=permit_ram)
            row.update(status=result['status'],result=result);check()
            failed=bool(result['errors']) or result['status']=='failed_or_budget_exhausted'
        except Exception as exc:
            row.update(status='outer_runtime_failure',error=dict(type=type(exc).__name__,message=str(exc)));failed=True
        store.event(dict(kind='arm_return',row=deepcopy(row)))
    return rows


def model_sources(model,p):
    classes=[]
    for name,module in model.named_modules():
        file=str(Path(inspect.getfile(type(module))).resolve())
        require(file in p['sources'] and sha(file)==p['sources'][file],'actual native module class source pin')
        classes.append(dict(name=name,class_name=type(module).__module__+'.'+type(module).__qualname__,path=file,sha256=p['sources'][file]))
    return classes


def run(prepared,expected,output,*,authorization):
    require(authorization=='MAIN_APPROVED_HOST_CUDA','explicit MAIN source audit + HOST CUDA scheduling')
    started=time.monotonic();p=verify(prepared,expected);output=exclusive(output)
    require(shutil.disk_usage(output.parent).free>=p['profile']['output_bytes']+p['profile']['disk_floor_bytes'],'whole output allowance plus floor')
    store=signal.Store(output,p['profile']);store.started=started
    rows=[dict(**r,status='blocked_before_model',result=None) for r in roster()]
    errors=[];model=None;before=after=host=classes=classes_after=None;guards=[];source_ok=False
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
        before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original frozen model state')
        classes=model_sources(model,p);store.event(dict(kind='model_before',hashes=before,classes=classes))
        # Observe only instance forwards; no old module globals are replaced.
        forwards=[(m,'forward' in m.__dict__,m.__dict__.get('forward')) for m in model.modules()]
        with torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False),sdpa_kernel(SDPBackend.MATH):
            def guard():store.check();checked_precision(torch)
            guards.append(checked_precision(torch));store.event(dict(kind='precision_before',value=guards[-1]))
            rows=execute(p,lambda t,s:NativeTree(model,device,t,s),store,check=guard)
            guards.append(checked_precision(torch));store.event(dict(kind='precision_after',value=guards[-1]))
        torch.cuda.synchronize(device)
        require(all(('forward' in m.__dict__)==had and m.__dict__.get('forward') is old for m,had,old in forwards),'all instance forward hooks restored')
    except BaseException as exc:errors.append(dict(stage='run',type=type(exc).__name__,message=str(exc)))
    finally:
        if model is not None:
            try:
                after=signal.model_hash(model);classes_after=model_sources(model,p)
                require(before==after and classes==classes_after and all(not m.training for m in model.modules())
                    and all(not v.requires_grad for v in model.parameters()),'unchanged eval/frozen parameters and all buffers/classes')
                store.event(dict(kind='model_after',hashes=after,classes=classes_after))
            except BaseException as exc:errors.append(dict(stage='model_final',message=str(exc)))
        try:
            verify(prepared,expected);store.check();source_ok=True;store.event(dict(kind='source_after',sha256=digest(p['sources'])))
        except BaseException as exc:errors.append(dict(stage='source_or_limits_final',message=str(exc)))
        store.close()
        settled=all(r['result'] is not None and not r['result']['errors'] for r in rows)
        report=dict(schema=NAME+'.report',status='four_slots_settled_not_admitted' if settled and not errors else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,model_before=before,model_after=after,
            classes_before=classes,classes_after=classes_after,precision=guards,host=host,sources_verified=source_ok,
            source_manifest_sha256=digest(p['sources']),wall_seconds=store.clock()-store.started,
            CUDA_peak_bytes=store.memory() if store.memory else None,journal_tail=store.tail,
            artifact_sha256={n:sha(output/n) for n in ('binding.json','events.jsonl') if (output/n).exists()},**FLAGS)
        store.put('report.json',report)
    # Settled R/cap refusals are legitimate saved engineering outcomes, not process failure.
    require(report['status']=='four_slots_settled_not_admitted','partial saved; no retry: '+repr(errors))
    return dict(output=str(output),report_sha256=sha(output/'report.json'))


def stripped(value):
    """Only measured timestamps vary when saved core algebra is replayed."""
    if type(value) is list:return [stripped(v) for v in value]
    if type(value) is dict:return {k:stripped(v) for k,v in value.items() if k not in
        ('started_ns','ended_ns','wall_ns','inclusive_operation_wall_ns')}
    return value


def event_census(events):
    top={'denominator','arm_attempt','arm_return','source_before','source_after','model_before','model_after','precision_before','precision_after'}
    for e in events:
        if e['kind']=='arm_event':
            core.keys(e,('kind','slot','call','payload'));core.integer(e['slot']);require(e['slot']<4,'fixed four slots')
            if e['call'] is not None:core.integer(e['call'])
            x=e['payload'];kind=x['kind']
            require(kind in RESOURCE|CALL_ARRAYS|CORE_EVENTS|{'direction','native_call_attempt','native_call_return','native_call_failure'},'unknown native/core event kind')
            if kind in RESOURCE:core.keys(x,('kind','entry'))
            elif kind in CALL_ARRAYS:
                core.keys(x,('kind','scope','phase','step','operation_index','arrays'))
                require(e['call'] is not None and x['arrays'],'native arrays require exact invocation')
                for b in x['arrays'].values():require(signal.unblob(b).dtype==np.float32,'native original float32')
                if kind not in ('io_input','io_return'):
                    core.keys(x['arrays'],{'origin':('latent','history','raw_flat','mean','std'),
                        'packet':('raw','token','outgoing'),'boundary':('latent','history'),
                        'nominal':('value',),'AD_terminal':('value',),'jacobian':('value',),'raw_backward':('value',),
                        'program_packet':('latent','history','raw','mean','std','token','outgoing'),
                        'program_boundary':('latent','history')}[kind])
            elif kind=='direction':core.keys(x,('kind','scope','direction'));core.integer(x['direction'])
            elif kind=='native_call_attempt':core.keys(x,('kind','method','task','request'))
            elif kind=='native_call_return':core.keys(x,('kind','result'))
            elif kind=='native_call_failure':core.keys(x,('kind','error'))
            elif kind==parent.NAME+'.signal_map':core.keys(x,('kind','mapping'))
            else:core.keys(x,('kind','root_task_sha256','parent_task_sha256','request','bank','native_response','result'))
        else:
            require(e['kind'] in top,'unknown top event')
            core.keys(e,{'denominator':('kind','rows'),'arm_attempt':('kind','identity'),'arm_return':('kind','row'),
                'source_before':('kind','sha256'),'source_after':('kind','sha256'),
                'model_before':('kind','hashes','classes'),'model_after':('kind','hashes','classes'),
                'precision_before':('kind','value'),'precision_after':('kind','value')}[e['kind']])


def resource_census(receipt,events):
    core.audit_ledger(receipt);actual=[e for e in events if e['kind'] in RESOURCE];expected=[];stack=[]
    for e in actual:
        op=e['entry'];require(op['index']<len(receipt['events']),'actual operation index')
        if e['kind']=='resource_attempt':stack.append(op['index'])
        elif e['kind']!='resource_blocked':require(stack and stack.pop()==op['index'],'proper nested actual dispatch order')
    require(not stack,'no unclosed native ledger attempt')
    for op in receipt['events']:
        if op['status']=='blocked_before_dispatch':expected.append(dict(kind='resource_blocked',entry=op))
        else:
            expected.append(dict(kind='resource_attempt',entry=dict(op,status='attempted',ended_ns=None,error=None)))
            expected.append(dict(kind='resource_return' if op['status']=='returned' else 'resource_failure',entry=op))
    require(sorted(map(canonical,actual))==sorted(map(canonical,expected)),'complete all-attempt/return/failure census')
    linked={}
    for e in events:
        if e.get('operation_index') is not None:linked.setdefault(e['operation_index'],[]).append(e)
    for op in receipt['events']:
        items=linked.pop(op['index'],[])
        names=['io_input','io_return'] if op['resource'] in ('predict','action_encoder','predictor','pred_proj','grad') else ['raw_backward'] if op['resource']=='raw_backward' else []
        if op['status']=='returned':require([x['kind'] for x in items]==names,'all and only paid native buffers')
        elif names:require([x['kind'] for x in items] in ([],names[:1]),'only actually available failed-call buffers')
        else:require(not items,'no invented scalar IO')
        for x in items:require(x['scope']==op['scope'],'native scope exact operation identity')
        if items and items[0]['kind']=='io_input':
            resource=op['resource'];keys={'predict':('latent','act_emb'),'action_encoder':('0',),'predictor':('0','1'),'pred_proj':('0',),'grad':('output','raw','basis')}[resource]
            core.keys(items[0]['arrays'],tuple(resource+':'+k for k in keys))
            if resource!='grad':require(items[0]['arrays'][resource+':'+keys[0]]['shape']==op['shape'],'actual paid shape/sample/token')
        for x in items:
            if x['kind']=='io_return':core.keys(x['arrays'],(op['resource']+':output',))
            elif x['kind']=='raw_backward':core.keys(x['arrays'],('value',))
    require(not linked,'no orphan native IO')


def audit_native(call):
    attempt=call[0];require(attempt['kind']=='native_call_attempt' and call[-1]['kind'] in ('native_call_return','native_call_failure'),'bounded native call disposition')
    task=core.validate_task(attempt['task']);method=attempt['method'];req=attempt['request'];body=call[1:-1]
    require(all(x['kind'] in RESOURCE|CALL_ARRAYS|{'direction'} for x in body),'no caller/core events smuggled into native call')
    # Failure-prefix evidence is retained/censused, but never labeled full native closure.
    if call[-1]['kind']=='native_call_failure':return False
    out=call[-1]['result']
    if method=='replay':
        core.keys(req,('parent_context','actions','segmented'))
        require(type(req['segmented']) is bool and req['parent_context']==task['root_context'],'actual parent replay')
        core.checked_path(task,req['parent_context'],req['actions'],out)
        live.audit_trajectory(task,body,'segmented' if req['segmented'] else 'whole',len(req['actions'])//5,req['actions'],out,program=True)
        require(all(x.get('phase')==('segmented' if req['segmented'] else 'whole') for x in body if x['kind'] in CALL_ARRAYS),'no foreign replay phase')
        allowed=RESOURCE|{'program_packet','program_boundary','io_input','io_return'}
        expected=live.predict_axes(len(req['actions'])//5)
    else:
        require(method=='signal','known native method');h=req['horizon']
        require(req==core.request(task,dict(context=task['root_context']),operation='compile_finite_horizon_J',horizon=h),'exact native signal request')
        core.keys(out,('binding_sha256','parent_context_sha256','horizon','nominal','jacobian','producer_sha256'))
        require(out['binding_sha256']==req['binding_sha256'] and out['parent_context_sha256']==req['parent_context_sha256'] and out['horizon']==h and out['producer_sha256']==sha(signal.__file__),'actual original native signal producer')
        n=core.finite(out['nominal'],(192,),True);j=core.finite(out['jacobian'],(192,10*h),True)
        for phase in ('nominal','AD'):
            terminal=live.audit_trajectory(task,body,phase,h,np.zeros((5*h,2),np.float32),None,program=False)
            require(np.array_equal(n,terminal),'nominal and AD exact terminal')
        def values(kind):return [signal.unblob(x['arrays']['value']) for x in body if x['kind']==kind]
        for kind,value in (('nominal',n),('AD_terminal',n),('jacobian',j)):
            v=values(kind);require(len(v)==1 and np.array_equal(v[0],value),'native response buffer exact '+kind)
        require([x['direction'] for x in body if x['kind']=='direction']==list(range(192)),'all192 VJP directions')
        gi=[x for x in body if x['kind']=='io_input' and 'grad:basis' in x['arrays']]
        go=[x for x in body if x['kind']=='io_return' and 'grad:output' in x['arrays']]
        rw=values('raw_backward');require(len(gi)==len(go)==len(rw)==192,'all grad/raw witnesses')
        for i,(a,b,c) in enumerate(zip(gi,go,rw,strict=True)):
            require(np.array_equal(signal.unblob(a['arrays']['grad:basis']),np.eye(192,dtype=np.float32)[i]) and
                np.array_equal(signal.unblob(a['arrays']['grad:output']),n) and
                np.array_equal(signal.unblob(a['arrays']['grad:raw']),np.zeros(10*h,np.float32)) and
                np.array_equal(signal.unblob(b['arrays']['grad:output']),j[i]) and np.array_equal(c,j[i]),'basis/raw/gradient/J exact row')
        require(all(x.get('phase') in ('nominal','AD') for x in body if x['kind'] in CALL_ARRAYS),'signal phase whitelist')
        allowed=RESOURCE|(CALL_ARRAYS-{'program_packet','program_boundary'})|{'direction'}
        expected=live.predict_axes(2*h);expected.update(jacobian_API=1,VJP_directions=192,grad_API=192,raw_backward_callbacks=192)
    require(all(x['kind'] in allowed for x in body),'all-and-only invocation event types')
    actual=dict.fromkeys(core.AXES,0)
    for x in body:
        if x['kind']=='resource_attempt':
            for k in core.AXES:actual[k]+=x['entry']['delta'][k]
    require(actual==expected,'exact actual native invocation resource axes')
    return True


def exception_from(error):
    allowed={'ValueError':ValueError,'RuntimeError':RuntimeError,'BudgetExceeded':core.BudgetExceeded,
        'AssertionError':AssertionError,'MemoryError':MemoryError,'TypeError':TypeError,'KeyError':KeyError}
    require(error['type'] in allowed,'unknown failed-native exception; preserved but not verified')
    # KeyError repr changes on reconstruction; refuse rather than rewrite messages.
    require(error['type']!='KeyError','KeyError needs explicit future saved support')
    return allowed[error['type']](error['message'])


def replay_resources(body,ledger):
    """Reissue metered operation tree only; saved outputs are inert, no model execution."""
    es=[x for x in body if x['kind'] in RESOURCE];index=0
    def operation():
        nonlocal index
        start=es[index];index+=1;op=start['entry']
        require(start['kind'] in ('resource_attempt','resource_blocked'),'native operation predispatch')
        def work():
            nonlocal index
            try:
                while index<len(es) and es[index]['kind'] in ('resource_attempt','resource_blocked'):operation()
            except Exception:
                require(index<len(es) and es[index]['kind']=='resource_failure' and es[index]['entry']['index']==op['index'],'parent failure unwind')
                index+=1;raise
            require(index<len(es) and es[index]['entry']['index']==op['index'],'matching native operation end')
            end=es[index];index+=1
            if end['kind']=='resource_failure':
                name,sep,message=end['entry']['error'].partition(': ');require(sep,'literal native failure')
                raise exception_from(dict(type=name,message=message))
            require(end['kind']=='resource_return','matching native return')
        ledger.call(op['resource'],work,shape=op['shape'],directions=op['directions'])
    while index<len(es):operation()


class SavedBackend:
    kind='HOST_CUDA'  # Inert saved replay only, never accepted as a runtime model certificate.
    def __init__(self,task,state,kind='HOST_CUDA'):self.task=task;self.state=state;self.kind=kind
    def invoke(self,method,req,ledger):
        pos=self.state['index'];calls=self.state['calls'];require(pos<len(calls),'no unrecorded native callback')
        call=calls[pos];self.state['index']+=1;a=call[0]
        require(a['method']==method and a['task']==self.task and a['request']==req,'exact parent-native invocation order/inputs')
        replay_resources(call[1:-1],ledger)
        if call[-1]['kind']=='native_call_failure':raise exception_from(call[-1]['error'])
        return deepcopy(call[-1]['result'])
    def signal(self,req,ledger):return self.invoke('signal',req,ledger)
    def replay(self,context,actions,segmented,ledger):return self.invoke('replay',dict(parent_context=context,actions=actions,segmented=segmented),ledger)


def validate_arm(task,row,wrapped,*,permit_fixture=False):
    r=row['result'];require(r is not None and r['mode']==row['mode'] and row['status']==r['status'],'settled original arm identity')
    events=[x['payload'] for x in wrapped];resource_census(r['resource_receipt'],events)
    calls=[];ids=[]
    for e in wrapped:
        cid=e['call']
        if cid is None:
            require(e['payload']['kind'] in RESOURCE|CORE_EVENTS,'unscoped native event refused');continue
        if not ids or ids[-1]!=cid:
            require(cid==len(ids),'contiguous explicit call ids');ids.append(cid);calls.append([])
        calls[-1].append(e['payload'])
    require(r['backend_kind']=='HOST_CUDA' or (permit_fixture is True and r['backend_kind']=='RAM_FIXTURE'),'no implicit RAM saved production')
    complete=[audit_native(c) for c in calls]
    state=dict(calls=calls,index=0);sink=type('Sink',(),{'event':lambda self,e:None})()
    replay=parent.run_core(task,lambda t,s:SavedBackend(t,state,r['backend_kind']),sink,mode=row['mode'],caps=r['resource_receipt']['caps'],permit_ram=permit_fixture)
    require(state['index']==len(calls),'all saved native calls consumed by original queue')
    require(stripped(replay)==stripped(r),'independent saved choose/queue/bank/raw/context/Q/replay/failure algebra')
    require([x['mapping'] for x in events if x['kind']==parent.NAME+'.signal_map']==r['signal_mappings'],'all dual identity mapping events')
    generated=[x for x in events if x['kind']==parent.NAME+'.generated']
    jobs=[j for j in r['jobs'] if j['payload'] is not None]
    require(len(generated)==len(jobs) and [x['result'] for x in generated]==[j['payload'] for j in jobs],'all generated native paths journaled')
    for e,j in zip(generated,jobs,strict=True):
        require(e['request']==j['request'] and e['root_task_sha256']==digest(task) and
            e['parent_task_sha256']==digest(parent.parent_task(task,j['request']['parent_context'])),'generated root/parent/source identity')
        matches=[x['native_response'] for x in r['signal_mappings'] if x['outer_request']['parent_context_sha256']==j['request']['parent_context_sha256']
            and x['native_response']['horizon']==j['request']['horizon']]
        require(matches and all(x==e['native_response'] for x in matches),'bank uses exact paid same-parent J')
        dummy=core.Ledger(parent.default_caps(row['mode']));dummy.scope=dict(saved_recompute=True)
        bank=live.generate(parent.parent_task(task,j['request']['parent_context']),e['native_response'],j['request']['candidates'],dummy)
        require(bank==e['bank'] and e['result']['actions']==bank['actions'][j['request']['candidate_index']],'saved exact original bounded-ridge bank')
    return dict(slot=row['slot'],root=row['root'],mode=row['mode'],status=r['status'],
        native_calls=len(calls),complete_native_calls=sum(complete),
        model_plan_available=r['model_plan'] is not None,realized_progress_depth=r['realized_progress_depth'],
        realized_widths=r['realized_widths'],resources=r['resource_receipt'],scalar_backward_calls=None,AD_primal_predict_API=None,Jacobian_elements=None)


def load_events(output,r):
    path=regular(output/'events.jsonl',profile()['output_bytes']);records=[];tail='0'*64
    with path.open('rb') as f:
        for i,line in enumerate(f):
            require(line.endswith(b'\n') and len(line)<=64*1024**2,'bounded complete event record')
            x=json.loads(line,object_pairs_hook=signal.unique_pairs);core.keys(x,('index','previous','payload','sha256'))
            h=x.pop('sha256');require(type(x['index']) is int and x['index']==i and x['previous']==tail and digest(x)==h,'journal chain')
            tail=h;records.append(x['payload'])
    require(tail==r['journal_tail'],'bound final journal tail');return records


def validate_saved(p,r,events,*,permit_fixture=False):
    event_census(events)
    require(len(r['rows'])==4 and [(x['slot'],x['root'],x['mode']) for x in r['rows']]==[(x['slot'],x['root'],x['mode']) for x in roster()],'all4 fixed denominator')
    initial=[dict(**x,status='not_requested',result=None) for x in roster()]
    require([x['rows'] for x in events if x['kind']=='denominator']==[initial],'denominator before native work')
    attempted=[x for x in r['rows'] if x['result'] is not None]
    require([x['identity'] for x in events if x['kind']=='arm_attempt']==[roster()[x['slot']] for x in attempted], 'fixed slot order/no retry')
    require([x['row'] for x in events if x['kind']=='arm_return']==attempted,'every attempted arm persisted')
    previous=next(i for i,e in enumerate(events) if e['kind']=='denominator')
    for row in attempted:
        starts=[i for i,e in enumerate(events) if e['kind']=='arm_attempt' and e['identity']['slot']==row['slot']]
        ends=[i for i,e in enumerate(events) if e['kind']=='arm_return' and e['row']['slot']==row['slot']]
        require(len(starts)==len(ends)==1 and previous<starts[0]<ends[0],'ordered nonoverlapping slot execution')
        require(all(starts[0]<i<ends[0] for i,e in enumerate(events) if e['kind']=='arm_event' and e['slot']==row['slot']),'all native work inside real arm dispatch')
        previous=ends[0]
    summaries=[];failed=False
    for row in r['rows']:
        wrapped=[e for e in events if e['kind']=='arm_event' and e['slot']==row['slot']]
        if row['result'] is None:
            require(failed and row['status']=='blocked_after_runtime_failure' and not wrapped,'later slots blocked only by runtime failure');continue
        require(not failed and row['result']['resource_receipt']['caps']==p['profile']['resource_caps'][row['mode']],'original slot caps/no postfailure continuation')
        summaries.append(validate_arm(p['tasks'][row['root']],row,wrapped,permit_fixture=permit_fixture))
        failed=bool(row['result']['errors']) or row['status']=='failed_or_budget_exhausted'
    return summaries


def consume(output,report_sha,prepared,prepare_sha):
    output=Path(output).absolute();require(output.resolve()==output and not output.is_symlink(),'exact saved output directory')
    core.hash_value(report_sha);rp=regular(output/'report.json',profile()['final_reserve_bytes'])
    require(sha(rp)==report_sha,'external actual report SHA');r=signal.read(rp,profile()['final_reserve_bytes']);p=verify(prepared,prepare_sha)
    core.keys(r,('schema','status','prepared','prepare_sha256','rows','errors','model_before','model_after','classes_before','classes_after',
        'precision','host','sources_verified','source_manifest_sha256','wall_seconds','CUDA_peak_bytes','journal_tail','artifact_sha256',*FLAGS))
    require(r['schema']==NAME+'.report' and r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha,'own report authority')
    require(all(r[k] is False for k in FLAGS) and not r['errors'] and r['sources_verified'] is True and r['source_manifest_sha256']==digest(p['sources']),'engineering only/final source closure')
    require(r['status'] in ('four_slots_settled_not_admitted','partial_or_failed'),'known report status')
    require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash'],'frozen state/buffers before-after')
    require(r['classes_before']==r['classes_after'] and r['classes_before'],'complete module classes before-after')
    for c in r['classes_before']:
        core.keys(c,('name','class_name','path','sha256'));require(p['sources'].get(c['path'])==c['sha256'],'all installed class files presealed')
    require(r['host']['device']=='cuda:0' and r['host']['allocation_sum']==4. and r['host']['visible_nvidia_nodes'],'actual host preflight')
    require(r['precision']==[precision_contract()]*2,'persisted before/after precision guards')
    require(set(r['artifact_sha256'])=={'binding.json','events.jsonl'} and {x.name for x in output.iterdir()}=={'binding.json','events.jsonl','report.json'},'exact saved artifacts')
    for n,h in r['artifact_sha256'].items():require(sha(regular(output/n,profile()['output_bytes']))==h,'actual saved artifact SHA')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'persisted prepare binding')
    events=load_events(output,r)
    for phase in ('before','after'):
        require([e for e in events if e['kind']=='model_'+phase]==[dict(kind='model_'+phase,hashes=r['model_'+phase],classes=r['classes_'+phase])],'model/class event closure')
        require([e for e in events if e['kind']=='source_'+phase]==[dict(kind='source_'+phase,sha256=digest(p['sources']))],'source event closure')
        require([e['value'] for e in events if e['kind']=='precision_'+phase]==[precision_contract()],'guard event closure')
    require(type(r['wall_seconds']) in (float,int) and 0<=r['wall_seconds']<=p['profile']['wall_seconds'],'whole run wall cap')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=p['profile'][k]
        for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'))),'actual CUDA peak axes')
    require(sum(x.stat().st_size for x in output.iterdir())<=p['profile']['output_bytes'],'whole output bytes')
    summaries=validate_saved(p,r,events)
    settled=all(x['result'] is not None and not x['result']['errors'] for x in r['rows'])
    require((r['status']=='four_slots_settled_not_admitted')==settled,'settled vs failed report exact semantics')
    require(sha(rp)==report_sha and all(sha(f)==h for f,h in p['sources'].items()) and
        all(sha(output/n)==h for n,h in r['artifact_sha256'].items()),'all source/report/artifact bytes unchanged after strict')
    return dict(schema=NAME+'.consume',status='saved_four_slot_engineering_verified' if settled else 'saved_partial_engineering_verified',
        prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha,report_sha256=report_sha,denominator=4,
        arms=summaries,slot_statuses=[x['status'] for x in r['rows']],historical_acquisition=p['authority']['history'],
        saved_verification='original queue/core/ridge recomputed; full native IO algebra; no neural parity recomputation',
        total_FLOPs=None,**FLAGS)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True)
    for cmd in ('verify','run','consume'):
        q=sub.add_parser(cmd);q.add_argument('--prepared',required=True);q.add_argument('--prepare-sha',required=True)
        if cmd=='run':q.add_argument('--output',required=True);q.add_argument('--authorization',required=True)
        if cmd=='consume':q.add_argument('--output',required=True);q.add_argument('--report-sha',required=True);q.add_argument('--receipt',required=True)
    a=ap.parse_args(argv)
    if a.cmd=='prepare':out=prepare(a.output)
    elif a.cmd=='verify':out=dict(schema=verify(a.prepared,a.prepare_sha)['schema'],verified=True)
    elif a.cmd=='run':out=run(a.prepared,a.prepare_sha,a.output,authorization=a.authorization)
    else:out=write_new(a.receipt,consume(a.output,a.report_sha,a.prepared,a.prepare_sha),128*1024**2)
    print(canonical(out).decode())


if __name__=='__main__':main()
