"""MAIN-only wall-capped native CUDA runtime and saved-only strict replay."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np
from experiments.spectral_reachability import agent_p2_compute_matched_protocol_20261029_v2 as protocol

NAME='agent_p2_wallcapped_cuda_20261029_v1'
AUTH='MAIN_P2_WALLCAPPED_KNOWN2_HOST_CUDA_ONLY'
old=protocol.old
ROOT=old.ROOT
require=old.require
CAP,FINAL,FLOOR,WALL=512*1024**2,32*1024**2,512*1024**2,2400


class BudgetStop(Exception):pass


def packet_request(index,context,actions,tag):
    require(type(index) is int and 0<=index<1902,'own full1902 forward bound')
    a=np.asarray(actions)
    require(a.dtype==np.float32 and a.shape==(5,2) and np.isfinite(a).all() and (abs(a)<1).all(),'literal native raw5')
    old.kernel.validate_context(context)
    return dict(index=index,tag=tag,input_context_sha256=old.digest(context),raw_sha256=old.digest(old.blob(a)),completed=False)


def validate_packet(f):
    c=f['input'];a=old.unblob(f['raw'])
    old.need_equal(f['request'],packet_request(f['index'],c,a,f['tag']),'own request/full raw/context')
    m=f['modules'];require(len(m)==3,'full3 modules')
    ins=[[old.unblob(b) for b in e['inputs']] for e in m];outs=[old.unblob(e['output']) for e in m]
    tokens=np.concatenate((np.asarray(c['action_history'],np.float32),old.kernel.normalized_history(a)))
    require([e['resource'] for e in m]==list(protocol.MODULES) and
        all(e['index']==3*f['index']+k and e['completed'] is True for k,e in enumerate(m)),'module sequence and ordinals')
    require(len(ins[0])==1 and ins[0][0].shape==(1,3,10) and np.array_equal(ins[0][0][0],tokens)
        and outs[0].shape==(1,3,192) and len(ins[1])==2
        and ins[1][0].shape==(1,3,192) and np.array_equal(ins[1][0][0],c['latent_history'])
        and np.array_equal(ins[1][1],outs[0]) and outs[1].shape==(1,3,192)
        and len(ins[2])==1 and ins[2][0].shape==(3,192)
        and np.array_equal(ins[2][0],outs[1].reshape(3,192)) and outs[2].shape==(3,192),'source-faithful full3 native IO')
    out=dict(c,latent_history=np.concatenate((np.asarray(c['latent_history'],np.float32)[1:],outs[2][-1:])).tolist(),action_history=tokens[1:].tolist())
    old.need_equal(f['output'],out,'exact full shifted context');require(f['completed'] is True,'full native packet return')
    return out


def ridge_from_svd(left,s,right,residual):
    target=left.T@residual
    def solution(multiplier):return right.T@(s*target/(s*s+old.compiler.DAMPING+multiplier))
    multiplier=0.;step=solution(multiplier)
    if np.linalg.norm(step)>old.CAP_LOGIT:
        low,high=0.,1.
        while np.linalg.norm(solution(high))>old.CAP_LOGIT:high*=2
        for _ in range(70):
            mid=(low+high)/2
            if np.linalg.norm(solution(mid))>old.CAP_LOGIT:low=mid
            else:high=mid
        multiplier,step=high,solution(high)
    # Match frozen bounded_ridge output, including residual evaluation.
    return multiplier,step


def make_bank(j,n,g,backend):
    j=old.finite(j,(192,30));n=old.finite(n,(192,));g=old.finite(g,(192,))
    _,s,right=backend.cpu('root_SVD',np.linalg.svd,j,full_matrices=False)
    for v in right:
        if v[np.argmax(abs(v))]<0:v*=-1
    left,s2,right2=backend.cpu('root_SVD',np.linalg.svd,j,full_matrices=False)
    multiplier,step=backend.cpu('root_ridge',ridge_from_svd,left,s2,right2,g-n)
    residual=g-n-j@step;d=old.compiler.DAMPING
    solve=dict(params=step.tolist(),singular_values=s2.tolist(),linear_terminal_cost=float(residual@residual/192),
        ridge_penalty=float(d*(step@step)/192),own_reach_value=float((residual@residual+d*(step@step))/192),
        radius=old.CAP_LOGIT,norm=float(np.linalg.norm(step)),ball_multiplier=multiplier)
    seeds=[np.zeros(30),step];kinds=['zero','root_dls_progress']
    for r in old.contract()['radii']:
        for mode in range(10):
            for sign in (-1,1):seeds.append(sign*r*right[mode]);kinds.append([r,mode,sign])
    return dict(params=[old.inward(v,old.CAP_LOGIT).tolist() for v in seeds],kinds=kinds,
        unrounded_seeds=[v.tolist() for v in seeds],singular_values=s.tolist(),right_vectors=right.tolist(),solve=solve)


def drive(slot,arm,backend,control):
    """Same fixed algorithm for actual and saved replay; never accepts outcomes."""
    c=slot['packet']['root_context'];g=slot['packet']['goal'];states={};nominal=None;bank=None;export=None;aborted=None
    def work(label):
        nonlocal nominal,bank,export
        if label=='nominal':
            nominal=backend.rollout(c,old.chart(np.zeros(30)),'nominal');return dict(logits=[0.]*30,contexts=nominal)
        if label=='bank':
            j=backend.jacobian(c,np.zeros(30),'bank',True)
            old.need_equal(backend.last_ad_contexts,nominal,'root J nominal full context')
            bank=make_bank(j,nominal[-1]['latent_history'][-1],g,backend);return None
        if label.startswith('seed:'):
            i=int(label.split(':')[1]);u=bank['params'][i]
            states[i]=dict(logits=u,contexts=backend.rollout(c,old.chart(u),label));return deepcopy(states[i])
        if label.startswith('joint:'):
            i=int(label.split(':')[2]);r=protocol.prior.local_joint(c,states[i]['logits'],g,backend,label)
            states[i]=dict(logits=r['logits'],contexts=r['contexts']);return deepcopy(states[i])
        if label.startswith('H2:'):
            i=int(label.split(':')[1]);u=np.asarray(states[i]['logits'],np.float64)
            first=backend.rollout(c,old.chart(u[:10]),label+'/prefix')[0]
            r=old.optimize(first,u[:10],u[10:],g,backend,label)
            states[i]=dict(logits=np.concatenate((u[:10],r['selected_logits'])).tolist(),contexts=[first]+r['selected_contexts'])
            return deepcopy(states[i])
        if label.startswith('H1:'):
            i=int(label.split(':')[1]);u=np.asarray(states[i]['logits'],np.float64);first,mid=states[i]['contexts'][:2]
            r=old.optimize(mid,u[:20],u[20:],g,backend,label)
            states[i]=dict(logits=np.concatenate((u[:20],r['selected_logits'])).tolist(),contexts=[first,mid]+r['selected_contexts'])
            return deepcopy(states[i])
        require(label=='replay','fixed schedule only')
        w=control.current()['incumbent'];require(w is not None,'no export without incumbent');w=w['witness']
        whole=backend.rollout(c,old.chart(w['logits']),'export/whole');seg=[];current=deepcopy(c)
        for k,a in enumerate(old.chart(w['logits'])):
            current=backend.rollout(current,a[None],'export/segment'+str(k))[0];seg.append(current)
        old.need_equal(whole,w['contexts'],'export all3 winner contexts');old.need_equal(seg,whole,'export segmented context identity')
        export=deepcopy(w);return None
    for label in protocol.schedule(arm):
        if not control.call('start_unit',label):break
        try:
            witness=work(label);control.call('finish_unit')
            if witness is not None:control.call('commit',witness,g,label)
        except (BudgetStop,ValueError,RuntimeError) as exc:
            aborted=label
            if hasattr(backend,'unwind'):backend.unwind()
            if hasattr(control,'fail'):control.fail(exc)
            break
    if export is None and control.current()['stop']=='resource_cap':
        if control.call('start_unit','replay'):
            try:work('replay');control.call('finish_unit')
            except (BudgetStop,ValueError,RuntimeError) as exc:
                aborted='replay'
                if hasattr(backend,'unwind'):backend.unwind()
                if hasattr(control,'fail'):control.fail(exc)
    return dict(aborted_unit=aborted,export=export)


class Store(old.Store):
    def __init__(self,output,start):
        self.output=Path(output).absolute();self.started=start
        require(self.output.name.startswith(NAME+'_') and not self.output.exists() and not self.output.is_symlink()
            and shutil.disk_usage(self.output.parent).free>=CAP+FLOOR,'exclusive own512MiB output+floor')
        self.output.mkdir();self.previous='0'*64;self.index=0;self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=False
        self._write('events.jsonl.gz',b'')
    def check(self):require(time.monotonic()-self.started<=WALL,'outer2400s boundary safety')
    def _write(self,name,data,append=False,final=False):
        if not final:self.check()
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat artifact')
        require(sum(p.stat().st_size for p in self.output.iterdir())+len(data)<=CAP-(0 if final else FINAL)
            and shutil.disk_usage(self.output).free>=FLOOR+len(data),'actual bytes/final reserve/floor')
        with (self.output/name).open('ab' if append else 'xb') as f:require(f.write(data)==len(data),'full bytes');f.flush();os.fsync(f.fileno())
        old.fsync_dir(self.output)


class Control:
    def __init__(self,slot,arm,clock,emit):
        self.clock,self.emit=clock,emit;self.now=clock()
        self.ledger=protocol.Ledger(slot['root_slot'],arm,lambda:self.now,binding=slot)
        emit(dict(kind='ledger_init',now=self.now,root=slot['root_slot'],arm=arm))
    def call(self,name,*args,**kwargs):
        self.now=self.clock();payload=dict(kind='ledger_intent',method=name,args=deepcopy(list(args)),kwargs=deepcopy(kwargs),now=self.now)
        self.emit(payload)
        try:r=getattr(self.ledger,name)(*args,**kwargs)
        except ValueError as exc:
            self.emit(dict(kind='ledger_failure',method=name,error=str(exc)));raise BudgetStop(str(exc)) from exc
        self.emit(dict(kind='ledger_return',method=name,result_sha256=old.digest(r)));return r
    def current(self):return dict(stop=self.ledger.stop,incumbent=deepcopy(self.ledger.incumbent))
    def fail(self,exc):
        self.emit(dict(kind='arm_exception',type=type(exc).__name__,message=str(exc)))
        if self.ledger.stop is None:self.ledger.stop='algorithm_failure'


class HostBackend(old.HostBackend):
    def __init__(self,model,device,t,store,row,emit,control):
        self.control=control;self.emit_native=emit;self.opened=[]
        super().__init__(model,device,t,store,row,self.meter_event)
    def check(self):
        self.store.check();self.t.cuda.synchronize(self.device)
        self.control.call('sample_memory',self.t.cuda.memory_allocated(self.device),self.t.cuda.memory_reserved(self.device))
        state=self.control.current()
        if state['stop'] is not None and not (state['stop']=='resource_cap' and self.control.ledger.unit=='replay'):
            raise BudgetStop(str(state['stop']))
    def meter_event(self,e):
        kind=e['kind']
        if kind in ('forward_intent','module_intent','AD_intent'):
            resource='predict' if kind=='forward_intent' else 'jacobian' if kind=='AD_intent' else e['resource']
            kw={}
            if resource=='jacobian':kw=dict(horizon=len(e['request']['at_logits'])//10,dimension=len(e['request']['at_logits']))
            op=self.control.call('intent',resource,**kw);self.opened.append((kind,op))
        elif kind in ('forward_return','module_return','AD_return'):
            expected={'forward_return':'forward_intent','module_return':'module_intent','AD_return':'AD_intent'}[kind]
            require(self.opened and self.opened[-1][0]==expected,'actual callback lifecycle')
            self.control.call('returned',self.opened[-1][1]);self.opened.pop()
        self.emit_native(dict(kind='native',payload=deepcopy(e)))
    def unwind(self):
        while self.opened:
            _,op=self.opened.pop();self.control.call('returned',op,success=False)
        self.current=None;self.pending=None;self.api=False
    def cpu(self,kind,fn,*args,**kwargs):
        op=self.control.call('intent',kind);self.emit_native(dict(kind='cpu_intent',resource=kind))
        try:r=fn(*args,**kwargs)
        except BaseException:
            self.control.call('returned',op,success=False);raise
        self.control.call('returned',op);self.emit_native(dict(kind='cpu_return',resource=kind));return r
    def _rollout(self,c,commands,tag,raw_tensor=None):
        t=self.t;state=t.tensor(c['latent_history'],device=self.device,dtype=t.float32)[None]
        history=t.tensor(c['action_history'],device=self.device,dtype=t.float32)[None];paths=[]
        actions=t.tensor(commands,device=self.device,dtype=t.float32) if raw_tensor is None else raw_tensor
        for k in range(len(commands)):
            self.check();a=actions[k];a_np=a.detach().cpu().numpy().copy();n=len(self.row['forwards'])
            request=packet_request(n,c,a_np,tag+'/'+str(k));self.event(dict(kind='forward_intent',request=request))
            f=dict(index=n,tag=tag+'/'+str(k),request=request,input=deepcopy(c),raw=old.blob(a_np),modules=[],completed=False)
            self.row['forwards'].append(f);self.current=f;self.event(dict(kind='forward_inputs',index=n,context=c,raw=f['raw']))
            state,history=self.transition(state,history,a);self.check()
            c=dict(c,latent_history=state[0].detach().cpu().tolist(),action_history=history[0].detach().cpu().tolist())
            f.update(completed=True,output=deepcopy(c));validate_packet(f)
            self.event(dict(kind='forward_return',index=n,context=c,completed=True));self.current=None;paths.append(deepcopy(c))
        return state,paths


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build(protocol_prepared,protocol_sha):
    require(old.sha(protocol_prepared)==protocol_sha,'actual external v2 protocol SHA')
    frozen=json.loads(Path(protocol_prepared).read_bytes());a=frozen['authority']
    old.need_equal(frozen,protocol.build(Path(a['report']).parent,a['receipt']),'actual complete v2 protocol rebuilt')
    p=deepcopy(frozen);p.update(schema=NAME+'.prepare',status='runtime_prepared_MAIN_CUDA_only',runtime_implementation_required=False,
        protocol_prepared=str(Path(protocol_prepared).resolve()),protocol_sha256=protocol_sha,
        original_protocol=deepcopy(frozen),output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,outer_wall_seconds=WALL)
    p['policy']=old.verify(a['prepare'],a['prepare_sha256'])['policy']
    p['bindings_sha256'][str(Path(protocol_prepared).resolve())]=protocol_sha
    for f in own_files():p['bindings_sha256'][str(f)]=old.sha(f)
    return p


def write_new(path,p):
    path=Path(path);require(path.is_absolute() and path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink(),'new own absolute metadata')
    b=old.raw(p);cap=128*1024**2 if p.get('schema')==NAME+'.consume' else 4*1024**2
    require(len(b)<=cap,'own prepare/receipt separate byte caps')
    with path.open('xb') as f:require(f.write(b)==len(b),'complete metadata');f.flush();os.fsync(f.fileno())
    old.fsync_dir(path.parent);return dict(path=str(path),sha256=old.sha(path))


def verify(prepared,expected):
    require(old.sha(prepared)==expected,'own external prepare SHA');p=json.loads(Path(prepared).read_bytes())
    old.need_equal(p,build(p['protocol_prepared'],p['protocol_sha256']),'entire own preparation rebuilt');return p


def run_cuda(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA authority before I/O/imports')
    start=time.monotonic();p=verify(prepared,expected);store=Store(output,start);rows=[];errors=[];model=None;before=after=None;host=None
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,before=old.loader.loaded_model(p,torch,device,store,expected)
        def clock():torch.cuda.synchronize(device);return time.monotonic()
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
            for i,(root,arm) in enumerate(p['root_arm_order']):
                slot=p['slots'][root];row=dict(index=i,root_slot=root,arm=arm,forwards=[],AD=[]);rows.append(row)
                emit=lambda e,i=i:store.event(dict(kind='arm_event',index=i,payload=e))
                control=Control(slot,arm,clock,emit);backend=HostBackend(model,device,torch,store,row,emit,control)
                try:
                    with backend.observe():row['result']=drive(slot,arm,backend,control)
                except BaseException as exc:
                    backend.unwind();control.fail(exc);row['result']=dict(aborted_unit='outer',export=None)
                control.call('seal');row['accounting']=control.call('report')
                store.put(f'arm{i}.json.gz',row)
                store.event(dict(kind='arm_sealed',index=i,row_sha256=old.digest(row)))
    except BaseException as exc:errors.append(old.loader.enc.error('execution',exc))
    finally:
        if model is not None:
            try:old.loader.enc.frozen(model);after=old.loader.enc.tensor_hash(model)
            except BaseException as exc:errors.append(old.loader.enc.error('model_final',exc))
        try:old.parent.check_files(p)
        except BaseException as exc:errors.append(old.loader.enc.error('source_final',exc))
        for row in rows:
            if not (store.output/f'arm{row["index"]}.json.gz').exists():store.put(f'arm{row["index"]}.json.gz',row,final=True)
        store.close()
        report=dict(schema=NAME+'.report',prepared=str(Path(prepared).absolute()),prepare_sha256=expected,
            errors=errors,model_before=before,model_after=after,model_unchanged=before is not None and before==after==p['expected_model_hash'],
            host=host,rows=[dict(index=r['index'],root_slot=r['root_slot'],arm=r['arm'],file=f'arm{r["index"]}.json.gz') for r in rows],
            fixed_denominator=4,status='recorded_not_admitted' if len(rows)==4 and not errors else 'failed_partial',
            arm_stops=[r.get('accounting',{}).get('stop') for r in rows],
            wall_seconds=time.monotonic()-start,equal_FLOPs=False,total_FLOPs=None,engine_internal_backward_calls=None,
            P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False,
            artifact_sha256={f.name:old.sha(f) for f in store.output.iterdir()})
        store.put('report.json',report,final=True)
    return dict(report=str(store.output/'report.json'),sha256=old.sha(store.output/'report.json'),errors=errors)


def replay_accounting(events,slot,arm):
    """Independent chronological ledger reconstruction, no trusted flags."""
    require(events and events[0]['kind']=='ledger_init','actual ledger start')
    init=events[0];require(init['root']==slot['root_slot'] and init['arm']==arm,'fixed root/arm ledger')
    now=[init['now']];l=protocol.Ledger(slot['root_slot'],arm,lambda:now[0],binding=slot)
    records=[];native=[];cpu=[];stack=[];started=set();ended=set();i=1;previous=None
    allowed={'start_unit','finish_unit','commit','sample_memory','intent','returned','seal','report'}
    while i<len(events):
        e=events[i];kind=e['kind']
        if kind=='ledger_intent':
            require(set(e)=={'kind','method','args','kwargs','now'} and e['method'] in allowed,'exact allowed ledger command')
            require(type(e['now']) in (int,float) and np.isfinite(e['now']) and e['now']>=now[0],'full monotone command clock, including post-seal')
            now[0]=e['now'];result=None;error=None
            try:result=getattr(l,e['method'])(*e['args'],**e['kwargs'])
            except ValueError as exc:error=str(exc)
            require(i+1<len(events),'every durable ledger command closed');end=events[i+1]
            if error is None:old.need_equal(end,dict(kind='ledger_return',method=e['method'],result_sha256=old.digest(result)),'exact ledger return')
            else:old.need_equal(end,dict(kind='ledger_failure',method=e['method'],error=error),'exact budget/error refusal')
            rec=dict(command=e,result=result,error=error,state=dict(stop=l.stop,incumbent=deepcopy(l.incumbent)))
            records.append(rec);previous=rec;i+=2;continue
        if kind in ('native','cpu_intent','cpu_return'):
            payload=e['payload'] if kind=='native' else e
            k=payload['kind'];starts={'forward_intent':'predict','module_intent':payload.get('resource'),'AD_intent':'jacobian','cpu_intent':payload.get('resource')}
            ends={'forward_return','module_return','AD_return','cpu_return'}
            if k in starts:
                require(previous is not None and previous['error'] is None and previous['command']['method']=='intent'
                    and previous['command']['args']==[starts[k]],'actual native callback bound to its charged intent')
                op=previous['result'];require(op not in started,'no duplicate native start');started.add(op);stack.append((k,op))
                if k=='AD_intent':
                    r=payload['request'];d=len(r['at_logits'])
                    old.need_equal(previous['command']['kwargs'],dict(horizon=d//10,dimension=d),'exact requested J shape/horizon')
            elif k in ends:
                expected={'forward_return':'forward_intent','module_return':'module_intent','AD_return':'AD_intent','cpu_return':'cpu_intent'}[k]
                require(stack and stack[-1][0]==expected,'native strict callback nesting')
                _,op=stack.pop()
                require(previous is not None and previous['error'] is None and previous['command']['method']=='returned'
                    and previous['command']['args']==[op] and previous['command']['kwargs'].get('success',True) is True,'actual native return bound to charged operation')
                ended.add(op)
            else:require(kind=='native' and k in ('forward_inputs','module_inputs','predict_API_intent','predict_API_return','local_selection'),'no unknown native event')
            (native if kind=='native' else cpu).append(payload);previous=None;i+=1;continue
        if kind=='arm_exception':
            require(set(e)=={'kind','type','message'} and isinstance(e['message'],str),'retained actual exception')
            if l.stop is None:l.stop='algorithm_failure'
            previous=None;i+=1;continue
        raise ValueError('unknown full arm event '+str(kind))
    require(started==set(range(len(l.operations))),'every charged callback has actual native dispatch intent')
    require(ended=={o['index'] for o in l.operations if o['success'] is True},'every successful callback has native return')
    require(all(l.operations[op]['closed'] and l.operations[op]['success'] is False for _,op in stack),'incomplete native callbacks must retain failure, no refund')
    require(records[-2]['command']['method']=='seal' and records[-1]['command']['method']=='report'
        and sum(r['command']['method']=='seal' for r in records)==1,'unique terminal seal/report')
    report=l.report();require(report['sealed'] and not report['pending'],'all attempted operations accounted at seal')
    require(report['peak_allocated_bytes'] is not None,'actual per-arm CUDA allocator samples required')
    return report,records,native,cpu


class ReplayControl:
    def __init__(self,records,final):
        self.records=[r for r in records if r['command']['method'] in ('start_unit','finish_unit','commit')]
        self.i=0;self.state=dict(stop=None,incumbent=None);self.final=final
    def call(self,name,*args,**kwargs):
        require(self.i<len(self.records),'no extra algorithm control call');r=self.records[self.i];self.i+=1
        old.need_equal({k:r['command'][k] for k in ('method','args','kwargs')},dict(method=name,args=list(args),kwargs=kwargs),'same fixed algorithm control request')
        self.state=deepcopy(r['state'])
        if r['error'] is not None:raise BudgetStop(r['error'])
        return r['result']
    def current(self):
        r=deepcopy(self.state)
        if self.i==len(self.records):r['stop']=self.final['stop']
        return r
    def fail(self,exc):pass  # exception contents independently retained/replayed in ledger timeline


def partial_packet(f):
    """Validate returned modules of an unfinished native forward, never export it."""
    old.need_equal(f['request'],packet_request(f['index'],f['input'],old.unblob(f['raw']),f['tag']),'partial literal request')
    require(f['completed'] is False and len(f['modules'])<=3,'actual unfinished forward')
    c=f['input'];tokens=np.concatenate((np.asarray(c['action_history'],np.float32),old.kernel.normalized_history(old.unblob(f['raw']))))
    outputs=[]
    for k,e in enumerate(f['modules']):
        require(e['index']==3*f['index']+k and e['resource']==protocol.MODULES[k] and e['completed'] is True,'partial returned module prefix')
        ins=[old.unblob(b) for b in e['inputs']];out=old.unblob(e['output'])
        wanted=[tokens[None]] if k==0 else [np.asarray(c['latent_history'],np.float32)[None],outputs[0]] if k==1 else [outputs[1].reshape(3,192)]
        require(len(ins)==len(wanted) and all(np.array_equal(a,b) for a,b in zip(ins,wanted))
            and out.shape==((3,192) if k==2 else (1,3,192)),'partial native tensor IO, original normalization')
        outputs.append(out)


class SavedBackend:
    def __init__(self,row,native,cpu,load):
        self.row,self.native,self.cpu_events,self.load=row,native,cpu,load
        self.n=self.ad=self.cursor=self.cpu_cursor=0
    def accept(self,e):
        require(self.cursor<len(self.native),'missing native event')
        old.need_equal(self.native[self.cursor],e,'exact complete native event chronology');self.cursor+=1
    def event(self,e):self.accept(e)
    def rollout(self,c,commands,tag):
        paths=[]
        for k,a in enumerate(commands):
            if self.n==len(self.row['forwards']):raise BudgetStop('saved interrupted before next forward')
            f=self.row['forwards'][self.n];require(f['index']==self.n,'all actual forward ordinals contiguous');self.n+=1
            req=packet_request(f['index'],c,np.asarray(a,np.float32),tag+'/'+str(k))
            old.need_equal(f['request'],req,'scheduled raw/context/tag exact, not only recorded self-consistency')
            old.need_equal(f['input'],c,'full incoming context binding')
            require(old.unblob(f['raw']).tobytes()==np.asarray(a,np.float32).tobytes(),'literal raw bits, no repair')
            self.accept(dict(kind='forward_intent',request=req));self.accept(dict(kind='forward_inputs',index=f['index'],context=c,raw=f['raw']))
            if not f['completed']:
                partial_packet(f)
                api_entered=False
                for kmod,entry in enumerate(f['modules']):
                    if kmod==1:
                        self.accept(dict(kind='predict_API_intent',forward=f['index'],completed=False));api_entered=True
                    self.accept(dict(kind='module_intent',index=entry['index'],resource=entry['resource'],completed=False))
                    self.accept(dict(kind='module_inputs',index=entry['index'],inputs=entry['inputs']))
                    self.accept(dict(kind='module_return',entry=entry))
                # Consume only observed API prefixes. Third module return does
                # NOT imply its enclosing model.predict wrapper has returned.
                kmod=len(f['modules'])
                if kmod==1 and self.cursor<len(self.native) and self.native[self.cursor]['kind']=='predict_API_intent':
                    self.accept(dict(kind='predict_API_intent',forward=f['index'],completed=False));api_entered=True
                if kmod==3 and self.cursor<len(self.native) and self.native[self.cursor]['kind']=='predict_API_return':
                    self.accept(dict(kind='predict_API_return',forward=f['index'],completed=True))
                if kmod<3 and self.cursor<len(self.native):
                    require(kmod!=1 or api_entered,'predictor cannot enter before actual API intent')
                    r=protocol.MODULES[kmod];idx=3*f['index']+kmod
                    self.accept(dict(kind='module_intent',index=idx,resource=r,completed=False))
                    if self.cursor<len(self.native):
                        e=self.native[self.cursor];require(e['kind']=='module_inputs' and e['index']==idx,'pending module input event')
                        outs=[old.unblob(m['output']) for m in f['modules']]
                        tokens=np.concatenate((np.asarray(c['action_history'],np.float32),old.kernel.normalized_history(a)))
                        wanted=[tokens[None]] if kmod==0 else [np.asarray(c['latent_history'],np.float32)[None],outs[0]] if kmod==1 else [outs[1].reshape(3,192)]
                        require(len(e['inputs'])==len(wanted) and all(np.array_equal(old.unblob(b),x) for b,x in zip(e['inputs'],wanted)),'pending actual module inputs')
                        self.cursor+=1
                raise BudgetStop('saved unfinished forward')
            for e in old.module_events(f):self.accept(e)
            c=validate_packet(f);self.accept(dict(kind='forward_return',index=f['index'],context=c,completed=True));paths.append(deepcopy(c))
        return paths
    def jacobian(self,c,u,tag,logit):
        if self.ad==len(self.row['AD']):raise BudgetStop('saved interrupted before next AD')
        e=self.row['AD'][self.ad];idx=self.ad;self.ad+=1
        req=dict(index=idx,tag=tag,input_context_sha256=old.digest(c),at_logits=np.asarray(u).tolist(),raw=old.chart(u).tolist(),
            coordinates='logit_zero' if logit else 'raw_incumbent',shape=[192,len(u)],mathematical_VJP=192,vectorize=True,completed=False)
        old.need_equal(e['request'],req,'same literal J/context/coordinates')
        self.accept(dict(kind='AD_intent',request=req));paths=self.rollout(c,old.chart(u),tag+'/forward')
        if not e['completed']:raise BudgetStop('saved AD primal returned, matrix incomplete')
        old.need_equal(e['contexts'],paths,'AD all imagined contexts');a=self.load(e['array'])
        require(a.dtype==np.float32 and a.shape==(192,len(u)),'native J full array')
        self.accept(dict(kind='AD_return',index=idx,array=e['array'],contexts=paths,completed=True));self.last_ad_contexts=paths
        return a.astype(np.float64)
    def cpu(self,kind,fn,*args,**kwargs):
        if self.cpu_cursor==len(self.cpu_events):raise BudgetStop('saved stopped before CPU callback')
        old.need_equal(self.cpu_events[self.cpu_cursor],dict(kind='cpu_intent',resource=kind),'fixed CPU source schedule');self.cpu_cursor+=1
        result=fn(*args,**kwargs)
        if self.cpu_cursor==len(self.cpu_events):raise BudgetStop('saved CPU failure')
        old.need_equal(self.cpu_events[self.cpu_cursor],dict(kind='cpu_return',resource=kind),'exact CPU callback return');self.cpu_cursor+=1
        return result
    def complete(self):
        require(self.n==len(self.row['forwards']) and self.ad==len(self.row['AD']) and self.cursor==len(self.native)
            and self.cpu_cursor==len(self.cpu_events),'all calls and unknown/partial events consumed, no filtering')


def validate_arm(row,events,slot,load):
    accounting,records,native,cpu=replay_accounting(events,slot,row['arm'])
    old.need_equal(accounting,row['accounting'],'full actual accounting reconstructed')
    backend=SavedBackend(row,native,cpu,load);control=ReplayControl(records,accounting)
    result=drive(slot,row['arm'],backend,control);backend.complete()
    require(control.i==len(control.records),'every algorithm control command consumed')
    old.need_equal(result,row['result'],'entire fixed algorithm and anytime/export rebuilt')
    require(len(row['forwards'])==accounting['attempted']['predict'] and sum(f['completed'] for f in row['forwards'])==accounting['completed']['predict']
        and len(row['AD'])==accounting['attempted']['jacobian'] and sum(e['completed'] for e in row['AD'])==accounting['completed']['jacobian'],'actual array/callback census')
    return accounting


def journal(output):
    plain,eof=old.parent.overlay.blind.inflate((Path(output)/'events.jsonl.gz').read_bytes(),1024*1024**2)
    require(eof and plain.endswith(b'\n'),'own complete bounded1GiB expanded journal');previous='0'*64;out=[]
    for i,line in enumerate(plain.splitlines()):
        e=json.loads(line);require(set(e)=={'index','previous_sha256','payload','event_sha256'}
            and type(e['index']) is int and e['index']==i and e['previous_sha256']==previous,'full ordinal/hash envelope')
        unsigned={k:v for k,v in e.items() if k!='event_sha256'}
        require(old.digest(unsigned)==e['event_sha256'] and old.raw(e)==line+b'\n','exact canonical journal bytes')
        previous=e['event_sha256'];out.append(e['payload'])
    return out


def consume(output,expected):
    output=Path(output).absolute();require(old.sha(output/'report.json')==expected,'actual external report SHA')
    r=old.parent.overlay.read(output/'report.json',4*1024**2);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='recorded_not_admitted' and r['errors']==[]
        and r['fixed_denominator']==4 and r['model_unchanged'] is True
        and r['model_before']==r['model_after']==p['expected_model_hash'],'full four engineering records, frozen model')
    names=set(r['artifact_sha256'])
    require(set(f.name for f in output.iterdir())==names|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=CAP,'exact bounded artifact census')
    for name,h in r['artifact_sha256'].items():
        require(Path(name).name==name and (output/name).is_file() and not (output/name).is_symlink() and old.sha(output/name)==h,'every stable artifact SHA')
    old.need_equal(old.parent.overlay.read(output/'binding.json'),dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'exact execution binding')
    loader=old.parent.overlay.read(output/'loader.json');old.loader.validate_loader(loader,p,r['prepare_sha256'])
    events=journal(output);cursor=0
    def accept(e):
        nonlocal cursor
        require(cursor<len(events),'missing journal event');old.need_equal(events[cursor],e,'entire journal chronology');cursor+=1
    for f in loader['files']:
        accept(dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']));accept(dict(kind='consumed_buffer_frozen',receipt=f))
    used={'binding.json','loader.json','events.jsonl.gz'};plans=[];accounts=[];previous_arm_end=None
    require(len(r['rows'])==4,'fixed4 denominator')
    for i,(root,arm) in enumerate(p['root_arm_order']):
        meta=dict(index=i,root_slot=root,arm=arm,file=f'arm{i}.json.gz');old.need_equal(r['rows'][i],meta,'fixed serial counterbalanced order')
        used.add(meta['file']);row=old.parent.overlay.read(output/meta['file'],512*1024**2)
        require(row['index']==i and row['root_slot']==root and row['arm']==arm,'actual whole row identity')
        arm_events=[]
        while cursor<len(events) and events[cursor].get('kind')=='arm_event':
            e=events[cursor];require(set(e)=={'kind','index','payload'} and e['index']==i,'arm envelope index');arm_events.append(e['payload']);cursor+=1
        require(arm_events and (previous_arm_end is None or arm_events[0]['now']>=previous_arm_end),'serial arms, no overlapping clock windows')
        previous_arm_end=max(e['now'] for e in arm_events if 'now' in e)
        def load(d):
            name=d['file'];require(name==f'root{i}_J{len([n for n in used if n.startswith("root"+str(i)+"_J")]):03}.npy'
                and old.sha(output/name)==d['sha256'],'ordered actual J filename/hash')
            b=(output/name).read_bytes();s=io.BytesIO(b);a=np.load(s,allow_pickle=False)
            require(s.tell()==len(b) and list(a.shape)==d['shape'] and d['dtype']=='float32' and np.isfinite(a).all(),'literal native array bytes')
            used.add(name);return a
        account=validate_arm(row,arm_events,p['slots'][root],load);accounts.append(account)
        accept(dict(kind='arm_sealed',index=i,row_sha256=old.digest(row)))
        witness=row['result']['export'];plan=None
        if witness is not None:
            slot=p['slots'][root];plan=dict(schema=NAME+'.plan',root_slot=root,arm=arm,task_id=slot['task_id'],
                task_sha256=slot['task_sha256'],root_context=slot['packet']['root_context'],goal=slot['packet']['goal'],
                warm_raw10=slot['warm_raw10'],commands=old.chart(witness['logits']).reshape(15,2).tolist(),
                raw_float32_bits=old.chart(witness['logits']).tobytes().hex(),contexts=witness['contexts'],
                model_hash=p['expected_model_hash'],model_Q=old.objective(witness['contexts'][-1],slot['packet']['goal']),
                report_sha256=expected,P1_admission=False,P2_admission=False,selection_uses_physics=False)
            plan['selection_digest']=old.digest(plan)
        plans.append(plan)
    require(cursor==len(events) and used==names,'no unknown files or terminal events')
    old.parent.check_files(p)
    for name,h in r['artifact_sha256'].items():require(old.sha(output/name)==h,'saved outputs unchanged')
    require(old.sha(output/'report.json')==expected,'report unchanged')
    require(r['arm_stops']==[a['stop'] for a in accounts],'actual arm failure/budget stops preserved in report')
    bindings=deepcopy(p['bindings_sha256']);bindings[str(Path(r['prepared']).resolve())]=r['prepare_sha256']
    bindings[str(output/'report.json')]=expected
    for n,h in r['artifact_sha256'].items():bindings[str(output/n)]=h
    summaries=[]
    for a in accounts:
        s={k:v for k,v in a.items() if k!='events'};s['ledger_events_sha256']=old.digest(a['events']);summaries.append(s)
    return dict(schema=NAME+'.consume',status='wallcapped_records_verified_not_admitted',plans=plans,accounts=summaries,bindings=bindings,
        report_sha256=expected,fixed_denominator=4,known_roots=2,fresh_roots=0,
        comparisons=[protocol.compare_reports([accounts[i] for i in range(4) if accounts[i]['root']==root]) for root in (0,1)],
        total_FLOPs=None,engine_internal_backward_calls=None,equal_FLOPs=False,equal_total_compute=False,
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,P2_admission=False,P2_mechanism=False,physical_release=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('protocol-prepared','protocol-sha','output','prepared','prepare-sha','report-sha','authorization','receipt'):ap.add_argument('--'+k)
    a=ap.parse_args()
    if a.command=='prepare':result=write_new(a.output,build(a.protocol_prepared,a.protocol_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);result=dict(verified=True,order=p['root_arm_order'])
    elif a.command=='run':result=run_cuda(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:
        receipt=consume(a.output,a.report_sha)
        result=write_new(a.receipt,receipt) if a.receipt else {k:v for k,v in receipt.items() if k not in ('plans','accounts')}
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
