"""MAIN-only common-cap three-arm calibration. Import/prepare/consume: no torch.

No global monkeypatch, old GRASP strict, training, physical executor or outcomes.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np
from experiments.spectral_reachability import agent_p2_wallcapped_cuda_20261029_v1 as wall
from experiments.spectral_reachability import agent_p2_history_joint30_grasp_cuda_20261029_v1 as grasp
from experiments.spectral_reachability import agent_p2_history_joint30_grasp_saved_consumer_20261029_v2 as adam_reference

NAME='agent_p2_common_resource_grasp_cuda_20261029_v1'
AUTH='MAIN_P2_COMMON_RESOURCE_GRASP_HOST_CUDA_ONLY'
old=wall.old
ROOT=old.ROOT
need=old.require
equal=old.need_equal
CAP,FINAL,FLOOR,OUTER=1536*1024**2,96*1024**2,512*1024**2,3600
SEARCH=450.
ORDER=[[0,'joint'],[0,'composition'],[0,'grasp'],[1,'grasp'],[1,'composition'],[1,'joint']]
PINS={wall.__file__:'5e733e75f56b3dd2f5598d6e9c4ae7878c1e3dd0d613c5b4777c1ecd891b9776',
      grasp.__file__:'6c44630f75259ce107b5b001a8dfc7f50b60c4f7b6ab920371e2994cde1ca9a1',
      adam_reference.__file__:'20863c44843ef76434067f3241c5b310e044160a6d21132ee8d722dae63bf34f',
      **adam_reference.ARITHMETIC_PINS}


def phases():
    for k in range(532):
        yield 'joint',k,None
        if (k+1)%50==0:
            for j in range(10):yield 'sync',k,j


def contract():
    return dict(name=NAME,roots=[0,1],condition='off',order=ORDER,fixed_denominator=6,
        raw_steps=15,batch=1,context_length=3,predict_API_cap=1902,predict_samples_cap=1902,
        search_wall_seconds=SEARCH,outer_safety_seconds=OUTER,
        grasp_joint=532,grasp_sync=100,grasp_scalar_backward_cap=632,final_replay=6,
        search_scope='synchronized clock; nominal/bank/AD/optimization/CPU/capture/search IO/export; not loader or saved strict',
        wall_cooperative=True,padding=False,retries=0,failed_attempts_refunded=False,
        total_FLOPs=None,engine_internal_backward_calls=None,equal_FLOPs=False,equal_total_compute=False,
        P1_admission=False,P2_admission=False,P2_mechanism=False,fresh=False,physical_release=False,
        official_GRASP_reproduction=False,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        allocated_cap_bytes=4*1024**3,reserved_cap_bytes=6*1024**3)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def join_tasks(slots):
    need(len(slots)==2 and [(s['root_slot'],s['condition']) for s in slots]==[(0,'off'),(1,'off')],'only original two off slots')
    tasks=[]
    for s in slots:
        e=grasp.inputs.parse(grasp.inputs.read_bound(s['task_file'],s['task_sha256'],compressed=True));t=e['task']
        need(grasp.inputs.core.task_id(t)==s['task_id']==e['task_id'],'owner task identity')
        equal(t['root_context'],s['packet']['root_context'],'same complete causal context')
        equal(t['goal'],s['packet']['goal'],'same goal')
        equal(t['projection'],s['q'],'same original Q');equal(s['q'],np.eye(192).tolist(),'identity Q')
        equal(t['root_raw_action_history'],s['warm_raw10'],'same raw warm10')
        for k,v in zip(('action_mean','action_std'),('mean','std')):
            equal(t[k],grasp.inputs.normalization()[v],'same exact raw normalizer')
        need(s['task_sha256']==grasp.inputs.TASK_SHAS[s['root_slot']],'original pinned source task')
        tasks.append(t)
    return tasks


def build(protocol_prepared,protocol_sha):
    for f,h in PINS.items():need(old.sha(f)==h,'frozen kernel source '+str(f))
    base=wall.build(protocol_prepared,protocol_sha)
    join_tasks(base['slots'])
    # Do not inherit contradictory four-arm/cap512MiB/outer2400s fields.
    p=dict(schema=NAME+'.prepare',contract=contract(),root_arm_order=deepcopy(ORDER),
        protocol_prepared=str(Path(protocol_prepared).resolve()),protocol_sha256=protocol_sha,
        slots=deepcopy(base['slots']),policy=base['policy'],expected_model_hash=base['expected_model_hash'],
        wall_protocol_authority=deepcopy(base['authority']),bindings_sha256=deepcopy(base['bindings_sha256']))
    for f in grasp.source_files()+own_files()+[ROOT/'innerdocs/agent_p2_common_resource_grasp_spec_20261029_v1.md']:
        p['bindings_sha256'][str(f.resolve())]=old.sha(f)
    p['bindings_sha256'].update({str(Path(f).resolve()):h for f,h in PINS.items()})
    old.parent.check_files(p)
    return p


def write_new(path,value,cap=8*1024**2):
    p=Path(path).absolute();b=old.raw(value)
    need(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(b)<=cap,'exclusive own bounded metadata')
    need(shutil.disk_usage(p.parent).free>=FLOOR+len(b),'metadata disk floor')
    with p.open('xb') as f:need(f.write(b)==len(b),'full metadata');f.flush();os.fsync(f.fileno())
    old.fsync_dir(p.parent);return dict(path=str(p),sha256=old.sha(p))


def verify(prepared,expected):
    need(old.sha(prepared)==expected,'external prepare SHA')
    p=old.parent.overlay.read(prepared,8*1024**2)
    equal(p,build(p['protocol_prepared'],p['protocol_sha256']),'complete common prepare rebuild')
    return p


class Store(old.Store):
    def __init__(self,output,start):
        self.output=Path(output).absolute();self.started=start
        need(self.output.name.startswith(NAME+'_') and not self.output.exists() and not self.output.is_symlink(),'exclusive own directory')
        need(shutil.disk_usage(self.output.parent).free>=CAP+FLOOR,'total output cap and floor')
        self.output.mkdir();old.fsync_dir(self.output.parent)
        self.previous='0'*64;self.index=0;self.comp=zlib.compressobj(6,zlib.DEFLATED,31);self.closed=False
        self._write('events.jsonl.gz',b'')
    def check(self):need(time.monotonic()-self.started<OUTER,'outer safety3600s, not search scope')
    def _write(self,name,data,append=False,final=False):
        if not final:self.check()
        need(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat regular artifact')
        need(sum(f.stat().st_size for f in self.output.iterdir())+len(data)<=CAP-(0 if final else FINAL)
            and shutil.disk_usage(self.output).free>=FLOOR+len(data),'output cap/reserve/floor including partials')
        with (self.output/name).open('ab' if append else 'xb') as f:need(f.write(data)==len(data),'full artifact');f.flush();os.fsync(f.fileno())
        old.fsync_dir(self.output)


class Stop(Exception):
    """Cooperative search refusal, distinct from execution failure."""


G_CAPS=dict(packet=1902,predict_API=1902,predict_samples=1902,action_encoder=1902,predictor=1902,
    pred_proj=1902,scalar_backward=632,Adam_step=632,virtual_noise=532)
G_ATTEMPTS=dict(forward_attempt='packet',predict_attempt='predict_API',backward_attempt='scalar_backward',
    optimizer_attempt='Adam_step',noise_attempt='virtual_noise')
G_RETURNS=dict(forward_return='packet',predict_return='predict_API',backward_return='scalar_backward',
    optimizer_return='Adam_step',noise='virtual_noise')


class GAccount:
    """Pure state machine, replayed with saved times; no assumed neural counters."""
    def __init__(self,start):
        need(type(start) in (int,float) and np.isfinite(start),'finite synchronized start')
        self.start=self.last=start;self.attempted=dict.fromkeys(G_CAPS,0);self.completed=dict.fromkeys(G_CAPS,0)
        self.active=[];self.units=0;self.in_unit=False;self.replay=False;self.stop=None;self.ended=None
        self.peak_allocated=None;self.peak_reserved=None
    def tick(self,now):
        need(type(now) in (int,float) and np.isfinite(now) and now>=self.last and self.ended is None,'monotone unsealed clock')
        self.last=now
    def admit_unit(self,now,replay=False):
        self.tick(now);need(not self.in_unit and not self.replay and not self.active,'unit boundary')
        allowed=now-self.start<SEARCH and self.stop in (None,'resource_cap')
        count=6 if replay else 9
        if self.attempted['packet']+count>1902:allowed=False;self.stop='resource_cap'
        if now-self.start>=SEARCH:self.stop='wall_cap'
        if allowed:
            if replay:self.replay=True
            else:need(self.units<632,'at most632 optimizer units');self.in_unit=True
        return allowed
    def event(self,e,now):
        self.tick(now);kind=e['kind'];resource=G_ATTEMPTS.get(kind);returned=G_RETURNS.get(kind)
        if kind=='optimization_attempt':need(self.in_unit and not self.replay and not self.active,'optimization after admitted unit')
        if kind=='initialized':need(not self.in_unit and not self.replay and self.units==0,'initialization before units')
        if kind in ('actions_frozen','export_complete'):need(self.replay and not self.active,'export only in admitted replay')
        if kind=='module_attempt':resource=e['resource']
        if kind=='module_return':returned=e['resource']
        if resource:
            need(resource in G_CAPS and (self.in_unit or self.replay),'known active resource')
            if now-self.start>=SEARCH:self.stop='wall_cap';return False
            charge={resource:1}
            if resource=='predictor':charge['predict_samples']=1
            if any(self.attempted[k]+v>G_CAPS[k] for k,v in charge.items()):self.stop='resource_cap';return False
            for k,v in charge.items():self.attempted[k]+=v
            self.active.append((kind,resource))
        elif returned:
            expected={'forward_return':'forward_attempt','predict_return':'predict_attempt','backward_return':'backward_attempt',
                'optimizer_return':'optimizer_attempt','noise':'noise_attempt','module_return':'module_attempt'}[kind]
            need(self.active and self.active[-1]==(expected,returned),'strict nested resource return')
            self.active.pop();self.completed[returned]+=1
            if returned=='predictor':self.completed['predict_samples']+=1
            if now-self.start>=SEARCH:self.stop='wall_cap'
        return True
    def memory(self,now,allocated,reserved):
        self.tick(now)
        need(type(allocated) is int and type(reserved) is int and 0<=allocated<=reserved,'actual allocator sample')
        self.peak_allocated=max(self.peak_allocated or 0,allocated);self.peak_reserved=max(self.peak_reserved or 0,reserved)
        return allocated<=4*1024**3 and reserved<=6*1024**3
    def finish(self,now):
        self.tick(now);need(self.in_unit and not self.active,'complete update unit')
        self.in_unit=False;self.units+=1
        accepted=now-self.start<SEARCH
        if not accepted:self.stop='wall_cap'
        return accepted
    def summary(self):
        return dict(start=self.start,end=self.ended,elapsed_seconds=self.last-self.start,wall_overrun=self.last-self.start>SEARCH,
            attempted=deepcopy(self.attempted),completed=deepcopy(self.completed),pending=[list(x) for x in self.active],
            completed_units=self.units,incomplete_unit=self.in_unit,stop=self.stop,
            peak_allocated_bytes=self.peak_allocated,peak_reserved_bytes=self.peak_reserved,
            caps=deepcopy(G_CAPS),total_FLOPs=None,engine_internal_backward_calls=None)


class GControl:
    def __init__(self,clock,emit):
        self.clock,self.emit=clock,emit;self.a=GAccount(clock());emit(dict(kind='g_start',now=self.a.start))
    def unit(self,replay=False):
        now=self.clock();ok=self.a.admit_unit(now,replay)
        self.emit(dict(kind='g_admission',now=now,replay=replay,allowed=ok))
        if not ok:raise Stop(str(self.a.stop))
    def event(self,e):
        now=self.clock();ok=self.a.event(e,now)
        self.emit(dict(kind='g_native',now=now,allowed=ok,payload=deepcopy(e)))
        if not ok:raise Stop(str(self.a.stop))
    def finish(self,x):
        now=self.clock();accepted=self.a.finish(now)
        self.emit(dict(kind='g_commit',now=now,accepted=accepted,x=deepcopy(x)))
        return accepted
    def memory(self,allocated,reserved):
        now=self.clock();ok=self.a.memory(now,allocated,reserved)
        self.emit(dict(kind='g_memory',now=now,allocated=allocated,reserved=reserved,allowed=ok))
        need(ok,'actual CUDA memory cap')
    def seal(self,error=None):
        now=self.clock();self.a.tick(now)
        if error is not None:
            self.a.stop='algorithm_failure';self.emit(dict(kind='g_error',now=now,type=type(error).__name__,message=str(error)))
        if self.a.stop is None:self.a.stop='natural_completion'
        self.a.ended=now;s=self.a.summary();self.emit(dict(kind='g_seal',now=now,summary=s));return s


class GNative(grasp.Native):
    """Inherit only raw native kernels, not old optimize/build/strict."""
    def check(self):
        self.store.check();self.t.cuda.synchronize(self.device)
        self.event.__self__.memory(self.t.cuda.memory_allocated(self.device),self.t.cuda.memory_reserved(self.device))
    @contextmanager
    def observe(self):
        # Unlike the old complete-only context, a retained unfinished callback
        # is legitimate here. GAccount and the new consumer authenticate it.
        handles=[];pending=None
        def enter(name):
            def hook(module,args):
                nonlocal pending
                need(self.active is not None and pending is None and name==('action_encoder','predictor','pred_proj')[self.active],'native module order')
                check_tensor_abi(name,[tuple(x.shape) for x in args])
                self.check();self.event(dict(kind='module_attempt',index=self.n,resource=name,inputs=[self.snap(x) for x in args]));pending=(module,name)
            return hook
        def returned(module,args,out):
            nonlocal pending
            need(pending is not None and pending[0] is module,'native paired module return')
            self.check();self.event(dict(kind='module_return',index=self.n,resource=pending[1],output=self.snap(out)))
            pending=None;self.active+=1
        def forbid(module,args):raise ValueError('new encoder/projector forbidden')
        try:
            for name in ('action_encoder','predictor','pred_proj'):
                module=getattr(self.model,name);handles.extend((module.register_forward_pre_hook(enter(name)),module.register_forward_hook(returned)))
            for name in ('encoder','projector'):handles.append(getattr(self.model,name).register_forward_pre_hook(forbid))
            yield
        finally:
            for handle in reversed(handles):handle.remove()


def check_tensor_abi(name,shapes):
    expected={'action_encoder':[(1,3,10)],'predictor':[(1,3,192),(1,3,192)],'pred_proj':[(3,192)]}
    need(name in expected and shapes==expected[name],'batch1/context3 module ABI before dispatch')


def grasp_optimize(task,slot,b,c):
    t=b.t;z,_=b.root();goal=b.tensor(task['goal'])[None]
    fraction=t.linspace(0,1,4,device=b.device)[1:-1][None,:,None]
    v=(z[:,-1,None]+fraction*(goal[:,None]-z[:,-1,None])).detach().requires_grad_(True)
    x=t.zeros(30,device=b.device,dtype=t.float64,requires_grad=True)
    t.manual_seed(grasp.seed(slot));t.cuda.manual_seed_all(grasp.seed(slot))
    def adam(joint):
        return t.optim.Adam([{'params':[v],'lr':.1},{'params':[x],'lr':.005}] if joint else [x],
            lr=.005 if joint else .05,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=False,fused=False)
    opt=adam(True);sopt=None;snapshot=np.zeros(30,np.float64);export=None;failure=None
    b.event(dict(kind='initialized',seed=grasp.seed(slot),fraction=b.snap(fraction),x=b.snap(x),v=b.snap(v)))
    try:
        for phase,k,j in phases():
            c.unit()
            if phase=='sync' and j==0:sopt=adam(False)
            current=opt if phase=='joint' else sopt;current.zero_grad(set_to_none=True)
            b.event(dict(kind='optimization_attempt',phase=phase,iteration=k,sync_step=j,x=b.snap(x),v=b.snap(v)))
            if phase=='joint':loss,dyn,dense=b.soft(v,x)
            else:
                terminal,_=b.serial(x,'sync');loss=(terminal-goal).square().mean();dyn=dense=None
            b.event(dict(kind='backward_attempt',phase=phase,iteration=k,sync_step=j,loss=b.snap(loss),
                dyn=None if dyn is None else b.snap(dyn),dense=None if dense is None else b.snap(dense)))
            need(bool(t.isfinite(loss)),'finite objective');loss.backward();b.check()
            need(x.grad is not None and bool(t.isfinite(x.grad).all()) and
                (phase!='joint' or v.grad is not None and bool(t.isfinite(v.grad).all())),'finite gradients')
            b.event(dict(kind='backward_return',gx=b.snap(x.grad),gv=b.snap(v.grad) if phase=='joint' else None))
            bx=grasp.optimizer_snapshot(current,x,b.snap);bv=grasp.optimizer_snapshot(current,v,b.snap) if phase=='joint' else None
            b.event(dict(kind='optimizer_attempt',phase=phase,iteration=k,sync_step=j))
            current.step();b.check();proposal=b.snap(x)
            with t.no_grad():x.copy_(b.tensor(grasp.project(x.detach().cpu().numpy()),t.float64))
            b.event(dict(kind='optimizer_return',x_before=bx,v_before=bv,x_after=grasp.optimizer_snapshot(current,x,b.snap),
                v_after=grasp.optimizer_snapshot(current,v,b.snap) if phase=='joint' else None,proposal=proposal,x=b.snap(x),v=b.snap(v)))
            if phase=='joint':
                b.event(dict(kind='noise_attempt'))
                with t.no_grad():noise=.005*t.randn_like(v);v.add_(noise)
                b.event(dict(kind='noise',noise=b.snap(noise),v=b.snap(v)))
            if phase=='sync' and j==9:
                x=x.detach().clone().requires_grad_(True);opt=adam(True)
                b.event(dict(kind='joint_optimizer_reset',after=k+1,x=b.snap(x),v=b.snap(v)))
            saved=b.snap(x)
            if c.finish(saved):snapshot=grasp.unpack(saved,(30,),np.float64).copy()
    except Stop:
        # Incomplete update remains in the trace; no new dispatch after wall.
        pass
    except (ValueError,RuntimeError) as exc:failure=exc
    if failure is None and not c.a.in_unit:
        try:
            c.unit(replay=True)
            b.event(dict(kind='actions_frozen',x=grasp.pack(snapshot),commands=grasp.pack(grasp.chart(snapshot).reshape(15,2))))
            with t.no_grad():
                sx=b.tensor(snapshot,t.float64);_,whole=b.serial(sx,'whole');_,seg=b.serial(sx,'segmented',True)
            equal(whole,seg,'complete three fresh replay contexts')
            witness=dict(logits=snapshot.tolist(),contexts=whole)
            accepted=c.a.last-c.a.start<SEARCH
            b.event(dict(kind='export_complete',witness_sha256=old.digest(witness),accepted=accepted))
            if accepted:export=witness
        except Stop:pass
        except (ValueError,RuntimeError) as exc:failure=exc
    return dict(export=export,error=None if failure is None else dict(type=type(failure).__name__,message=str(failure))),failure


def replay_g_account(events):
    need(events and set(events[0])=={'kind','now'} and events[0]['kind']=='g_start','grasp clock start')
    a=GAccount(events[0]['now']);native=[];commits=[];admissions=[];error=None
    for i,e in enumerate(events[1:]):
        kind=e['kind'];now=e['now']
        if kind=='g_admission':
            need(set(e)=={'kind','now','replay','allowed'} and type(e['replay']) is bool,'exact admission')
            equal(e['allowed'],a.admit_unit(now,e['replay']),'recompute full-unit reservation/deadline');admissions.append(e)
        elif kind=='g_native':
            need(set(e)=={'kind','now','allowed','payload'},'exact native envelope')
            equal(e['allowed'],a.event(e['payload'],now),'actual predispatch admission');native.append(e)
        elif kind=='g_commit':
            need(set(e)=={'kind','now','accepted','x'},'exact snapshot commit')
            equal(e['accepted'],a.finish(now),'deadline excludes late update');grasp.unpack(e['x'],(30,),np.float64);commits.append(e)
        elif kind=='g_memory':
            need(set(e)=={'kind','now','allocated','reserved','allowed'},'exact memory event')
            equal(e['allowed'],a.memory(now,e['allocated'],e['reserved']),'allocator bound')
        elif kind=='g_error':
            need(error is None and set(e)=={'kind','now','type','message'},'single real execution failure')
            a.tick(now);a.stop='algorithm_failure';error=dict(type=e['type'],message=e['message'])
        elif kind=='g_seal':
            need(i==len(events)-2 and set(e)=={'kind','now','summary'},'unique last seal')
            a.tick(now)
            if a.stop is None:a.stop='natural_completion'
            a.ended=now;equal(e['summary'],a.summary(),'full grasp ledger rebuilt')
        else:raise ValueError('unknown grasp envelope '+str(kind))
    need(a.ended is not None,'sealed grasp accounting');return a.summary(),native,commits,admissions,error


class PrefixEnd(Exception):
    """An actual captured prefix ends, not an invented neural return."""


class GCursor:
    def __init__(self,events):self.events=events;self.i=0
    def get(self,kind):
        if self.i==len(self.events):raise PrefixEnd()
        e=self.events[self.i];self.i+=1
        need(e['payload']['kind']==kind,'exact scheduled grasp event '+kind)
        if not e['allowed']:raise PrefixEnd()
        return e['payload']
    def expect(self,e):equal(self.get(e['kind']),e,'complete grasp event fields')


def rounding(actual,expected,ops=256):
    # Saved scalar/reduction/Adam arithmetic only, never neural-output tolerance.
    a=np.asarray(actual);b=np.asarray(expected);eps=np.finfo(b.dtype).eps
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and
        np.all(abs(a-b)<=ops*eps*(abs(b)+1e-20)),'saved arithmetic bound')


def check_adam(before,after,param,grad,proposal,lr):
    need(set(before)==set(after)=={'step','m','q'} and type(after['step']) is int and after['step']==before['step']+1,'Adam chronology')
    m=grasp.unpack(before['m'],param.shape,param.dtype);q=grasp.unpack(before['q'],param.shape,param.dtype)
    am=grasp.unpack(after['m'],param.shape,param.dtype);aq=grasp.unpack(after['q'],param.shape,param.dtype)
    rounding(am,adam_reference.native_lerp(m,grad),64);rounding(aq,.999*q+(1-.999)*grad*grad,64)
    n=after['step'];update=lr/(1-.9**n)*am/(np.sqrt(aq)/np.sqrt(1-.999**n)+1e-8)
    need(np.isfinite(proposal).all() and np.all(abs(proposal-(param-update))<=128*np.finfo(param.dtype).eps*(abs(param)+abs(update)+1e-20)),'saved Adam proposal')


def validate_grasp(task,slot,events,result):
    """Own full schedule/partial consumer; never calls old GRASP consume/validate_root."""
    account,native,commits,admissions,error=replay_g_account(events)
    c=GCursor(native);b=grasp.Saved(task,c)
    u=lambda d,shape,dtype=np.float32:grasp.unpack(d,shape,np.dtype(dtype))
    x=np.zeros(30,np.float64);snapshot=x.copy();mx=mv=None
    goal=np.asarray(task['goal'],np.float32)[None];root=task['root_context'];ci=ai=0;export=None;reached=False
    try:
        e=c.get('initialized');need(set(e)=={'kind','seed','fraction','x','v'} and e['seed']==grasp.seed(slot),'fixed initialization')
        f=u(e['fraction'],(1,2,1));rounding(f,np.asarray([1/3,2/3],np.float32).reshape(1,2,1),8)
        equal(e['x'],grasp.pack(x),'zero logits');v=u(e['v'],(1,2,192))
        need(np.array_equal(v,np.asarray(root['latent_history'],np.float32)[-1,None,None]+f*(goal[:,None]-np.asarray(root['latent_history'],np.float32)[-1,None,None])),'actual fraction init')
        mx=adam_reference.initial_memory(x);mv=adam_reference.initial_memory(v)
        for phase,k,j in phases():
            need(ai<len(admissions) and admissions[ai]['replay'] is False,'fixed optimization admission sequence')
            admission=admissions[ai];ai+=1
            if not admission['allowed']:break
            if phase=='sync' and j==0:mx=adam_reference.initial_memory(x)
            c.expect(dict(kind='optimization_attempt',phase=phase,iteration=k,sync_step=j,x=grasp.pack(x),v=grasp.pack(v)))
            preds=b.soft(v,x) if phase=='joint' else [b.serial(x,'sync')[0]]
            loss=c.get('backward_attempt');need(set(loss)=={'kind','phase','iteration','sync_step','loss','dyn','dense'} and
                (loss['phase'],loss['iteration'],loss['sync_step'])==(phase,k,j),'scalar loss request')
            mse=lambda a,b:np.mean((a-b)**2,dtype=np.float32)
            if phase=='joint':
                dyn=np.float32(0);dense=np.float32(0)
                for kk,pred in enumerate(preds):
                    dyn=np.float32(dyn+mse(pred,v[:,kk] if kk<2 else goal));dense=np.float32(dense+mse(pred,goal))
                rounding(u(loss['dyn'],()),dyn,256);rounding(u(loss['dense'],()),dense,256);rounding(u(loss['loss'],()),np.float32(dyn+dense),512)
            else:
                need(loss['dyn'] is loss['dense'] is None,'sync terminal objective');rounding(u(loss['loss'],()),mse(preds[-1],goal),256)
            rawgrad=c.get('chart_backward');need(set(rawgrad)=={'kind','raw_gradient'},'one chart gradient')
            gr=u(rawgrad['raw_gradient'],(3,5,2));grad=c.get('backward_return');need(set(grad)=={'kind','gx','gv'},'full returned gradients')
            gx=u(grad['gx'],(30,),np.float64)
            need(np.array_equal(gx,gr.reshape(30).astype(np.float64)*(1-np.tanh(x)**2)),'exact single chart derivative')
            gv=None
            if phase=='joint':
                gv=u(grad['gv'],(1,2,192));rounding(gv,np.stack([2/192*(v[:,kk]-preds[kk]) for kk in range(2)],1),32)
            else:need(grad['gv'] is None,'no virtual sync update')
            c.expect(dict(kind='optimizer_attempt',phase=phase,iteration=k,sync_step=j));op=c.get('optimizer_return')
            need(set(op)=={'kind','x_before','v_before','x_after','v_after','proposal','x','v'},'complete optimizer evidence')
            equal(op['x_before'],mx,'actual action Adam memory');proposal=u(op['proposal'],(30,),np.float64)
            check_adam(mx,op['x_after'],x,gx,proposal,.005 if phase=='joint' else .05);mx=op['x_after']
            xx=u(op['x'],(30,),np.float64);need(np.array_equal(xx,grasp.project(proposal)),'exact global30 projection')
            vv=u(op['v'],(1,2,192))
            if phase=='joint':equal(op['v_before'],mv,'actual virtual memory');check_adam(mv,op['v_after'],v,gv,vv,.1);mv=op['v_after']
            else:need(op['v_before'] is op['v_after'] is None and np.array_equal(vv,v),'virtual unchanged in sync')
            x,v=xx,vv
            if phase=='joint':
                c.expect(dict(kind='noise_attempt'));e=c.get('noise');need(set(e)=={'kind','noise','v'},'actual saved noise')
                nn=u(e['noise'],(1,2,192));nv=u(e['v'],(1,2,192));need(np.array_equal(nv,v+nn),'actual noise addition');v=nv
            if phase=='sync' and j==9:
                c.expect(dict(kind='joint_optimizer_reset',after=k+1,x=grasp.pack(x),v=grasp.pack(v)))
                mx=adam_reference.initial_memory(x);mv=adam_reference.initial_memory(v)
            need(ci<len(commits),'complete unit requires commit');equal(commits[ci]['x'],grasp.pack(x),'commit exactly complete update')
            if commits[ci]['accepted']:snapshot=x.copy()
            ci+=1
        if ai<len(admissions):
            need(admissions[ai]['replay'] is True,'final only replay admission');ok=admissions[ai]['allowed'];ai+=1
            if ok:
                c.expect(dict(kind='actions_frozen',x=grasp.pack(snapshot),commands=grasp.pack(grasp.chart(snapshot).reshape(15,2))))
                _,whole=b.serial(snapshot,'whole');_,seg=b.serial(snapshot,'segmented',True);equal(whole,seg,'all three literal replay boundaries')
                witness=dict(logits=snapshot.tolist(),contexts=whole)
                e=c.get('export_complete')
                accepted=native[c.i-2]['now']-account['start']<SEARCH
                cexpected=dict(kind='export_complete',witness_sha256=old.digest(witness),accepted=accepted)
                equal(e,cexpected,'complete replay and wall admission')
                if accepted:export=witness
        reached=True
    except PrefixEnd:
        need(account['stop'] in ('wall_cap','resource_cap','algorithm_failure'),'prefix end requires actual refusal/failure')
    need(c.i==len(native) and ci==len(commits) and ai==len(admissions),'all native events/commits/admissions consumed, no extras')
    if not reached:need(export is None,'incomplete algorithm cannot publish')
    equal(result,dict(export=export,error=error),'full grasp result reconstructed')
    if export is not None:need(account['completed']['packet']==account['attempted']['packet'] and not account['pending'],'full replay returned')
    return account


def resource_vector(arm,account,events):
    """Actual API != packet admission on interrupted spectral forwards."""
    if arm=='grasp':
        attempt=deepcopy(account['attempted']);complete=deepcopy(account['completed'])
        for d in (attempt,complete):
            d.update(Jacobian=0,mathematical_VJP_requested=0,Jacobian_elements_requested=0,root_SVD=0,root_ridge=0)
        # Forward before backward refusal is still paid; correlate returns
        # with the actual packet labels, not scalar backward requests.
        starts={e['payload']['index']:e['payload']['label'] for e in events
            if e['kind']=='g_native' and e['allowed'] and e['payload']['kind']=='forward_attempt'}
        attempt['AD_primal_packet_steps']=sum(not label.startswith(('whole/','segmented/')) for label in starts.values())
        complete['AD_primal_packet_steps']=sum(not starts[e['payload']['index']].startswith(('whole/','segmented/'))
            for e in events if e['kind']=='g_native' and e['allowed'] and e['payload']['kind']=='forward_return')
    else:
        native=[e['payload'] for e in events if e['kind']=='native']
        forward_tags={e['request']['index']:e['request']['tag'] for e in native if e['kind']=='forward_intent'}
        attempt={};complete={}
        for d,success in ((attempt,False),(complete,True)):
            source=account['completed' if success else 'attempted']
            d.update(packet=source['predict'],predict_API=sum(e['kind']==('predict_API_return' if success else 'predict_API_intent') for e in native),
                predict_samples=sum(e['kind']==('module_return' if success else 'module_intent') and
                    (e['entry']['resource'] if success else e.get('resource'))=='predictor' for e in native),
                scalar_backward=0,Adam_step=0,virtual_noise=0,
                Jacobian=source['jacobian'],mathematical_VJP_requested=source['mathematical_VJP'],
                Jacobian_elements_requested=source['jacobian_elements'],AD_primal_packet_steps_requested=source['AD_rollout_steps'],
                AD_primal_packet_steps=sum('/forward/' in (forward_tags[e['index']] if success else e['request']['tag'])
                    for e in native if e['kind']==('forward_return' if success else 'forward_intent')),
                root_SVD=source['root_SVD'],root_ridge=source['root_ridge'])
            for name in ('action_encoder','predictor','pred_proj'):d[name]=source[name]
    for d in (attempt,complete):
        d.update(predictor_context_tokens=3*d['predict_samples'],pred_proj_rows=3*d['pred_proj'],encoder=0,projector=0,physics=0,training=0)
        need(d['predict_API']<=1902 and d['predict_samples']<=1902,'actual separate common API/sample caps')
    return dict(attempted=attempt,completed=complete,engine_internal_backward_calls=None,independent_JVP_calls=None,total_FLOPs=None)


def run_cuda(prepared,expected,output,authorization):
    need(authorization==AUTH,'MAIN explicit authority before any I/O or torch import')
    start=time.monotonic();p=verify(prepared,expected);verified_at=time.monotonic();store=Store(output,start)
    rows=[];errors=[];before=after=host=None;model=None;phases_time={};guard_before=guard_after=None;stage_peak=None
    persistence=0.
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        begun=time.monotonic();phases_time['host_setup_seconds']=begun-verified_at
        model,before=old.loader.loaded_model(p,torch,device,store,expected)
        torch.cuda.synchronize(device);phases_time['loader_seconds']=time.monotonic()-begun
        tasks=join_tasks(p['slots'])
        def clock():torch.cuda.synchronize(device);return time.monotonic()
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
            guard_before=grasp.precision(torch)
            for i,(root,arm) in enumerate(ORDER):
                slot=p['slots'][root];row=dict(index=i,root_slot=root,arm=arm);rows.append(row)
                emit=lambda e,i=i:store.event(dict(kind='arm_event',index=i,payload=e))
                if arm=='grasp':
                    c=GControl(clock,emit);b=GNative(model,device,tasks[root],c.event,store)
                    failure=None
                    try:
                        with b.observe():result,failure=grasp_optimize(tasks[root],root,b,c)
                    except (ValueError,RuntimeError) as exc:
                        failure=exc;result=dict(export=None,error=dict(type=type(exc).__name__,message=str(exc)))
                    account=c.seal(failure)
                    row.update(result=result,accounting=account)
                else:
                    row.update(forwards=[],AD=[]);c=wall.Control(slot,arm,clock,emit)
                    b=wall.HostBackend(model,device,torch,store,row,emit,c)
                    try:
                        with b.observe():row['result']=wall.drive(slot,arm,b,c)
                    except (ValueError,RuntimeError) as exc:
                        b.unwind();c.fail(exc);row['result']=dict(aborted_unit='outer',export=None)
                    c.call('seal');row['accounting']=c.call('report')
                ended=c.a.ended if arm=='grasp' else c.ledger.ended
                store.put(f'arm{i}.json.gz',row);store.event(dict(kind='arm_sealed',index=i,row_sha256=old.digest(row)))
                persistence+=time.monotonic()-ended
            guard_after=grasp.precision(torch)
        stage_peak=dict(allocated=torch.cuda.max_memory_allocated(device),reserved=torch.cuda.max_memory_reserved(device))
        need(stage_peak['allocated']<=4*1024**3 and stage_peak['reserved']<=6*1024**3,'whole-stage actual CUDA peak caps')
    except BaseException as exc:errors.append(old.loader.enc.error('execution',exc))
    finally:
        begun=time.monotonic()
        if model is not None:
            try:old.loader.enc.frozen(model);after=old.loader.enc.tensor_hash(model)
            except BaseException as exc:errors.append(old.loader.enc.error('model_final',exc))
        try:old.parent.check_files(p)
        except BaseException as exc:errors.append(old.loader.enc.error('source_final',exc))
        for row in rows:
            if not (store.output/f'arm{row["index"]}.json.gz').exists():store.put(f'arm{row["index"]}.json.gz',row,final=True)
        store.close();phases_time['final_identity_seconds']=time.monotonic()-begun
        report=dict(schema=NAME+'.report',prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=contract(),
            rows=[dict(index=r['index'],root_slot=r['root_slot'],arm=r['arm'],file=f'arm{r["index"]}.json.gz') for r in rows],
            fixed_denominator=6,errors=errors,status='records_sealed' if len(rows)==6 and not errors else 'failed_partial',
            model_before=before,model_after=after,host=host,precision_before=guard_before,precision_after=guard_after,
            stage_cuda_peak_bytes=stage_peak,
            timings=dict(phases_time,total_seconds=time.monotonic()-start,initial_prepare_verify_seconds=verified_at-start,
                post_search_persistence_seconds=persistence),
            artifact_sha256={f.name:old.sha(f) for f in store.output.iterdir()})
        store.put('report.json',report,final=True)
    return dict(report=str(store.output/'report.json'),sha256=old.sha(store.output/'report.json'),errors=errors)


def journal(output):
    """Bounded streaming single-gzip chain. No giant whole-journal allocation."""
    previous='0'*64
    for i,line in enumerate(grasp.journal_lines(Path(output)/'events.jsonl.gz')):
        e=grasp.inputs.parse(line)
        need(set(e)=={'index','previous_sha256','payload','event_sha256'} and type(e['index']) is int and
            e['index']==i and e['previous_sha256']==previous,'journal ordinal/chain')
        equal(e['event_sha256'],old.digest({k:v for k,v in e.items() if k!='event_sha256'}),'journal SHA')
        need(old.raw(e)==line+b'\n','canonical journal');previous=e['event_sha256'];yield e['payload']


def consume(output,expected):
    output=Path(output).absolute();need(old.sha(output/'report.json')==expected,'actual external report SHA')
    r=old.parent.overlay.read(output/'report.json',8*1024**2);p=verify(r['prepared'],r['prepare_sha256'])
    need(r['schema']==NAME+'.report' and r['status']=='records_sealed' and r['errors']==[] and r['fixed_denominator']==6,'all six engineering records')
    equal(r['contract'],contract(),'contract unchanged');need(r['model_before']==r['model_after']==p['expected_model_hash'],'frozen model exact')
    timing=r['timings']
    need(set(timing)=={'loader_seconds','final_identity_seconds','total_seconds','initial_prepare_verify_seconds','host_setup_seconds','post_search_persistence_seconds'} and
        all(type(v) in (int,float) and np.isfinite(v) and v>=0 for v in timing.values()) and
        timing['total_seconds']<OUTER and sum(v for k,v in timing.items() if k!='total_seconds')<=timing['total_seconds'],'finite explicit time scopes')
    guard=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,MATH=True,FLASH=False,MEM_EFFICIENT=False,CUDNN=False)
    equal(r['precision_before'],guard,'entry precision');equal(r['precision_after'],guard,'exit precision')
    peak=r['stage_cuda_peak_bytes']
    need(set(peak)=={'allocated','reserved'} and type(peak['allocated']) is type(peak['reserved']) is int and
        0<=peak['allocated']<=peak['reserved'] and peak['allocated']<=4*1024**3 and peak['reserved']<=6*1024**3,'whole-stage allocator peaks')
    names=set(r['artifact_sha256']);need({f.name for f in output.iterdir()}==names|{'report.json'} and sum(f.stat().st_size for f in output.iterdir())<=CAP,'exact output census/cap')
    for n,h in r['artifact_sha256'].items():
        need(Path(n).name==n and not (output/n).is_symlink() and old.sha(output/n)==h,'all artifact hashes')
    equal(old.parent.overlay.read(output/'binding.json'),dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution binding')
    loader=old.parent.overlay.read(output/'loader.json');old.loader.validate_loader(loader,p,r['prepare_sha256'])
    events=iter(journal(output))
    def accept(e):
        try:got=next(events)
        except StopIteration:raise ValueError('missing journal event') from None
        equal(got,e,'full journal chronology')
    for f in loader['files']:
        accept(dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']));accept(dict(kind='consumed_buffer_frozen',receipt=f))
    plans=[];accounts=[];resources=[];used={'loader.json','binding.json','events.jsonl.gz'};tasks=join_tasks(p['slots']);previous_end=None
    need(len(r['rows'])==6,'fixed six denominator')
    for i,(root,arm) in enumerate(ORDER):
        meta=dict(index=i,root_slot=root,arm=arm,file=f'arm{i}.json.gz');equal(r['rows'][i],meta,'fixed serial order')
        used.add(meta['file']);row=old.parent.overlay.read(output/meta['file'],1024*1024**2)
        need(row['index']==i and row['root_slot']==root and row['arm']==arm,'row identity');es=[]
        for e in events:
            if e['kind']=='arm_event':
                need(set(e)=={'kind','index','payload'} and e['index']==i,'own arm scope');es.append(e['payload'])
            else:
                equal(e,dict(kind='arm_sealed',index=i,row_sha256=old.digest(row)),'unique arm seal');break
        else:raise ValueError('missing arm seal')
        def load(d):
            n=d['file'];ordinal=len([x for x in used if x.startswith(f'root{i}_J')])
            need(n==f'root{i}_J{ordinal:03}.npy' and old.sha(output/n)==d['sha256'],'exact ordered J path/SHA')
            stream=io.BytesIO((output/n).read_bytes());a=np.load(stream,allow_pickle=False)
            need(stream.tell()==len(stream.getvalue()) and a.dtype==np.float32 and list(a.shape)==d['shape'] and
                d['dtype']=='float32' and np.isfinite(a).all(),'full finite J bytes');used.add(n);return a
        if arm=='grasp':
            account=validate_grasp(tasks[root],root,es,row['result']);equal(account,row['accounting'],'grasp actual accounting')
            need(account['peak_allocated_bytes'] is not None,'actual grasp allocator samples')
            begin,end=account['start'],account['end']
        else:
            account=wall.validate_arm(row,es,p['slots'][root],load)
            begin=es[0]['now'];end=begin+account['elapsed_seconds']
        need(previous_end is None or begin>=previous_end,'nonoverlapping synchronized search windows');previous_end=end
        witness=row['result']['export'];plan=None
        if witness is not None:
            slot=p['slots'][root];plan=dict(schema=NAME+'.plan',root_slot=root,arm=arm,task_id=slot['task_id'],
                task_file=slot['task_file'],task_sha256=slot['task_sha256'],root_context=slot['packet']['root_context'],goal=slot['packet']['goal'],
                warm_raw10=slot['warm_raw10'],commands=old.chart(witness['logits']).reshape(15,2).tolist(),
                raw_float32_bits=old.chart(witness['logits']).tobytes().hex(),contexts=witness['contexts'],model_hash=p['expected_model_hash'],
                model_Q=old.objective(witness['contexts'][-1],slot['packet']['goal']),report_sha256=expected,physical_release=False)
            plan['selection_digest']=old.digest(plan)
        plans.append(plan);accounts.append({k:v for k,v in account.items() if k!='events'});resources.append(resource_vector(arm,account,es))
    try:next(events)
    except StopIteration:pass
    else:raise ValueError('unknown terminal journal event')
    need(used==names,'no unknown artifacts');old.parent.check_files(p)
    for n,h in r['artifact_sha256'].items():need(old.sha(output/n)==h,'final saved hash')
    need(old.sha(output/'report.json')==expected,'final report hash')
    return dict(schema=NAME+'.consume',status='all6_common_cap_records_verified_not_admitted',plans=plans,accounts=accounts,resources=resources,
        report_sha256=expected,fixed_denominator=6,contract=contract(),new_neural_calls=0,new_physics_calls=0,
        inherited_encoding_cohort=dict(encoder=16,projector=16,scope='original four tasks once; off-task allocation unknown',off_allocation=None),
        bindings=dict(p['bindings_sha256'],**{str(output/n):h for n,h in r['artifact_sha256'].items()},**{str(output/'report.json'):expected}))


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for name in ('protocol-prepared','protocol-sha','prepared','prepare-sha','output','authorization','report-sha','receipt'):
        q.add_argument('--'+name)
    a=q.parse_args(argv)
    if a.command=='prepare':v=write_new(a.output,build(a.protocol_prepared,a.protocol_sha))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v=dict(verified=True,contract=contract())
    elif a.command=='run':v=run_cuda(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:
        v=consume(a.output,a.report_sha)
        if a.receipt:v=write_new(a.receipt,v,64*1024**2)
    print(old.raw(v).decode(),end='');return 0


if __name__=='__main__':raise SystemExit(main())
