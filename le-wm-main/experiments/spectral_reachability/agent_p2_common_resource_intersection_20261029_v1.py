"""Outcome-blind 27-axis conjunctive budget kernel, not a production runner.

No numpy/torch/model/physics imports, no results reader. All six old method/root
slots remain. Accounting is NOT forward/gradient/source fidelity certification.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

NAME='agent_p2_common_resource_intersection_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
AXES=('predict_API','predictor_samples','AE_attempts','AE_dispatched','AE_returned','AE_captured','AE_rows','AE_tokens',
 'predictor_attempts','predictor_dispatched','predictor_returned','predictor_captured','predictor_context_tokens',
 'lowproj_attempts','lowproj_dispatched','lowproj_returned','lowproj_captured','lowproj_rows',
 'candidate_cost_calls','candidate_trajectories','mean_paid_replays','replay_trajectories','AD_primal','J','VJP','backward','FLOPs')
ORDER=((0,'joint'),(0,'composition'),(0,'grasp'),(1,'grasp'),(1,'composition'),(1,'joint'))
WALL=450.0
SIDE_CAPS=dict(root_SVD=2,root_ridge=1,Adam_step=632,virtual_noise=532,Jacobian_elements_requested=489600)
MODULE_SHAPES=dict(AE=[[1,3,10]],predictor=[[1,3,192],[1,3,192]],lowproj=[[3,192]])
PINS={
 'experiments/spectral_reachability/agent_p2_common_resource_grasp_cuda_20261029_v1.py':'ea763d96eee1040652a3ea02badcb5e8e37b5781530da721f73d2a63d9cb147d',
 'experiments/spectral_reachability/agent_p2_compute_matched_protocol_20261029_v2.py':'f17101d1b4b1d8a2c32bcd0218bda9708a2c992c1747a5cfe863e89fb0fff2ee',
 'experiments/spectral_reachability/agent_p2_six_target_controller_cuda_20261029_v1.py':'075a5c09b8c57051b3e7bc3eb3acac7b824f0c1a983151ec3f8ee10c1c813a5c'}


def need(ok,message):
    if not ok:raise ValueError(message)


def raw(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(raw(v)).hexdigest()
def zeros():return {k:None if k=='FLOPs' else 0 for k in AXES}


def add(*vectors):
    out=zeros()
    for v in vectors:
        need(set(v)==set(AXES) and v['FLOPs'] is None,'exact27 axes; FLOPs unknown')
        for k in AXES[:-1]:
            need(type(v[k]) is int and v[k]>=0,'nonnegative integer counted axis')
            out[k]+=v[k]
    return out


def caps():
    # Old source preregistration bounds, not fitted from actual result totals.
    c=zeros()
    c.update(predict_API=1902,predictor_samples=1902,AE_rows=1902,AE_tokens=5706,
        predictor_context_tokens=5706,lowproj_rows=5706,candidate_cost_calls=1902,
        candidate_trajectories=1902,mean_paid_replays=0,replay_trajectories=2,
        AD_primal=1896,J=85,VJP=16320,backward=632)
    for prefix in ('AE','predictor','lowproj'):
        for phase in ('attempts','dispatched','returned','captured'):c[prefix+'_'+phase]=1902
    return c


def forward(n):
    need(type(n) is int and n>=0,'packet count')
    v=zeros();v.update(predict_API=n,predictor_samples=n,AE_rows=n,AE_tokens=3*n,
        predictor_context_tokens=3*n,lowproj_rows=3*n)
    for prefix in ('AE','predictor','lowproj'):
        for phase in ('attempts','dispatched','returned','captured'):v[prefix+'_'+phase]=n
    return v


@lru_cache(maxsize=3)
def schedule(arm):
    need(arm in ('joint','composition','grasp'),'all three fixed methods retained')
    if arm=='grasp':
        labels=[]
        for i in range(532):
            labels.append(f'joint:{i}')
            if (i+1)%50==0:labels.extend(f'sync:{i}:{j}' for j in range(10))
        return tuple(labels+['replay'])
    local=([f'joint:{p}:{i}' for p in (0,1) for i in range(42)] if arm=='joint'
        else [f'{h}:{i}' for i in range(42) for h in ('H2','H1')])
    return tuple(['nominal','bank']+[f'seed:{i}' for i in range(42)]+local+['replay'])


def envelope(arm,label):
    need(label in schedule(arm),'fixed source schedule label')
    if label=='replay':v=forward(6);v['replay_trajectories']=2;return v
    if arm=='grasp':v=forward(3);v.update(AD_primal=3,backward=1,candidate_cost_calls=1,candidate_trajectories=1);return v
    n=3 if label in ('nominal','bank') or label.startswith('seed:') else 21 if label.startswith('joint:') else 15 if label.startswith('H2:') else 7
    v=forward(n)
    if label=='bank' or label.startswith(('joint:','H2:','H1:')):
        h=2 if label.startswith('H2:') else 1 if label.startswith('H1:') else 3
        v.update(J=1,VJP=192,AD_primal=h)
    # These are conservative score dispatch envelopes, not equality targets.
    # Lifted GRASP trajectory scope must be disclosed, never a root-full path.
    v.update(candidate_cost_calls=n,candidate_trajectories=n)
    return v


def source_check():
    for p,h in PINS.items():
        f=ROOT/p;need(f.is_file() and f.stat().st_size<=2*1024**2,'bounded named source')
        need(hashlib.sha256(f.read_bytes()).hexdigest()==h,'frozen source pin '+p)
    return deepcopy(PINS)


def protocol():
    return dict(schema=NAME+'.protocol',axes=list(AXES),order=[list(x) for x in ORDER],caps=caps(),
        search_wall_seconds=WALL,reserve=envelope('joint','replay'),
        stop='first coordinate refusal, elapsed>=450, operation failure or finite schedule exhaustion',
        intersection='AND of coordinate upper bounds; not min of methods usage or scalarized FLOPs',
        actual_cost_equality=False,equal_FLOPs=False,total_FLOPs=None,engine_backward_calls=None,
        independent_JVP_calls=None,scientific_admission=False,fresh=False,physical_release=False,
        production_runner_implemented=False,source_semantics_independently_verified=False)


class Refused(Exception):
    """No new paid operation may follow refusal, except reserved replay at resource stop."""


class Journal:
    """Exclusive bounded durable transport; one file, no model arrays or outcomes."""
    def __init__(self,path,*,cap_bytes):
        self.path=Path(path);need(self.path.is_absolute() and self.path.parent.is_dir()
            and not self.path.exists() and type(cap_bytes) is int and cap_bytes>0,'new bounded journal')
        self.cap=cap_bytes;self.used=0
        with self.path.open('xb') as f:f.flush();os.fsync(f.fileno())
        fd=os.open(self.path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def __call__(self,event):
        b=raw(event)+b'\n';need(self.used+len(b)<=self.cap,'journal hard byte cap')
        with self.path.open('ab') as f:need(f.write(b)==len(b),'full event');f.flush();os.fsync(f.fileno())
        self.used+=len(b)


class Ledger:
    """Inject synchronized clock and durable sink. No implicit production defaults.

    Full-unit admission is conservative; callbacks are charged only when entered.
    Module returned and captured are separate events, never inferred from each other.
    """
    def __init__(self,root,arm,*,clock,sink):
        need(type(root) is int and root in (0,1) and arm in ('joint','composition','grasp'),'fixed six logical slots')
        self.root,self.arm,self.clock,self.sink=root,arm,clock,sink
        self.start=self.last=float(clock());need(math.isfinite(self.start),'synchronized finite clock')
        self.used=zeros();self.unit=None;self.unit_used=None;self.labels=schedule(arm);self.cursor=0
        self.stop=None;self.stack=[];self.operations=[];self.events=[];self.committed=[];self.failed_transport=False
        self.side=dict(root_SVD=0,root_ridge=0,Adam_step=0,virtual_noise=0,Jacobian_elements_requested=0,
            CPU_capture_bytes=0,CPU_capture_seconds=0.0,engine_backward_calls=None,independent_JVP_calls=None)
        self.unit_side=None;self.side_returns={k:0 for k in SIDE_CAPS}
        self._emit('start',root=root,arm=arm,start=self.start,protocol_sha256=digest(protocol()))
    def _emit(self,kind,**fields):
        e=dict(index=len(self.events),previous_sha256=self.events[-1]['sha256'] if self.events else None,kind=kind,**fields)
        e['sha256']=digest(e)
        try:self.sink(deepcopy(e))
        except BaseException:self.failed_transport=True;self.stop='transport_failure';raise
        self.events.append(e)
    def tick(self):
        need(not self.failed_transport,'transport failure cannot resume')
        now=float(self.clock());need(math.isfinite(now) and now>=self.last,'monotone synchronized clock')
        self.last=now
        if now-self.start>=WALL:self.stop='wall_cap'
        return now-self.start
    def admit(self,label):
        elapsed=self.tick();need(self.unit is None and not self.stack,'complete unit boundary')
        need(self.cursor<len(self.labels),'schedule already exhausted')
        replay=label=='replay'
        need(label==self.labels[self.cursor] or (replay and self.stop is not None),'no skipped/reselected optimizer units')
        reserve=zeros() if replay else envelope(self.arm,'replay')
        cost=envelope(self.arm,label);total=add(self.used,cost,reserve)
        over=[k for k in AXES[:-1] if total[k]>caps()[k]]
        allowed=elapsed<WALL and (self.stop is None or (self.stop=='resource_cap' and replay)) and not over
        if over and self.stop is None:self.stop='resource_cap'
        self._emit('admission',label=label,elapsed=elapsed,allowed=allowed,over=over,envelope=cost,reserve=reserve)
        if not allowed:return False
        self.unit=label;self.unit_used=zeros();self.unit_side=deepcopy(self.side_returns);return True
    def _charge(self,delta,kind,**fields):
        elapsed=self.tick();need(self.unit is not None,'operation only in admitted unit')
        if self.stop is not None and not (self.stop=='resource_cap' and self.unit=='replay'):raise Refused(self.stop)
        total=add(self.used,delta);unit=add(self.unit_used,delta);limit=envelope(self.arm,self.unit)
        over=[k for k in AXES[:-1] if total[k]>caps()[k] or unit[k]>limit[k]]
        if over:
            self.stop='resource_cap';self._emit('refusal',kind_requested=kind,elapsed=elapsed,over=over);raise Refused('resource_cap')
        self._emit('debit',operation_kind=kind,elapsed=elapsed,delta=delta,fields=fields)
        self.used,self.unit_used=total,unit
    def call(self,kind,callback,**fields):
        """Typed non-module callbacks. No arbitrary caller-supplied resource vector."""
        d=zeros()
        if kind=='predict':
            need(not self.stack or self.stack[-1]['kind'] in ('J','replay'),'predict nesting')
            d['predict_API']=1
            if any(x['kind']=='J' for x in self.stack) or (self.arm=='grasp' and self.unit!='replay'):d['AD_primal']=1
        elif kind=='J':
            need(self.arm!='grasp' and not self.stack and set(fields)=={'horizon','dimension'}
                and (fields['horizon'],fields['dimension']) in ((1,10),(2,20),(3,30)),'explicit source J shape')
            expected_h=2 if self.unit.startswith('H2:') else 1 if self.unit.startswith('H1:') else 3
            need((fields['horizon'],fields['dimension'])==(expected_h,10*expected_h),'J shape matches source unit')
            need(self.side['Jacobian_elements_requested']+192*fields['dimension']<=SIDE_CAPS['Jacobian_elements_requested'],'J element cap')
            d.update(J=1,VJP=192)
        elif kind=='backward':need(self.arm=='grasp' and not self.stack and not fields,'scalar backward, not fullJ');d['backward']=1
        elif kind=='score':
            need(not self.stack and set(fields)=={'trajectories','scope'} and type(fields['trajectories']) is int
                and fields['trajectories']>=0 and fields['scope'] in ('root_full_path','lifted_loss','serial_terminal'), 'explicit score scope')
            d.update(candidate_cost_calls=1,candidate_trajectories=fields['trajectories'])
        elif kind=='replay':
            need(self.unit=='replay' and not self.stack and fields in ({'part':'whole'},{'part':'segmented'}),'fixed paid replay trajectories')
            d['replay_trajectories']=1 # None of these three methods returns a CEM mean.
        else:raise ValueError('unrecognized callback kind')
        self._charge(d,kind,**fields)
        if kind=='J':self.side['Jacobian_elements_requested']+=192*fields['dimension']
        op=dict(index=len(self.operations),kind=kind,status='pending',modules=[],fields=fields)
        self.operations.append(op);self.stack.append(op)
        try:
            value=callback()
            if kind=='predict':need(op['modules']==['AE','predictor','lowproj'],'all actual returned/captured modules required')
            elapsed=self.tick();self._emit('return',operation=op['index'],elapsed=elapsed)
            op['status']='returned';return value
        except BaseException as e:
            op['status']='failed';self.stop=self.stop or 'operation_failure'
            if not self.failed_transport:self._emit('failure',operation=op['index'],error=type(e).__name__)
            raise
        finally:
            if self.stack and self.stack[-1] is op:self.stack.pop()
    def module(self,name,callback,capture,*,shapes):
        need(name in ('AE','predictor','lowproj') and self.stack and self.stack[-1]['kind']=='predict','module inside predict')
        need(shapes==MODULE_SHAPES[name],'literal batch1/history3 native tensor ABI')
        parent=self.stack[-1];need(name==('AE','predictor','lowproj')[len(parent['modules'])],'module order, no retry')
        d=zeros();d[name+'_attempts']=1;self._charge(d,'module_attempt',name=name)
        d=zeros();d[name+'_dispatched']=1
        if name=='AE':d.update(AE_rows=1,AE_tokens=3)
        elif name=='predictor':d.update(predictor_samples=1,predictor_context_tokens=3)
        else:d['lowproj_rows']=3
        self._charge(d,'module_dispatch',name=name)
        result=callback() # CUDA callback entered only after durable dispatch debit.
        d=zeros();d[name+'_returned']=1
        # Returns of an already-entered kernel remain recorded even after wall.
        self._record_completion(d,'module_return',name=name)
        elapsed=self.tick()
        if self.stop=='wall_cap':raise Refused('wall_cap: no new CPU capture after deadline')
        self._emit('capture_intent',name=name,elapsed=elapsed)
        begin=float(self.clock());need(math.isfinite(begin) and begin>=self.last,'capture start clock')
        desc=capture(result);end=float(self.clock())
        need(math.isfinite(end) and end>=begin and set(desc)=={'bytes','sha256'} and type(desc['bytes']) is int
            and desc['bytes']>=0 and type(desc['sha256']) is str and len(desc['sha256'])==64,'actual captured bytes/hash')
        self.side['CPU_capture_bytes']+=desc['bytes'];self.side['CPU_capture_seconds']+=end-begin
        d=zeros();d[name+'_captured']=1;self._record_completion(d,'module_capture',name=name,descriptor=desc,capture_seconds=end-begin)
        parent['modules'].append(name);return result
    def auxiliary(self,kind,callback):
        need(kind in ('root_SVD','root_ridge','Adam_step','virtual_noise') and self.unit is not None
            and not self.stack,'typed side callback boundary')
        need((kind in ('root_SVD','root_ridge') and self.arm!='grasp' and self.unit=='bank')
            or (kind=='Adam_step' and self.arm=='grasp' and self.unit!='replay')
            or (kind=='virtual_noise' and self.arm=='grasp' and self.unit.startswith('joint:')),'source side-operation scope')
        elapsed=self.tick()
        if self.stop is not None:raise Refused(self.stop)
        need(self.side[kind]<SIDE_CAPS[kind],'side cap before dispatch')
        self._emit('auxiliary_intent',operation_kind=kind,elapsed=elapsed);self.side[kind]+=1
        try:value=callback()
        except BaseException:
            self.stop='operation_failure';self._emit('auxiliary_failed',operation_kind=kind);raise
        self._emit('auxiliary_return',operation_kind=kind,elapsed=self.tick());self.side_returns[kind]+=1
        return value
    def _record_completion(self,d,kind,**fields):
        elapsed=self.tick();total=add(self.used,d);unit=add(self.unit_used,d)
        need(all(total[k]<=caps()[k] and unit[k]<=envelope(self.arm,self.unit)[k] for k in AXES[:-1]),'completion bound invariant')
        self._emit('debit',operation_kind=kind,elapsed=elapsed,delta=d,fields=fields)
        self.used,self.unit_used=total,unit
    def commit(self):
        elapsed=self.tick();need(self.unit is not None and not self.stack,'complete callback boundary')
        accepted=self.stop in (None,'resource_cap') and elapsed<WALL and all(x['status']=='returned' for x in self.operations)
        # Score envelopes are conservative, not artificial equality quotas.
        # All actual forward/AD/replay work must nevertheless be complete.
        required=envelope(self.arm,self.unit)
        work=[k for k in AXES[:-1] if k not in ('candidate_cost_calls','candidate_trajectories')]
        accepted=accepted and all(self.unit_used[k]==required[k] for k in work)
        side_required=dict(root_SVD=2 if self.unit=='bank' else 0,root_ridge=1 if self.unit=='bank' else 0,
            Adam_step=int(self.arm=='grasp' and self.unit!='replay'),
            virtual_noise=int(self.arm=='grasp' and self.unit.startswith('joint:')))
        accepted=accepted and all(self.side_returns[k]-self.unit_side[k]==v for k,v in side_required.items())
        if not accepted and self.stop is None:self.stop='incomplete_unit'
        self._emit('unit_close',label=self.unit,elapsed=elapsed,accepted=accepted,used=self.unit_used)
        if accepted:self.committed.append(self.unit);self.cursor=len(self.labels) if self.unit=='replay' else self.cursor+1
        self.unit=None;self.unit_used=None;self.unit_side=None
        if self.cursor==len(self.labels) and self.stop is None:self.stop='natural_completion'
        return accepted
    def summary(self):
        return dict(root=self.root,arm=self.arm,used=None if self.failed_transport else deepcopy(self.used),
            observed_local_used=deepcopy(self.used),stop=self.stop,elapsed_seconds=self.last-self.start,
            wall_overrun=self.last-self.start>WALL,active_unit=self.unit,committed=list(self.committed),
            operations=deepcopy(self.operations),side=deepcopy(self.side),transport_failed=self.failed_transport,
            accounting_only=True,actual_cost_equality=False,equal_FLOPs=False,scientific_admission=False)


def verify_journal(path,*,cap_bytes):
    p=Path(path);need(p.is_file() and not p.is_symlink() and p.stat().st_size<=cap_bytes,'bounded saved journal')
    total=zeros();events=[];tail=None
    for i,line in enumerate(p.read_bytes().splitlines(keepends=True)):
        e=json.loads(line);need(line==raw(e)+b'\n' and e['index']==i and e['previous_sha256']==tail
            and e['sha256']==digest({k:v for k,v in e.items() if k!='sha256'}),'complete canonical event hashchain')
        if e['kind']=='debit':total=add(total,e['delta'])
        need(all(total[k]<=caps()[k] for k in AXES[:-1]),'all coordinate caps independently summed')
        tail=e['sha256'];events.append(e)
    need(events and events[0]['kind']=='start','start required')
    return dict(events=len(events),used=total,tail_sha256=tail,scope='transport and additive budget only, not algorithm/NN validation')


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('verify-sources','spec'))
    a=q.parse_args(argv)
    print(raw(source_check() if a.command=='verify-sources' else protocol()).decode())


if __name__=='__main__':main()
