"""Wall-capped protocol and RAM accounting ABI, not a CUDA runtime.

Actual elapsed time and consumed resources are responses, not equality gates.
No torch, physics, outcome reading, fitting, or production execution here.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p2_compute_matched_protocol_20261029_v1 as prior

NAME='agent_p2_compute_matched_protocol_20261029_v2'
old=prior.old
ROOT=old.ROOT
require=old.require
WALL=450.
ARMS=('joint','composition')
FIELDS=('predict','action_encoder','predictor','pred_proj','jacobian','mathematical_VJP',
    'jacobian_elements','AD_rollout_steps','root_SVD','root_ridge','encoder','projector','physics','training')
MODULES=('action_encoder','predictor','pred_proj')


def zeros():return dict.fromkeys(FIELDS,0)


def caps():
    c=zeros();c.update(predict=1902,action_encoder=1902,predictor=1902,pred_proj=1902,
        jacobian=85,mathematical_VJP=16320,jacobian_elements=489600,AD_rollout_steps=255,root_SVD=2,root_ridge=1)
    return c


def forward_cost(n):
    c=zeros()
    for k in ('predict',)+MODULES:c[k]=n
    return c


def schedule(arm):
    require(arm in ARMS,'fixed strategy only')
    local=([f'joint:{p}:{i}' for p in (0,1) for i in range(42)] if arm=='joint'
        else [f'{h}:{i}' for i in range(42) for h in ('H2','H1')])
    return ['nominal','bank']+[f'seed:{i}' for i in range(42)]+local+['replay']


AD_SPEC={'bank':(3,30)}
for _a in ARMS:
    for _label in schedule(_a)[44:-1]:
        AD_SPEC[_label]=(3,30) if _label.startswith('joint:') else ((2,20) if _label.startswith('H2:') else (1,10))


def unit_cost(label):
    require(label in set(schedule('joint'))|set(schedule('composition')),'fixed unit only')
    n=(3 if label in ('nominal','bank') or label.startswith('seed:') else
        6 if label=='replay' else 21 if label.startswith('joint:') else 15 if label.startswith('H2:') else 7)
    c=forward_cost(n)
    if label in AD_SPEC:
        h,d=AD_SPEC[label];c.update(jacobian=1,mathematical_VJP=192,jacobian_elements=192*d,AD_rollout_steps=h)
    if label=='bank':c.update(root_SVD=2,root_ridge=1)
    return c


def protocol(authority):
    return dict(schema=NAME+'.prepare',authority=authority,status='wall_capped_protocol_prepared_runtime_not_implemented',
        comparison='wall_capped_not_equal_FLOPs',roots=[0,1],condition='off',known_roots=2,fresh_roots=0,
        root_arm_order=[[0,'joint'],[0,'composition'],[1,'composition'],[1,'joint']],
        wall_cap_seconds_per_root_arm=WALL,resource_caps_per_root_arm=caps(),
        peak_allocated_cap_bytes=4*1024**3,peak_reserved_cap_bytes=6*1024**3,
        schedules={a:schedule(a) for a in ARMS},unit_costs={s:unit_cost(s) for a in ARMS for s in schedule(a)},
        natural_totals={a:{k:sum(unit_cost(s)[k] for s in schedule(a)) for k in FIELDS} for a in ARMS},
        common_scan='all42 original root-bank raw15 seeds stable model scoring, individually paid in each arm',
        joint_rule='two round-major full30 passes, original-root context; frozen v1 local_joint rule; fixed4 alphas and incumbent guard',
        composition_rule='seed-major own prefix5 replay1 then original H2 optimize14 and H1 optimize7; original old.optimize rules; no cross-seed labels',
        nominal_rule='zero full15 first; establish complete fallback before root J/bank',
        reuse='no free old predictions/J or cross-arm cache; original root inputs/weights only',
        anytime='commit only complete full15 original-goal model witness, strict lower mean192 cost; tie keeps earlier completion',
        stop_rule='first deadline/resource refusal or unrecoverable failure; else finite schedule ends naturally; no padding/refunds/retries',
        timing='MAIN synchronizes CUDA before start and at every intent/return/checkpoint; monotonic host wall includes search, CPU SVD, capture and export; loading+strict outside window separately timed',
        deadline='no new dispatch at elapsed>=450; in-flight completion may overrun, report violation and stop; not kernel preemption',
        export='reserve6 forward/module calls for whole+segmented winner replay; no time guarantee; missing replay remains export_missing, no hidden tail',
        wall_is_response=True,elapsed_equality_required=False,resource_equality_required=False,
        total_FLOPs=None,engine_internal_backward_calls=None,equal_FLOPs=False,equal_total_compute=False,
        physical_outcome_access=False,training=False,Qfit=False,padding=False,retries=0,
        P1_admission=False,P2_admission=False,P2_mechanism=False,fresh=False,physical_release=False,
        runtime_implementation_required=True,new_neural_calls=0,new_physics_calls=0)


class Ledger:
    """Injected monotonic clock MUST synchronize CUDA first in MAIN runtime.

    Admission reserves future envelopes, but actual counters charge only entered
    callbacks. A failed intent remains charged forever, never refunded.
    This is accounting, NOT model-forward/source/witness authentication.
    """
    def __init__(self,root,arm,clock,binding=None):
        require(root in (0,1) and type(root) is int and arm in ARMS,'fixed root/arm')
        self.root,self.arm,self.clock=root,arm,clock;self.ended=None
        self.binding=deepcopy(binding);self.start=float(clock());self.last=self.start
        require(math.isfinite(self.start),'finite start clock')
        self.attempted=zeros();self.completed=zeros();self.events=[];self.operations=[]
        self.units=schedule(arm);self.cursor=0;self.unit=None;self.before=None;self.finished=[]
        self.stop=None;self.errors=[];self.incumbent=None;self.goal=None;self.commits=[];self.export_complete=False
        self.peak_allocated=None;self.peak_reserved=None

    def tick(self):
        if self.ended is not None:return self.ended-self.start
        now=float(self.clock());require(math.isfinite(now) and now>=self.last,'finite monotone synchronized clock')
        self.last=now;e=now-self.start
        if e>=WALL and self.stop is None:self.stop='wall_cap'
        return e

    def start_unit(self,label):
        elapsed=self.tick()
        if self.ended is not None:return False
        if self.stop is not None and not (self.stop=='resource_cap' and label=='replay' and elapsed<WALL):return False
        require(self.unit is None,'previous unit must finish')
        if self.cursor==len(self.units):self.stop='natural_completion';return False
        require(label==self.units[self.cursor] or (label=='replay' and self.stop=='resource_cap'),'fixed units, no outcome selection')
        cost=unit_cost(label);reserve=zeros() if label=='replay' else forward_cost(6)
        if any(self.attempted[k]+cost[k]+reserve[k]>caps()[k] for k in FIELDS):
            self.stop='resource_cap';return False
        self.unit=label;self.before=deepcopy(self.attempted)
        self.events.append(dict(kind='unit_start',unit=label,elapsed=elapsed));return True

    def intent(self,kind,*,horizon=None,dimension=None):
        elapsed=self.tick()
        require(self.ended is None,'sealed accounting cannot dispatch')
        require(self.unit is not None and (self.stop is None or (self.stop=='resource_cap' and self.unit=='replay')),'no dispatch after stop')
        require(kind in ('predict',)+MODULES+('jacobian','root_SVD','root_ridge'),'allowed model resource only')
        active=[o for o in self.operations if not o['closed']]
        inc=zeros();inc[kind]=1
        if kind=='jacobian':
            require(not active and AD_SPEC.get(self.unit)==(horizon,dimension),'exact unit AD horizon/dimension')
            inc.update(mathematical_VJP=192,jacobian_elements=192*dimension,AD_rollout_steps=horizon)
        elif kind=='predict':
            require(all(o['kind']=='jacobian' for o in active),'no overlapping predict')
            for ad in active:require(ad['children']<ad['horizon'],'no extra primal in Jacobian')
        elif kind in MODULES:
            require(active and active[-1]['kind']=='predict','module only within predict')
            p=active[-1];require(len(p['modules'])<3 and kind==MODULES[len(p['modules'])],'exact original module order')
            p['modules'].append(kind)
        else:require(not active and self.unit=='bank','root CPU operations only in bank')
        cost=unit_cost(self.unit)
        require(all(self.attempted[k]+inc[k]<=caps()[k] and self.attempted[k]+inc[k]-self.before[k]<=cost[k] for k in FIELDS),'actual callback cap before execution')
        o=dict(index=len(self.operations),kind=kind,charge=inc,closed=False,success=None,
            horizon=horizon,dimension=dimension,children=0,child_returns=0,modules=[])
        if kind=='predict':
            for ad in active:ad['children']+=1
        self.operations.append(o)
        for k in FIELDS:self.attempted[k]+=inc[k]
        self.events.append(dict(kind='intent',operation=o['index'],resource=kind,charge=deepcopy(inc),elapsed=elapsed))
        return o['index']

    def returned(self,index,success=True):
        elapsed=self.tick()
        require(self.ended is None,'sealed accounting cannot accept callbacks')
        require(type(success) is bool and type(index) is int and 0<=index<len(self.operations),'actual return index/flag')
        o=self.operations[index];require(not o['closed'] and not any(not p['closed'] for p in self.operations[index+1:]),'strict nested return chronology')
        if success:
            require(all(p['success'] is True for p in self.operations[index+1:]),'failed child cannot certify successful parent')
            if o['kind']=='predict':require(o['modules']==list(MODULES),'all actual module returns')
            if o['kind']=='jacobian':require(o['children']==o['child_returns']==o['horizon'],'full AD primal only, no partial matrix success')
        o.update(closed=True,success=success)
        if success:
            for k in FIELDS:self.completed[k]+=o['charge'][k]
            if o['kind']=='predict':
                for ad in self.operations[:index]:
                    if ad['kind']=='jacobian' and not ad['closed']:ad['child_returns']+=1
        else:
            self.errors.append(dict(operation=index,resource=o['kind'],elapsed=elapsed))
            if self.stop is None:self.stop='operation_failure'
        self.events.append(dict(kind='return',operation=index,success=success,elapsed=elapsed))

    def finish_unit(self):
        self.tick();require(self.unit is not None and all(o['closed'] for o in self.operations),'all started callbacks closed')
        require(not self.errors and all(self.attempted[k]-self.before[k]==unit_cost(self.unit)[k] for k in FIELDS),'complete exact unit, no padding/partial commit')
        label=self.unit;self.events.append(dict(kind='unit_return',unit=label,elapsed=self.last-self.start))
        self.finished.append(label)
        self.cursor=(len(self.units) if label=='replay' else self.cursor+1);self.unit=None
        if label=='replay':self.export_complete=True
        if self.cursor==len(self.units) and self.stop is None:self.stop='natural_completion'

    def commit(self,witness,goal,tag):
        elapsed=self.tick();require(self.ended is None and self.unit is None and self.cursor>0 and not self.errors,'completed unit witness only')
        require(type(tag) is str and tag and set(witness)=={'logits','contexts'},'model-only witness ABI')
        u=np.asarray(witness['logits'],np.float64);g=old.finite(goal,(192,))
        require(u.shape==(30,) and old.legal([],u) and len(witness['contexts'])==3,'complete legal full15 witness')
        current=None if self.binding is None else self.binding['packet']['root_context']
        for k,c in enumerate(witness['contexts']):
            old.kernel.validate_context(c)
            if current is not None:old.kernel.validate_advance(current,old.chart(u)[k],c)
            current=c
        bits=old.digest(goal)
        if self.binding is not None:old.need_equal(goal,self.binding['packet']['goal'],'original prepared goal')
        require(self.goal is None or self.goal==bits,'same immutable goal throughout search');self.goal=bits
        cost=old.objective(witness['contexts'][-1],goal)
        improved=elapsed<=WALL and (self.incumbent is None or cost<self.incumbent['cost'])
        if improved:
            self.incumbent=dict(tag=tag,cost=cost,witness=deepcopy(witness),raw_float32_bits=old.chart(u).tobytes().hex(),elapsed=elapsed)
        self.commits.append(dict(tag=tag,cost=cost,accepted=improved,late=elapsed>WALL,elapsed=elapsed))
        return improved

    def sample_memory(self,allocated,reserved):
        self.tick();require(self.ended is None and type(allocated) is int and type(reserved) is int
            and 0<=allocated<=reserved,'actual finite CUDA allocator sample')
        self.peak_allocated=max(self.peak_allocated or 0,allocated)
        self.peak_reserved=max(self.peak_reserved or 0,reserved)
        if allocated>4*1024**3 or reserved>6*1024**3:
            if self.stop is None:self.stop='memory_cap'

    def seal(self):
        self.tick();require(self.ended is None and self.stop is not None,'seal only fixed natural/budget/failure stop')
        self.ended=self.last

    def report(self):
        elapsed=self.tick()
        return dict(schema=NAME+'.accounting',root=self.root,arm=self.arm,binding=deepcopy(self.binding),
            elapsed_seconds=elapsed,wall_cap_seconds=WALL,wall_overrun=elapsed>WALL,
            attempted=deepcopy(self.attempted),completed=deepcopy(self.completed),caps=caps(),
            stop=self.stop,errors=deepcopy(self.errors),units_completed=len(self.finished),finished_units=deepcopy(self.finished),
            active_unit=self.unit,pending=[o['index'] for o in self.operations if not o['closed']],
            incumbent=deepcopy(self.incumbent),commits=deepcopy(self.commits),export_complete=self.export_complete,
            export_missing=not self.export_complete,events=deepcopy(self.events),
            sealed=self.ended is not None,peak_allocated_bytes=self.peak_allocated,peak_reserved_bytes=self.peak_reserved,
            accounting_only=True,model_witness_authenticated=False,total_FLOPs=None,engine_internal_backward_calls=None,
            equal_FLOPs=False,P2_admission=False)


def compare_reports(reports):
    require(len(reports)==2 and {r['arm'] for r in reports}==set(ARMS) and len({r['root'] for r in reports})==1,'paired fixed-root arms')
    for r in reports:
        require(type(r['elapsed_seconds']) in (int,float) and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>=0,'finite actual elapsed response')
        require(r['schema']==NAME+'.accounting' and r['caps']==caps() and r['wall_cap_seconds']==WALL,'same preregistered caps')
        require(set(r['attempted'])==set(r['completed'])==set(FIELDS),'complete resource vectors')
        for k in FIELDS:require(type(r['attempted'][k]) is int and type(r['completed'][k]) is int and 0<=r['completed'][k]<=r['attempted'][k]<=caps()[k],'actual cumulative caps')
    return dict(comparison='wall_capped_not_equal_FLOPs',elapsed_equality_required=False,
        within_wall_caps=all(r['elapsed_seconds']<=WALL for r in reports),
        elapsed_seconds={r['arm']:r['elapsed_seconds'] for r in reports},
        resource_responses={r['arm']:r['attempted'] for r in reports},equal_FLOPs=False,equal_total_compute=False,
        accounting_only=True,scientific_admission=False)


def files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build(output,receipt):
    base=prior.build(output,receipt)  # complete original saved consumer, no inference
    p=protocol(base['authority']);p['bindings_sha256']=deepcopy(base['bindings_sha256'])
    for f in files():p['bindings_sha256'][str(f)]=old.sha(f)
    report=json.loads((Path(output)/'report.json').read_bytes())
    compiled=old.verify(report['prepared'],report['prepare_sha256'])
    p['slots']=deepcopy(compiled['slots'])
    require([(s['root_slot'],s['condition']) for s in p['slots']]==[(0,'off'),(1,'off')],'same fixed known roots')
    p['expected_model_hash']=compiled['expected_model_hash']
    return p


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify'))
    ap.add_argument('--planner-output',required=True);ap.add_argument('--planner-receipt',required=True)
    ap.add_argument('--output');ap.add_argument('--prepared');ap.add_argument('--prepare-sha');a=ap.parse_args()
    p=build(a.planner_output,a.planner_receipt)
    if a.command=='prepare':
        path=Path(a.output);require(path.is_absolute() and path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink(),'exclusive new own absolute path')
        body=old.raw(p);require(len(body)<=4*1024**2,'prepare4MiB cap')
        with path.open('xb') as f:require(f.write(body)==len(body),'complete metadata');f.flush();os.fsync(f.fileno())
        old.fsync_dir(path.parent);result=dict(path=str(path),sha256=old.sha(path),comparison=p['comparison'])
    else:
        require(old.sha(a.prepared)==a.prepare_sha,'external own prepare SHA')
        old.need_equal(json.loads(Path(a.prepared).read_bytes()),p,'complete own protocol rebuild');result=dict(verified=True,comparison=p['comparison'])
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
