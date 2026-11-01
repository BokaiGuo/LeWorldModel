"""C3 same-task causal program runtime contract. No neural/physics loader.

MAIN supplies a reviewed, fully metered HOST CUDA compiler/replay adapter and
operation caps. RAM fixtures exercise plumbing only, never scientific evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import p3_adaptive_runtime_20261027 as existing
from experiments.spectral_reachability.d2p_adaptive_budget_v2 import FrozenSignals
from experiments.spectral_reachability.d2p_kinodynamic_program_search import _validated_context

NAME='agent_p3_runtime_contract_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
PROPOSAL=Path('/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt')
COLD_REPORT=ROOT/'results/agent_p3_resume_20261027_next_cuda_MAIN01/report.json'
COLD_REPORT_SHA='5bb52423c796713899a4947f9c0167ad9dfa32d1b336b7abf858d17dd45f8462'
COLD_CONSUME=ROOT/'results/agent_p3_resume_20261027_next_consume_MAIN01.json'
COLD_CONSUME_SHA='0403affb31fa137308502f5875a05960222c60540b9cf28b91012984310cd65c'
FLAGS=dict(P1_admission=False,P2_admission=False,P3_admission=False,freshness=False,
    full_C3_demonstrated=False,independent_confirmation=False,physical_release=False,
    compute_advantage=False,walltime_advantage=False)
BINDINGS=('root_sha256','task_id','task_sha256','task_contract_sha256','blind_sha256',
    'model_sha256','goal_sha256','projection_sha256','normalizer_sha256','context_sha256','action_domain_sha256')
AXES=('predict_API','predict_samples','predict_context_tokens','action_encoder_calls',
    'action_encoder_samples','action_encoder_context_tokens','predictor_calls','predictor_samples',
    'predictor_context_tokens','pred_proj_calls','pred_proj_rows','encoder_calls','encoder_samples',
    'projector_calls','projector_samples','jacobian_API','VJP_directions','grad_API',
    'raw_backward_callbacks','model_SVD_calls','array_SVD_calls','solve_calls',
    'signal_compile_calls','program_compile_calls','replay_calls','metric_calls','schedule_decisions')
SCALARS={'jacobian':'jacobian_API','grad':'grad_API','raw_backward':'raw_backward_callbacks',
    'model_svd':'model_SVD_calls','array_svd':'array_SVD_calls','solve':'solve_calls',
    'signal_compile':'signal_compile_calls','program_compile':'program_compile_calls','replay':'replay_calls',
    'metric':'metric_calls','schedule':'schedule_decisions'}


def require(ok,message):
    if not ok: raise ValueError(message)


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def keys(value,names): require(type(value) is dict and set(value)==set(names),'exact outcome-free fields: '+','.join(names))
def integer(value,minimum=0): require(type(value) is int and value>=minimum,'literal bounded integer')
def hash_value(value): require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'literal SHA256')


def read(path):
    path=Path(path); require(path.stat().st_size<=2*1024**2,'bounded metadata')
    def pairs(items):
        d={}
        for k,v in items: require(k not in d,'duplicate JSON key'); d[k]=v
        return d
    v=json.loads(path.read_bytes(),object_pairs_hook=pairs); canonical(v); return v


def finite(value,shape,float32=False):
    def check(x):
        if type(x) is list:
            for y in x: check(y)
        else: require(type(x) in (int,float) and math.isfinite(x),'literal finite numbers, no bool')
    check(value); a=np.asarray(value)
    require(a.shape==shape and a.dtype.kind in 'fi' and np.isfinite(a).all(),'finite exact array shape')
    if float32: require(np.array_equal(a,np.asarray(a,np.float32).astype(float)),'literal float32, no repair')
    return a


def context(value):
    c=_validated_context(value)
    require(c['context_length']==3 and c['action_dim']==10,'full three-state/two-token context')
    finite(c['latent_history'],(3,c['latent_dim']),True); finite(c['action_history'],(2,10),True)
    return c


def validate_task(task):
    keys(task,('binding','root_context','goal','projection','action_mean','action_std'))
    keys(task['binding'],BINDINGS)
    for h in task['binding'].values(): hash_value(h)
    c=context(task['root_context']); d=c['latent_dim']
    finite(task['goal'],(d,)); finite(task['projection'],(d,d))
    finite(task['action_mean'],(2,),True); std=finite(task['action_std'],(2,),True)
    require(np.all(std>0),'positive frozen action scale')
    b=task['binding']
    require(b['goal_sha256']==digest(task['goal']) and b['projection_sha256']==digest(task['projection']) and
        b['context_sha256']==digest(c) and b['normalizer_sha256']==digest([task['action_mean'],task['action_std']]),'same task/goal/Q/normalizer/context identity')
    require(b['action_domain_sha256']==digest(action_domain()),'fixed original raw action domain')
    return deepcopy(task)


def action_domain(): return dict(raw_bounds=[-1,1],packet_steps=5,action_dim=2,normalization='original saved float32 subtract/divide; no /100 or clipping')


def policy(task,node=None):
    # Same inherited engineering constants, NOT new calibrated thresholds.
    parent=task['root_context'] if node is None else node['context']
    return existing.RuntimePolicy(digest(dict(task_binding_sha256=digest(task['binding']),parent_context_sha256=digest(parent))))


def counters(value):
    keys(value,AXES)
    for v in value.values(): integer(v)
    return deepcopy(value)


def delta(resource,shape=None,directions=None):
    out=dict.fromkeys(AXES,0)
    if resource in SCALARS:
        require(shape is None and directions is None,'scalar operation metadata'); out[SCALARS[resource]]=1
    elif resource=='vjp':
        integer(directions,1); require(shape is None,'VJP direction units'); out['VJP_directions']=directions
    else:
        require(type(shape) is list and all(type(n) is int and n>0 for n in shape) and directions is None,'actual input shape metadata')
        if resource in ('predict','action_encoder','predictor'):
            require(len(shape)==3 and shape[1]==3,'full native batch/context input')
            if resource=='predict': out.update(predict_API=1,predict_samples=shape[0],predict_context_tokens=shape[0]*shape[1])
            else:
                out[resource+'_calls']=1; out[resource+'_samples']=shape[0]; out[resource+'_context_tokens']=shape[0]*shape[1]
        elif resource=='pred_proj':
            require(len(shape)==2,'projection flatten input shape'); out.update(pred_proj_calls=1,pred_proj_rows=shape[0])
        elif resource in ('encoder','projector'):
            require(len(shape)>=2,'actual encode/project input shape'); out[resource+'_calls']=1; out[resource+'_samples']=shape[0]
        else: raise ValueError('unknown operation; physics is not an allowed resource')
    return out


class BudgetExceeded(RuntimeError):
    """Predispatch refusal, never refunds prior failed/returned work."""


class Ledger:
    """Reviewed adapter hooks must call at actual entries, not estimated plans.

    Input shapes are hook metadata, not verified GPU tensors in this core.
    No counter can prove that a hidden/uninstrumented backend call did not run.
    """
    def __init__(self,caps,clock=time.monotonic_ns,sink=None):
        self.caps=counters(caps); self.clock=clock; self.start=clock(); self.events=[]
        self.sink=sink if sink is not None else lambda e:None
        self.attempted=dict.fromkeys(AXES,0); self.completed=dict.fromkeys(AXES,0)
        self.scope=None; self.failed=False

    def call(self,resource,fn,*,shape=None,directions=None):
        require(not self.failed and self.scope is not None,'scoped fail-stop ledger')
        change=delta(resource,shape,directions); index=len(self.events)
        e=dict(index=index,scope=deepcopy(self.scope),resource=resource,shape=shape,directions=directions,
            delta=change,status='attempted',started_ns=self.clock()-self.start,ended_ns=None,error=None)
        self.events.append(e)
        if any(self.attempted[k]+change[k]>self.caps[k] for k in AXES):
            e['status']='blocked_before_dispatch'; self.failed=True
            self.sink(dict(kind='resource_blocked',entry=deepcopy(e)))
            raise BudgetExceeded('resource cap before '+resource)
        for k in AXES: self.attempted[k]+=change[k]
        try:
            self.sink(dict(kind='resource_attempt',entry=deepcopy(e)))
            value=fn(); e['ended_ns']=self.clock()-self.start; e['status']='returned'
            self.sink(dict(kind='resource_return',entry=deepcopy(e)))
            for k in AXES: self.completed[k]+=change[k]
            return value
        except BaseException as exc:
            e.update(status='failed',ended_ns=self.clock()-self.start,error=type(exc).__name__+': '+str(exc)); self.failed=True
            try: self.sink(dict(kind='resource_failure',entry=deepcopy(e)))
            except Exception as failure: e['error']+='; failure journal: '+str(failure)
            raise

    def receipt(self):
        inclusive={}
        for e in self.events:
            if e['ended_ns'] is not None:
                inclusive[e['resource']]=inclusive.get(e['resource'],0)+e['ended_ns']-e['started_ns']
        return deepcopy(dict(caps=self.caps,attempted=self.attempted,completed=self.completed,events=self.events,
            wall_ns=self.clock()-self.start,backward_engine_calls=None,cuda_ns=None,FLOPs=None,
            inclusive_operation_wall_ns=inclusive,wall_semantics='nested inclusive; never sum into total/FLOPs',
            physics_steps=0,unknowns_are_not_zero=True))


def audit_ledger(receipt):
    caps=counters(receipt['caps']); attempted=dict.fromkeys(AXES,0); completed=dict.fromkeys(AXES,0)
    inclusive={}
    require([e['index'] for e in receipt['events']]==list(range(len(receipt['events']))),'ordered resource events')
    for e in receipt['events']:
        integer(e['index']); integer(e['started_ns']); require(e['scope'] is not None,'resource scope')
        change=delta(e['resource'],e['shape'],e['directions'])
        require(canonical(e['delta'])==canonical(change),'actual resource units, no sample/API substitution')
        if e['status']=='blocked_before_dispatch':
            require(any(attempted[k]+change[k]>caps[k] for k in AXES) and e['ended_ns'] is None,'real predispatch block')
            continue
        require(e['status'] in ('returned','failed'),'complete attempt disposition')
        integer(e['ended_ns']); require(e['ended_ns']>=e['started_ns'],'nonnegative measured operation wall')
        inclusive[e['resource']]=inclusive.get(e['resource'],0)+e['ended_ns']-e['started_ns']
        for k in AXES:
            attempted[k]+=change[k]
            if e['status']=='returned': completed[k]+=change[k]
        require(all(attempted[k]<=caps[k] for k in AXES),'nonrefundable actual caps')
    require(canonical(attempted)==canonical(receipt['attempted']) and canonical(completed)==canonical(receipt['completed']),'complete resource totals')
    require(all(receipt[k] is None for k in ('backward_engine_calls','cuda_ns','FLOPs')) and
        type(receipt['physics_steps']) is int and receipt['physics_steps']==0 and receipt['unknowns_are_not_zero'] is True,'unknowns remain null; no physical execution')
    integer(receipt['wall_ns']); require(all(e['ended_ns'] is None or e['ended_ns']<=receipt['wall_ns'] for e in receipt['events']),'operation within runtime wall')
    require(canonical(inclusive)==canonical(receipt['inclusive_operation_wall_ns']),'actual inclusive operation times')
    return dict(attempted=attempted,completed=completed)


def checked_path(task,parent,actions,path):
    raw=finite(actions,(len(actions),2),True)
    require(len(raw)>0 and len(raw)%5==0 and np.all(np.abs(raw)<=1),'complete legal raw packets; no padding')
    require(type(path) is list and len(path)==len(raw)//5,'all predicted packet boundaries')
    before=context(parent)
    for k,after in enumerate(path):
        after=context(after); require(after['latent_dim']==before['latent_dim'],'same latent layout')
        token=((raw[5*k:5*k+5].astype(np.float32)-np.asarray(task['action_mean'],np.float32))/np.asarray(task['action_std'],np.float32)).reshape(10)
        expected_actions=[before['action_history'][-1],token.astype(float).tolist()]
        require(canonical(after['action_history'])==canonical(expected_actions) and canonical(after['latent_history'][:-1])==canonical(before['latent_history'][1:]),'exact parent-child history/action identity closure')
        before=after
    return deepcopy(before)


def request(task,node,**fields):
    return dict(binding=deepcopy(task['binding']),binding_sha256=digest(task['binding']),
        parent_context=deepcopy(node['context']),parent_context_sha256=digest(node['context']),**fields)


def local_signal(task,node,backend,ledger,p):
    responses=[]; paid_before=deepcopy(ledger.attempted)
    d=len(task['goal'])
    for h in p.schedule().horizons:
        req=request(task,node,operation='compile_finite_horizon_J',horizon=h)
        result=ledger.call('signal_compile',lambda req=req:backend.signal(deepcopy(req),ledger))
        keys(result,('binding_sha256','parent_context_sha256','horizon','nominal','jacobian','producer_sha256'))
        require(result['binding_sha256']==req['binding_sha256'] and result['parent_context_sha256']==req['parent_context_sha256'] and
            type(result['horizon']) is int and result['horizon']==h,'same node/goal/model local derivative, not cross-root routing')
        hash_value(result['producer_sha256']); finite(result['nominal'],(d,)); finite(result['jacobian'],(d,10*h))
        responses.append(deepcopy(result))
    # Nominal and AD forward cost must BOTH be paid; backend may pay more.
    paid={k:ledger.attempted[k]-paid_before[k] for k in AXES}
    require(paid['predict_API']>=2*sum(p.schedule().horizons) and paid['jacobian_API']==2 and
        paid['VJP_directions']>=2*d and paid['grad_API']>=2 and paid['raw_backward_callbacks']>=2,
        'missing nominal/AD-forward/derivative resource evidence')
    q=np.asarray(task['projection'],float); r=q@(np.asarray(task['goal'])-np.asarray(responses[0]['nominal']))
    energy=float(r@r); require(math.isfinite(energy),'finite residual energy')
    covers=[]; details=[]
    for response in responses:
        k=q@np.asarray(response['jacobian'],float)
        u,s,_=ledger.call('array_svd',lambda k=k:np.linalg.svd(k,full_matrices=False))
        rank=int(np.count_nonzero(s>s[0]*p.svd_relative_cutoff)) if len(s) and s[0]>0 else 0
        parallel=u[:,:rank]@(u[:,:rank].T@r)
        chi=0. if energy==0 else float(np.clip(((r-parallel)@(r-parallel))/energy,0,1))
        covers.append(chi); details.append(dict(horizon=response['horizon'],rank=rank,singular_values=s.tolist(),
            unreachable_fraction_same_reference_residual=chi))
    signed_gain=covers[0]-covers[1]
    signals=FrozenSignals(policy(task,node).source_seal,covers[0],max(0.,signed_gain))
    return signals,dict(schema=NAME+'.signal',responses=responses,spectra=details,
        reference='short-horizon nominal residual for BOTH spaces; not best-cost stagnation or physical labels',
        residual=r.tolist(),residual_energy=energy,signed_horizon_coverage_gain=signed_gain,
        scheduling_signal=asdict(signals),no_control_or_safety_guarantee=True)


def choose(task,node,signals,mode):
    p=policy(task,node); inherited=p.schedule().choose(signals)
    require(mode in ('fixed','adaptive'),'fixed/adaptive paired method')
    high=signals.unreachable_fraction>=p.chi_threshold
    if mode=='fixed':
        kind,h,count,depth='progress',p.schedule().horizons[0],p.base_beam,p.max_depth
    else:
        kind='reconfiguration' if high else 'progress'; h=inherited['horizon_tokens']; count=inherited['expanded_candidates']
        depth=p.max_depth if high or signals.horizon_gain>=p.gain_threshold else 1
    return dict(program_type='composition' if node['depth'] else kind,extension_type=kind,
        horizon=h,candidates=count,program_depth_limit=depth,
        continue_expanding=node['depth']<depth,decision_source='same-task local frozen-model Jacobians',
        inherited_engineering_thresholds=dict(chi=p.chi_threshold,gain=p.gain_threshold),
        no_optimality_claim=True)


def model_cost(task,c):
    r=np.asarray(task['projection'])@(np.asarray(task['goal'])-np.asarray(c['latent_history'][-1]))
    v=float(np.mean(r*r)); require(math.isfinite(v),'finite same-task model-Q'); return v


def run_task(task,backend,caps,*,mode,permit_ram=False):
    task=validate_task(task); p=policy(task)
    require(getattr(backend,'kind',None)=='HOST_CUDA' or (permit_ram is True and getattr(backend,'kind',None)=='RAM_FIXTURE'),'reviewed HOST CUDA adapter or explicit RAM fixture only')
    require(mode in ('fixed','adaptive'),'paired mode')
    native=backend.kind=='HOST_CUDA'
    sink=getattr(backend,'event_sink',None)
    require(not native or callable(sink),'HOST CUDA adapter must supply predispatch durable event sink')
    ledger=Ledger(caps,sink=sink); rows=[dict(slot=i,status='not_requested',request=None,payload=None,error=None) for i in range(p.max_expansions)]
    result=dict(schema=NAME+'.runtime',mode=mode,binding=task['binding'],status='not_started',
        jobs=rows,signals=[],queues=[],replays=[dict(mode=m,status='not_requested',path=None,error=None) for m in ('whole','segmented')],
        diagnostic_best=None,model_plan=None,errors=[],resource_receipt=None,backend_kind=backend.kind,
        fixed_denominator=dict(task_branches=1,expansion_cap=p.max_expansions,replay_slots=2),**FLAGS)
    root=dict(context=task['root_context'],depth=0,indices=[],actions=[],path=[],edges=[])
    frontier=[root]; best=None; used=0; stop=False
    def rank(n): return (n['cost'],n['depth'],n['indices'])
    try:
        for level in range(p.max_depth):
            children=[]
            for parent in frontier:
                ledger.scope=dict(mode=mode,phase='signal',parent_context_sha256=digest(parent['context']))
                signal,receipt=local_signal(task,parent,backend,ledger,p)
                result['signals'].append(receipt)
                ledger.scope=dict(mode=mode,phase='schedule',parent_context_sha256=digest(parent['context']))
                decision=ledger.call('schedule',lambda:choose(task,parent,signal,mode))
                reqs=[request(task,parent,operation='compile_program_extension',
                    **decision,candidate_index=i,prefix_edges=deepcopy(parent['edges']),
                    prefix_actions=deepcopy(parent['actions']),signal_sha256=digest(receipt),
                    local_response=deepcopy(receipt['responses'])) for i in range(decision['candidates'])] if decision['continue_expanding'] else []
                result['queues'].append(dict(parent_context_sha256=digest(parent['context']),depth=parent['depth'],decision=decision,requests=reqs))
                if used+len(reqs)>p.max_expansions:
                    result['status']='expansion_cap_before_node_queue'; stop=True; break
                assigned=rows[used:used+len(reqs)]
                for row,req in zip(assigned,reqs,strict=True): row.update(status='queued',request=deepcopy(req))
                used+=len(reqs)
                for row,req in zip(assigned,reqs,strict=True):
                    row['status']='attempted'; ledger.scope=dict(mode=mode,phase='program',slot=row['slot'],parent_context_sha256=digest(parent['context']))
                    before=deepcopy(ledger.attempted)
                    v=ledger.call('program_compile',lambda req=req:backend.generate(deepcopy(req),ledger)); row['payload']=deepcopy(v)
                    keys(v,('binding_sha256','parent_context_sha256','extension_type','actions','path','producer_sha256'))
                    require(v['binding_sha256']==req['binding_sha256'] and v['parent_context_sha256']==req['parent_context_sha256'] and
                        v['extension_type']==req['extension_type'],'exact witnessed extension identity/type')
                    hash_value(v['producer_sha256']); require(len(v['actions'])==5*req['horizon'],'fixed selected horizon, no truncation')
                    terminal=checked_path(task,parent['context'],v['actions'],v['path'])
                    require(ledger.attempted['predict_API']-before['predict_API']>=req['horizon'],'no unpaid invented edge')
                    edge=dict(kind=v['extension_type'],input_context_sha256=digest(parent['context']),output_context_sha256=digest(terminal),
                        actions_sha256=digest(v['actions']),producer_sha256=v['producer_sha256'],actions=deepcopy(v['actions']),path=deepcopy(v['path']))
                    node=dict(context=terminal,depth=parent['depth']+1,indices=parent['indices']+[req['candidate_index']],
                        actions=parent['actions']+deepcopy(v['actions']),path=parent['path']+deepcopy(v['path']),
                        edges=parent['edges']+[edge],cost=ledger.call('metric',lambda:model_cost(task,terminal)))
                    row.update(status='returned',node=deepcopy(node)); children.append(node)
                    if best is None or rank(node)<rank(best): best=node
            if stop or not children: break
            children.sort(key=rank); width=p.base_beam*(p.expansion_factor if mode=='adaptive' else 1)
            frontier=children[:width]
        if not stop:
            require(best is not None,'no witnessed nonempty program')
            # Exact whole/segmented predicted replay; never a physical dispatch.
            for slot in result['replays']:
                slot['status']='attempted'; ledger.scope=dict(mode=mode,phase='verification',replay_mode=slot['mode'])
                before=ledger.attempted['predict_API']
                req=request(task,root,operation='verify_program',mode=slot['mode'],actions=deepcopy(best['actions']),edges=deepcopy(best['edges']))
                path=ledger.call('replay',lambda req=req:backend.replay(deepcopy(req),ledger)); slot['path']=deepcopy(path)
                terminal=checked_path(task,root['context'],best['actions'],path)
                require(canonical(path)==canonical(best['path']) and canonical(terminal)==canonical(best['context']),'exact full action witness independent replay')
                require(ledger.attempted['predict_API']-before>=len(best['actions'])//5,'whole/segmented replay actually paid')
                slot['status']='returned'
            require(not ledger.failed and ledger.attempted==ledger.completed,'every paid attempt returned; swallowed failures never release a plan')
            result.update(status='ram_complete_not_evidence' if backend.kind=='RAM_FIXTURE' else 'model_runtime_complete_not_admitted',model_plan=deepcopy(best))
    except Exception as exc:
        result['status']='failed_or_budget_exhausted'; result['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
        for row in rows+result['replays']:
            if row['status']=='attempted': row.update(status='failed',error=str(exc))
            elif row['status']=='queued': row.update(status='blocked_after_failure',error=str(exc))
    result['diagnostic_best']=deepcopy(best); result['resource_receipt']=ledger.receipt()
    audit_ledger(result['resource_receipt'])
    # Not a model authenticity proof: reviewed native hooks remain mandatory.
    for a,b in (('predict_API','predictor_calls'),('predict_API','action_encoder_calls'),('predict_API','pred_proj_calls'),
            ('predict_samples','predictor_samples'),('predict_samples','action_encoder_samples'),
            ('predict_context_tokens','predictor_context_tokens'),('predict_context_tokens','action_encoder_context_tokens'),
            ('predict_context_tokens','pred_proj_rows')):
        if ledger.attempted[a]!=ledger.attempted[b] and result['model_plan'] is not None:
            result.update(status='native_module_accounting_mismatch',model_plan=None)
            result['errors'].append(dict(type='AccountingMismatch',message=a+' != '+b))
    return deepcopy(result)


def run_panel(tasks,backend_factory,caps,*,permit_ram=False):
    require(type(tasks) is list and bool(tasks),'fixed nonempty complete task roster')
    identities=[digest(validate_task(t)['binding']) for t in tasks]
    require(len(set(identities))==len(identities),'no duplicate task aliases')
    rows=[dict(task_binding_sha256=h,mode=m,status='not_started',runtime=None,error=None) for h in identities for m in ('fixed','adaptive')]
    for row,task in zip(rows,[t for t in tasks for _ in ('fixed','adaptive')],strict=True):
        try:
            row['runtime']=run_task(task,backend_factory(deepcopy(task),row['mode']),caps,mode=row['mode'],permit_ram=permit_ram)
            row['status']=row['runtime']['status']
        except Exception as exc: row.update(status='setup_failed',error=type(exc).__name__+': '+str(exc))
    return dict(schema=NAME+'.panel',rows=rows,fixed_denominator=len(rows),failed_or_incomplete=sum(r['status'] not in ('ram_complete_not_evidence','model_runtime_complete_not_admitted') for r in rows),**FLAGS)


def prepare_material():
    from experiments.spectral_reachability import agent_p3_resume_20261027_next as prior
    require(sha(COLD_REPORT)==COLD_REPORT_SHA and sha(COLD_CONSUME)==COLD_CONSUME_SHA,'MAIN actual cold report/strictconsume SHA')
    receipt=read(COLD_CONSUME); require(receipt['status']=='cold_saved_verified_not_admitted' and receipt['report_sha256']==COLD_REPORT_SHA,'MAIN saved receipt binding')
    source=ROOT/'results/agent_p3_resume_20261027_next_prepare01.json'
    p,tasks,_=prior.verify(source,'47cc8ff5a32ef3ce36b0dd6807b7c1ff68fea3736137d0e948122d050b2e2846')
    runtime_tasks=[]
    for slot,t in zip(p['slots'],tasks,strict=True):
        task={k:deepcopy(t[k]) for k in ('root_context','goal','projection','action_mean','action_std')}
        task['binding']={k:slot[k] for k in BINDINGS if k in slot}
        task['binding'].update(model_sha256=p['expected_model_hash'],goal_sha256=digest(t['goal']),projection_sha256=digest(t['projection']),
            normalizer_sha256=digest([t['action_mean'],t['action_std']]),context_sha256=digest(t['root_context']),action_domain_sha256=digest(action_domain()))
        runtime_tasks.append(validate_task(task))
    files=[Path(__file__),ROOT/f'tests/{NAME}_test.py',ROOT/f'innerdocs/{NAME}.md',Path(existing.__file__),
        ROOT/'experiments/spectral_reachability/d2p_adaptive_budget_v2.py',ROOT/'experiments/spectral_reachability/d2p_kinodynamic_program_search.py',
        ROOT/'experiments/spectral_reachability/d2p_composition_witness_v2.py',source,COLD_REPORT,COLD_CONSUME,PROPOSAL]
    bindings=deepcopy(p['bindings_sha256']); bindings.update({str(f.resolve()):sha(f) for f in files})
    return dict(schema=NAME+'.prepare',status='contract_prepared_missing_native_adapter_and_operation_profile',tasks=runtime_tasks,
        policies=[asdict(policy(t)) for t in runtime_tasks],fixed_denominator=len(runtime_tasks)*2,
        source_sha256=bindings,resource_caps=None,native_backend=None,compiler_objective_spec=None,execution_authorized=False,
        root_requests=[dict(task_binding_sha256=digest(t['binding']),mode=m,operation='compile_finite_horizon_J',
            horizons=list(policy(t).schedule().horizons),program_queue=None,reason='must wait for local paid model signals') for t in runtime_tasks for m in ('fixed','adaptive')],
        historical_resources=p['historical_resources'],historical_encoder=p['historical_encoder'],historical_anchor_receipts=p['historical_anchor_receipts'],
        prior_cold_result='preserved cross-root proxy negative; NOT a full C3 experiment',
        missing_dependencies=['MAIN-reviewed metered native signal/progress/reconfiguration/replay adapter for same-node H1/H3',
            'presealed compiler objective/action-chart parameters and full native model/source guard plus saved-array writer/consumer',
            'presealed complete per-resource operation caps and reserve for independent replay; no cap tuned to outcomes',
            'independent P1 mechanism admission and applicable P2 composition-increment evidence',
            'certified fresh ordered task/root/goal cohort and full acquisition history',
            'actual HOST CUDA runtime traces and matched-budget controls/physical consequences'],**FLAGS)


def write_new(path,value):
    p=Path(path).absolute(); raw=canonical(value)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and not p.exists() and not p.is_symlink(),'new exclusive runtime prefix only')
    require(len(raw)<=2*1024**2 and shutil.disk_usage(p.parent).free>=200*1024**2+len(raw),'metadata cap/disk floor')
    with p.open('xb') as f: require(f.write(raw)==len(raw),'short write'); f.flush(); os.fsync(f.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(path=str(p),sha256=sha(p),bytes=len(raw))


def verify(path,expected):
    hash_value(expected); require(sha(path)==expected,'external prepare SHA')
    p=prepare_material(); require(canonical(read(path))==canonical(p) and sha(path)==expected,'full source/task/contract rebuild')
    return p


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('prepare','verify','queue'))
    parser.add_argument('--output'); parser.add_argument('--prepared'); parser.add_argument('--sha256'); a=parser.parse_args(argv)
    if a.action=='prepare': value=write_new(a.output,prepare_material())
    else:
        p=verify(a.prepared,a.sha256)
        value=dict(status=p['status'],execution_authorized=False,missing_dependencies=p['missing_dependencies'],**FLAGS)
        if a.action=='queue': value['root_requests']=p['root_requests']
    print(canonical(value).decode()); return 0


if __name__=='__main__': raise SystemExit(main())
