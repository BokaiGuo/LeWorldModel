"""Saved-only control preparation and full-commit node adapter; no live worker."""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p3_fullcommit_live_20261028_v1 as live

NAME = 'agent_p3_commitment_controls_20261029_v1'
ROOT = live.ROOT
core, signal = live.core, live.signal
require, sha, digest = live.require, live.sha, live.digest
MODES = ('fixedH1', 'fixedH3', 'adaptive')
REPORT = ROOT/'results/agent_p3_fullcommit_live_20261028_v1_MAIN01/report.json'
REPORT_SHA = '6c507b95a1cc60605d580032c0d92acfcd302abfbd79ebf9fa3c0a43e630b9ec'
STRICT = ROOT/'results/agent_p3_fullcommit_live_20261028_v1_consume_MAIN01.json'
STRICT_SHA = 'f7424d13342295e30fd432094913e2735b55f94155dcf3a171c313499d432721'
PREPARED = ROOT/'results/agent_p3_fullcommit_live_20261028_v1_prepare_MAIN01.json'
PREPARE_SHA = '80ac2306344fd1d63d68d5867ba733347d1c7b87343741ec1bb1e3f5afc4edfd'


def contract():
    return dict(schema=NAME+'.contract', modes=list(MODES), known_roots=2, episodes=6,
        node_slots=18, boundaries=[5,10,15], warmup=10, raw_control_target=20,
        original_prefix=5, fixedH1=dict(horizon=1,candidates=4),
        fixedH3=dict(horizon=3,candidates=8), adaptive='unchanged original chi/gain policy',
        pending='pay both signals; never replace active full commitment',
        tail='refuse unsupported or non-fitting whole node; no truncate/pad/retry/reselection',
        candidate_generator=live.native.rule(),
        budget_match='notmatched', equal_FLOPs=False, equal_wall=False, FLOPs=None,
        equal_actual_execution_horizon=None, independent_confirmation=False,
        freshness=False, P1_admission=False, P3_admission=False, scientific_admission=False,
        production_enabled=False, native_metrics='unchanged native7 reward and jointXY+angle success',
        source_use='seen known2 calibration; no historical outcome enters scheduler')


def roster():
    return [dict(episode_id=f's{s}_{m}',source_slot=s,mode=m,status='not_started',
        actual_control_returned=None,native_success=None,
        nodes=[dict(boundary=b,status='not_reached',receipt=None) for b in live.BOUNDARIES])
        for s in (0,1) for m in MODES]


def caps(mode):
    require(mode in MODES,'declared strategy')
    return live.arm_caps('fixed' if mode=='fixedH1' else 'adaptive')


def accounting():
    return dict(schema=NAME+'.accounting',budget_match='notmatched',
        per_episode=[dict(mode=m,cap=caps(m), signal_predict_if_all_boundaries=24,
            full_completion_predict=51 if m=='fixedH1' else 75 if m=='fixedH3' else None,
            actual_attempted=None,actual_completed=None,FLOPs=None,wall_ns=None,
            failed_attempts_refunded=False) for m in MODES],
        fixed_denominator=6,node_denominator=18,
        physical=dict(warmup_target=60,control_target=120,actual_public=None,
            actual_physics=None,actual_encoder=None,actual_projector=None),
        cap_is_not_actual_cost=True, padding_for_equalization=False)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),
            ROOT/'innerdocs'/(NAME+'.md')]


def upstream():
    """Full saved reconstruction, not a status-only or old-receipt fabrication."""
    for p,h in ((REPORT,REPORT_SHA),(STRICT,STRICT_SHA),(PREPARED,PREPARE_SHA)):
        require(sha(p)==h,'pinned original fullcommit authority')
    receipt=live.consume(REPORT.parent,REPORT_SHA,PREPARED,PREPARE_SHA)
    require(receipt==signal.read(STRICT) and receipt['complete'] is True,
            'exact independently reconstructed historical strict receipt')
    for p,h in ((REPORT,REPORT_SHA),(STRICT,STRICT_SHA),(PREPARED,PREPARE_SHA)):
        require(sha(p)==h,'historical authority unchanged through reconstruction')
    return signal.read(PREPARED)


def build():
    p=upstream()
    bindings={str(x):sha(x) for x in own_files()+[REPORT,STRICT,PREPARED]}
    for path,h in p['sources'].items():
        require(sha(path)==h,'historical immutable source closure')
        bindings[path]=h
    # tasks/layers/modelbinding are copied, never inferred from native outcomes.
    return dict(schema=NAME+'.prepare',contract=contract(),accounting=accounting(),
        authority=dict(report_sha256=REPORT_SHA,strict_sha256=STRICT_SHA,
            prepared_sha256=PREPARE_SHA,full_upstream_rebuilt_here=True),bindings=bindings,
        tasks=deepcopy(p['tasks']),layers=deepcopy(p['layers']),policy=deepcopy(p['policy']),
        modelbinding=deepcopy(p['modelbinding']),expected_model_hash=p['expected_model_hash'],
        source_manifest=p['source_manifest'],roster=roster(),
        production_blockers=['new six-episode live dispatcher/native census integration',
            'new strict RGB/native/journal consumer; no old strategy receipt relabeling',
            'MAIN review and explicit host CUDA plus physics authorization'])


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'external prepare SHA')
    p=signal.read(path)
    require(p==build() and sha(path)==expected,'complete read-only reconstruction')
    return p


def write_prepare(output):
    p=build();path=Path(output).absolute()
    require(path.parent==ROOT/'results' and path.name.startswith(NAME) and
            path.resolve()==path and not path.exists(),'exclusive new prepare')
    payload=core.canonical(p);require(len(payload)<=16*1024**2,'metadata cap')
    with path.open('xb') as f:
        require(f.write(payload)==len(payload),'whole metadata write')
        f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent)
    return dict(path=str(path),sha256=sha(path),production_enabled=False)


def make_request(base,context,*,slot,mode,boundary,raw10,observation,commitment=None):
    identity=dict(episode_id=f's{slot}_{mode}',source_slot=slot,mode=mode,
        boundary=boundary,raw10=np.asarray(raw10,np.float32).tolist(),observation=deepcopy(observation))
    return dict(schema=NAME+'.node',base_task_sha256=digest(base),
        task=live.local_task(base,context,identity),commitment=deepcopy(commitment),**identity)


def validate_request(req,base):
    core.keys(req,('schema','episode_id','source_slot','mode','boundary','raw10',
        'observation','task','commitment','base_task_sha256'))
    require(req['schema']==NAME+'.node' and req['mode'] in MODES and
        type(req['source_slot']) is int and req['source_slot'] in (0,1) and
        req['episode_id']==f"s{req['source_slot']}_{req['mode']}" and
        type(req['boundary']) is int and req['boundary'] in live.BOUNDARIES,'own fixed node ABI')
    t=core.validate_task(req['task']);identity={k:deepcopy(req[k]) for k in
        ('episode_id','source_slot','mode','boundary','raw10','observation')}
    require(req['base_task_sha256']==digest(base) and
        t==live.local_task(base,t['root_context'],identity),'original task and own current context binding')
    # Validation-only structural view: no old request is dispatched or receipted.
    view=deepcopy(req);view.update(schema=live.NAME+'.node',mode='fixed',
        episode_id=f"s{req['source_slot']}_fixed")
    live.validate_request(view)
    return t


def decision(task,mode,node,signals,receipt):
    require(mode in MODES,'fixed strategy roster')
    if mode=='adaptive':return signal.decide_root(task,'adaptive',node,signals,receipt)['decision']
    # Authenticate same-node signals even though fixed policies do not use their values.
    d=signal.decide_root(task,'fixed',node,signals,receipt)['decision']
    if mode=='fixedH3':d.update(horizon=3,candidates=8,decision_source=NAME+'.fixedH3')
    return d


def plan_node(req,base,factory,store,ledger,*,permit_ram=False):
    """Real lower-level compile/replay dispatch; no model loading or physics here."""
    t=validate_request(req,base);s=req['source_slot'];mode=req['mode']
    backend=factory(t,store)
    require(getattr(backend,'kind',None)=='HOST_CUDA' or
        (permit_ram is True and getattr(backend,'kind',None)=='RAM_FIXTURE'),'HOST CUDA or explicit RAM fixture')
    require(not ledger.events and all(ledger.caps[k]<=caps(mode)[k] for k in core.AXES),
        'fresh per-node ledger bounded by nonrefundable arm remainder')
    out=dict(schema=NAME+'.node_result',request_sha256=digest(req),status='attempted',
        responses=[],diagnostic=None,decision=None,bank=None,candidates=[],winner=None,
        commands=None,selected_path=None,refusal=None,error=None,resource_receipt=None)
    try:
        for h in (1,3):
            ledger.scope=dict(slot=s,horizon=h)
            request=core.request(t,dict(context=t['root_context']),operation='compile_finite_horizon_J',horizon=h)
            response=ledger.call('signal_compile',lambda request=request:backend.signal(request,ledger))
            out['responses'].append(response);store.event(dict(kind='local_signal',response=response))
        ledger.scope=dict(slot=s,phase='diagnostic')
        out['diagnostic']=signal.spectra(t,out['responses'],svd=lambda a:
            ledger.call('array_svd',lambda:np.linalg.svd(a,full_matrices=False)))
        if req['commitment'] is not None:
            out['status']='observed_commitment';return out
        node,signals,receipt=signal.runtime_signal(t,out['responses'],out['diagnostic'])
        ledger.scope=dict(slot=s,phase='schedule')
        d=ledger.call('schedule',lambda:decision(t,mode,node,signals,receipt));out['decision']=d
        h,n=d['horizon'],d['candidates']
        if d['program_type']!='progress' or d['extension_type']!='progress' or not d['continue_expanding']:
            out.update(status='refused',refusal='unsupported_original_decision');return out
        if 5*h>20-req['boundary']:
            out.update(status='refused',refusal='fullH_does_not_fit');return out
        out['candidates']=[dict(index=i,status='not_started',actions=None,whole=None,
            segmented=None,cost=None) for i in range(n)]
        ledger.scope=dict(slot=s,mode=mode,horizon=h,phase='program')
        response=next(x for x in out['responses'] if x['horizon']==h)
        out['bank']=ledger.call('program_compile',lambda:live.generate(t,response,n,ledger))
        for row in out['candidates']:
            row.update(status='attempted',actions=out['bank']['actions'][row['index']])
            ledger.scope=dict(slot=s,mode=mode,index=row['index'],horizon=h,phase='program')
            row['whole']=ledger.call('replay',lambda:backend.replay(t['root_context'],row['actions'],False,ledger))
            row['segmented']=ledger.call('replay',lambda:backend.replay(t['root_context'],row['actions'],True,ledger))
            require(row['whole']==row['segmented'],'all candidate full/segmented exact')
            terminal=core.checked_path(t,t['root_context'],row['actions'],row['whole'])
            row['cost']=ledger.call('metric',lambda:core.model_cost(t,terminal));row['status']='returned'
        winner=min(out['candidates'],key=lambda r:(r['cost'],r['index']))
        ledger.scope=dict(slot=s,mode=mode,index=winner['index'],horizon=h,phase='selected')
        path=ledger.call('replay',lambda:backend.replay(t['root_context'],winner['actions'],False,ledger))
        require(path==winner['whole'],'independent complete selected replay')
        require(ledger.attempted==ledger.completed==live.costs(h,n,schedule=True),'exact dispatch axes')
        out.update(status='sealed',winner=deepcopy(winner),commands=deepcopy(winner['actions']),selected_path=path)
    except BaseException as exc:
        out.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        out['resource_receipt']=ledger.receipt();store.event(dict(kind=NAME+'.node_final',result=deepcopy(out)))
    return out


def remaining_caps(mode,results):
    c=caps(mode)
    for r in results:
        rr=r['resource_receipt'];core.audit_ledger(rr)
        for k in c:c[k]-=rr['attempted'][k]
    require(all(v>=0 for v in c.values()),'failed attempts retained; arm exhausted')
    return c


class Commitment:
    """Packet release/actual-return gate; observation callbacks remain outside."""
    def __init__(self,request,result):
        require(result['status']=='sealed' and result['request_sha256']==digest(request),
            'same-node sealed program only')
        self.start=request['boundary'];self.actions=np.asarray(result['commands'],np.float32)
        require(self.actions.shape in ((5,2),(15,2)) and np.isfinite(self.actions).all()
            and (abs(self.actions)<=1).all() and self.start+len(self.actions)<=20,'whole legal fitting program')
        require(np.array_equal(self.actions,np.asarray(result['winner']['actions'],np.float32)),
            'winner action binding')
        self.actions=self.actions.copy();self.actions.flags.writeable=False
        self.seal=digest([request,result]);self.returned=0;self.pending=None;self.stopped=False

    def packet(self,boundary):
        require(not self.stopped and self.pending is None and type(boundary) is int and
            boundary==self.start+self.returned and self.returned<len(self.actions),'sequential full commitment; no replacement')
        self.pending=self.actions[self.returned:self.returned+5].copy();return self.pending.copy()

    def acknowledge(self,executed,*,terminated=False,truncated=False):
        require(type(terminated) is bool and type(truncated) is bool and self.pending is not None,'actual native return disposition')
        a=np.asarray(executed,np.float32)
        require(a.ndim==2 and a.shape[1]==2 and 1<=len(a)<=5 and
            a.tobytes()==self.pending[:len(a)].tobytes() and
            (len(a)==5 or terminated or truncated),'exact returned rawbits; short only native stop')
        self.returned+=len(a);self.pending=None;self.stopped=terminated or truncated

    def active(self,boundary):
        require(self.pending is None and boundary==self.start+self.returned,'ack before boundary observation')
        if self.stopped or boundary==self.start+len(self.actions):return None
        return dict(start=self.start,end=self.start+len(self.actions),seal_sha256=self.seal)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','accounting'))
    p.add_argument('--output');p.add_argument('--prepared');p.add_argument('--prepare-sha')
    a=p.parse_args(argv)
    if a.command=='prepare':r=write_prepare(a.output)
    elif a.command=='verify':r=dict(verified=True,schema=verify(a.prepared,a.prepare_sha)['schema'],production_enabled=False)
    else:r=accounting()
    print(core.canonical(r).decode())


if __name__=='__main__':main()
