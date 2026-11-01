"""Three-arm common-controller orchestration seam. MAIN supplies real adapters.

No model/simulator imports, training, network, implicit downloads or CPU fallback.
Prepare is engineering readiness only; run additionally requires external MAIN
P1/freshness/physical authority. Fixture dispatch is explicitly non-scientific.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path

from experiments.spectral_reachability import agent_subgoal_gap_contract_20261028_v1 as c

NAME='agent_subgoal_gap_runner_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
CONTRACT_SHA='f8d48e7dec75b01ebac1bed915456dbc163acf0a83fac00bfb3e5463aee11c25'
PHASES=('restore','proposal','controller','native')
KNOWN=('callback_calls','restore_calls','generator_calls','image_encoder_calls','projector_calls',
    'predict_calls','predict_samples','action_encoder_calls','jacobian_calls','math_vjp_directions',
    'svd_calls','solve_calls','physics_steps','renders','native_evaluator_calls')
UNKNOWN=('flops','cuda_kernel_calls','backward_engine_api_calls','device_time_ns','peak_gpu_bytes','inherited_context_cost')
COMMON=('model','encoder','normalizer','q','controller','observation','budget','native_success_evaluator')
ASSETS=COMMON+('learned_checkpoint','learned_pairing','arbitrary_rule','compiled_rule','adapter_source','restore_contract')
require=c.require


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def vector():return dict.fromkeys(KNOWN,0)|dict.fromkeys(UNKNOWN,None)


def reference(ref,bindings,blockers,label):
    if ref is None:blockers.append(label+': missing asset');return None
    c.keys(ref,('path','sha256'));c.sha(ref['sha256']);p=Path(ref['path'])
    require(p.is_absolute(),'absolute asset path')
    if not p.is_file():blockers.append(label+': file absent');return None
    require(not p.is_symlink() and file_sha(p)==ref['sha256'],'asset byte drift: '+label)
    bindings[str(p)]=ref['sha256'];return ref['sha256']


def closure():
    require(file_sha(c.__file__)==CONTRACT_SHA,'frozen contract v1 drift')
    paths=(Path(__file__),ROOT/'tests'/(NAME+'_tests.py'),ROOT/'innerdocs'/(NAME+'.md'),Path(c.__file__),
        ROOT/'experiments/spectral_reachability/d2p_composition_witness_v2.py')
    return {str(p.resolve()):file_sha(p) for p in paths}


def prepare(spec):
    c.keys(spec,('schema','latent_dim','checkpoints','assets','phase_caps','trials'))
    require(spec['schema']==NAME+'.input','runner input schema')
    d=spec['latent_dim'];steps=spec['checkpoints']
    require(type(d) is int and d>0 and type(steps) is list and steps
        and all(type(x) is int and x>0 for x in steps) and steps==sorted(set(steps)),'fixed latent/checkpoint dimensions')
    c.keys(spec['assets'],ASSETS);c.keys(spec['phase_caps'],PHASES)
    for cap in spec['phase_caps'].values():
        c.keys(cap,KNOWN+UNKNOWN)
        require(all(type(cap[k]) is int and cap[k]>=0 for k in KNOWN)
            and all(cap[k] is None for k in UNKNOWN) and cap['callback_calls']==1,'supported phase caps, unknown costs null')
    bindings=closure();blockers=[];hashes={k:reference(v,bindings,blockers,k) for k,v in spec['assets'].items()}
    require(type(spec['trials']) is list and spec['trials'],'fixed nonempty trial denominator')
    ids=[]
    for trial in spec['trials']:
        c.keys(trial,('trial_id','shared','input','restore_binding_sha256','proposal_inputs'))
        require(type(trial['trial_id']) is str and trial['trial_id'],'trial identity');ids.append(trial['trial_id'])
        c.keys(trial['shared'],c.SHARED);c.sha(trial['restore_binding_sha256'])
        for h in trial['shared'].values():c.sha(h)
        for key in COMMON:
            if hashes[key] is not None:require(trial['shared'][key]==hashes[key],'same common assets/controller for all arms')
        ih=reference(trial['input'],bindings,blockers,trial['trial_id']+'.input')
        if ih is not None:
            inp=c.bound(trial['input']['path'],ih);c.keys(inp,('root','task_goal_sha256','restore_binding'))
            require(c.digest(inp['root'])==trial['shared']['root'] and c.digest(inp['root']['context'])==trial['shared']['history']
                and inp['task_goal_sha256']==trial['shared']['task_goal']
                and c.digest(inp['restore_binding'])==trial['restore_binding_sha256'],'exact root/full-history/task-goal/restore input')
        c.keys(trial['proposal_inputs'],c.ARMS)
        for arm,ref in trial['proposal_inputs'].items():reference(ref,bindings,blockers,trial['trial_id']+'.'+arm)
    require(len(set(ids))==len(ids),'no duplicate/replacement trial slot')
    if hashes['budget'] is not None:
        require(c.bound(spec['assets']['budget']['path'],hashes['budget'])==dict(schema=NAME+'.budget',phase_caps=spec['phase_caps']),
            'phase caps must equal bound common budget asset')
    if hashes['learned_pairing'] is not None:
        pairing=c.bound(spec['assets']['learned_pairing']['path'],hashes['learned_pairing'])
        c.keys(pairing,('schema','method','prior_checkpoint_sha256','model_sha256','encoder_sha256','normalizer_sha256','observation_sha256'))
        require(pairing['schema']==NAME+'.pairing' and pairing['method'] in ('HiLeWM_hierarchical_adaptation','SAGE_generator_adaptation'),
            'real learned identity; random/fallback is not learned prior')
        for key in ('model','encoder','normalizer','observation'):
            if hashes[key] is not None:require(pairing[key+'_sha256']==hashes[key],'learned coordinate pairing')
        if hashes['learned_checkpoint'] is not None:require(pairing['prior_checkpoint_sha256']==hashes['learned_checkpoint'],'learned checkpoint pairing')
    if hashes['controller'] is not None:
        controller=c.bound(spec['assets']['controller']['path'],hashes['controller'])
        c.keys(controller,('schema','bindings','checkpoints','latent_dim','action_low','action_high'))
        require(controller['schema']==NAME+'.controller' and controller['checkpoints']==steps and controller['latent_dim']==d,'common controller horizon/space')
        c.keys(controller['bindings'],tuple(k for k in ASSETS if k not in ('controller','learned_checkpoint','learned_pairing','arbitrary_rule','compiled_rule')))
        for key,value in controller['bindings'].items():
            if hashes[key] is not None:require(value==hashes[key],'unified controller implementation/assets binding')
        low,high=controller['action_low'],controller['action_high']
        require(type(low) is list and low and len(low)==len(high),'explicit raw action domain')
        c.vector(low,len(low));c.vector(high,len(low));require(all(a<b for a,b in zip(low,high)),'ordered raw bounds')
    return dict(schema=NAME+'.prepared',spec=deepcopy(spec),bindings=bindings,asset_hashes=hashes,
        blockers=blockers,status='blocked' if blockers else 'engineering_ready_NOT_authorized',
        trial_denominator=len(ids),arm_denominator=3*len(ids),physical_execution_authorized=False,scientific_pass=False)


def verify(prepared,expected):
    c.sha(expected);require(c.digest(prepared)==expected,'external prepared SHA')
    require(prepare(prepared['spec'])==prepared,'prepared source/assets/config drift');return prepared


@dataclass(frozen=True)
class Adapter:
    """MAIN audited code only. One controller callable, not three arm controllers.

    restore(input,work); proposers[arm](input,proposal_input,work);
    controller(input,targets,checkpoints,work); native(input,tracking,work).
    Each underlying model/env primitive must use work.call(units, callable).
    preflight() returns CUDA/frozen flags plus full parameter/buffer state hashes.
    """
    backend:str
    bindings:dict
    restore:object
    proposers:dict
    controller:object
    native:object
    preflight:object


class Journal:
    """Exclusive append+fsync sink. A write failure stops dispatch, never retries."""
    def __init__(self,path):
        path=Path(path).absolute()
        require(path.parent.resolve()==ROOT/'results' and path.name.startswith(NAME),'exclusive runner journal under results')
        self.file=path.open('xb');self.previous='0'*64;self.seq=0;self.poisoned=False
        try:
            fd=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
        except BaseException:self.file.close();raise

    def __call__(self,event):
        require(not self.poisoned,'journal poisoned; no retry')
        record=dict(seq=self.seq,previous=self.previous,event=event);h=c.digest(record)
        try:
            data=c.encode(dict(record,sha256=h))+b'\n'
            require(len(data)<=4*1024**2,'bounded event')
            require(self.file.write(data)==len(data),'short journal write');self.file.flush();os.fsync(self.file.fileno())
        except BaseException:self.poisoned=True;raise
        self.seq+=1;self.previous=h

    def close(self):self.file.close()


class Work:
    """Cooperative host-entry accounting, NOT hardware profiler or CUDA sync."""
    def __init__(self,caps,emit):
        self.caps=caps;self.emit=emit;self.current=None;self.accounts={};self.serial=0;self.secondary_errors=[]
        self.poisoned=False;self.primary=None

    def persist(self,event):
        try:self.emit(event)
        except BaseException as exc:
            self.poisoned=True
            if self.primary is None:self.primary=exc
            raise

    def phase(self,slot,phase):
        self.current=(slot,phase);self.accounts[self.current]=dict(reserved=vector(),returned=vector(),pending={})

    def guard(self,ok,message):
        if not ok:
            self.poisoned=True;error=ValueError(message)
            if self.primary is None:self.primary=error
            raise error

    def call(self,units,fn):
        self.guard(not self.poisoned and self.current is not None and units and set(units)<=set(KNOWN)
            and all(type(v) is int and v>=0 for v in units.values()) and any(v>0 for v in units.values()),'explicit supported nonzero primitive units')
        slot,phase=self.current;a=self.accounts[self.current];cap=self.caps[phase]
        self.guard(all(a['reserved'][k]+units.get(k,0)<=cap[k] for k in KNOWN),'phase cap before dispatch')
        token=self.serial;self.serial+=1
        self.persist(dict(kind='reserve_intent',slot=slot,phase=phase,token=token,units=units,computation_observed=False))
        for k,v in units.items():a['reserved'][k]+=v
        a['pending'][token]=deepcopy(units)
        self.persist(dict(kind='entry_boundary',slot=slot,phase=phase,token=token,completion=None))
        try:value=fn()
        except BaseException as exc:
            self.poisoned=True
            if self.primary is None:self.primary=exc
            try:self.persist(dict(kind='failed_return',slot=slot,phase=phase,token=token,error=str(exc),completed_units=None))
            except BaseException as log_error:self.secondary_errors.append(dict(type=type(log_error).__name__,message=str(log_error)))
            raise
        self.guard(not self.poisoned,'nested primitive failed; no callback may hide/retry it')
        self.persist(dict(kind='host_return',slot=slot,phase=phase,token=token,units=units,
            scope='host callable returned; not synchronized device timing'))
        for k,v in units.items():a['returned'][k]+=v
        del a['pending'][token];return value


def adapter_binding(adapter,p,backend):
    require(type(adapter) is Adapter and adapter.backend==backend and adapter.bindings==p['asset_hashes'],'exact adapter/backend/assets binding')
    c.keys(adapter.proposers,c.ARMS)
    callbacks=(adapter.restore,adapter.controller,adapter.native,adapter.preflight,*adapter.proposers.values())
    require(all(callable(f) for f in callbacks),'concrete callbacks required')
    if backend=='HOST_CUDA':
        for callback in callbacks:
            source=inspect.getsourcefile(callback)
            require(source is not None and file_sha(source)==p['asset_hashes']['adapter_source'],
                'every production callback must come from the bound MAIN adapter source file')


def execute_core(p,adapter,emit,fixture):
    spec=p['spec'];work=Work(spec['phase_caps'],emit);rows=[];errors=[];before=None;after=None
    for trial in spec['trials']:
        for arm in c.ARMS:rows.append(dict(trial_id=trial['trial_id'],arm=arm,status='not_dispatched',
            target=None,predicted=None,observed=None,raw_actions=None,native_success=None,error=None))
    try:
        emit(dict(kind='header',prepare_sha256=c.digest(p),fixture_only=fixture,denominator=len(rows)))
        if not fixture:
            before=adapter.preflight();c.keys(before,('cuda_only','frozen_eval','parameters','buffers'))
            require(before['cuda_only'] is before['frozen_eval'] is True,'actual adapter must attest CUDA-only frozen eval')
            c.sha(before['parameters']);c.sha(before['buffers'])
        for slot,row in enumerate(rows):
            trial=spec['trials'][slot//3];arm=row['arm'];shared=trial['shared'];row['status']='attempted'
            inp=c.bound(trial['input']['path'],trial['input']['sha256'])
            for phase in PHASES:
                work.phase(slot,phase)
                def call(fn):return work.call({'callback_calls':1},fn)
                if phase=='restore':
                    restored=call(lambda:adapter.restore(deepcopy(inp),work))
                    require(restored==dict(shared_sha256=c.digest(shared),restore_binding_sha256=trial['restore_binding_sha256']),
                        'exact same native start/history/warmup/goal/renderer restore; approximation forbidden')
                    require(work.accounts[(slot,phase)]['returned']['restore_calls']==1,'meter exact restore once per arm')
                elif phase=='proposal':
                    ref=trial['proposal_inputs'][arm];proposal_input=c.bound(ref['path'],ref['sha256'])
                    proposal=call(lambda:adapter.proposers[arm](deepcopy(inp),deepcopy(proposal_input),work))
                    c.keys(proposal,('target','source_sha256','learned_receipt','witness'))
                    targets=proposal['target'];require(len(targets)==len(spec['checkpoints']),'proposal checkpoint denominator')
                    for t in targets:c.vector(t,spec['latent_dim'])
                    source=('arbitrary_rule','learned_checkpoint','compiled_rule')[c.ARMS.index(arm)]
                    require(proposal['source_sha256']==p['asset_hashes'][source],'original proposal source, no relabeling')
                    if arm==c.ARMS[1]:
                        c.learned_pairing(proposal['learned_receipt'],dict(target=targets),shared,list(p['asset_hashes'].values()))
                        require(proposal['learned_receipt']['prior_checkpoint_sha256']==p['asset_hashes']['learned_checkpoint']
                            and work.accounts[(slot,phase)]['returned']['generator_calls']>0,'meter actual learned generator, no fallback')
                    else:require(proposal['learned_receipt'] is None,'non-learned arm cannot impersonate prior')
                    if arm==c.ARMS[2]:c.compiled_pairing(proposal['witness'],dict(target=targets),shared,spec['checkpoints'])
                    else:require(proposal['witness'] is None,'non-compiled arm witness substitution')
                    row['target']=deepcopy(targets);emit(dict(kind='proposal_saved',slot=slot,payload=proposal))
                elif phase=='controller':
                    tracking=call(lambda:adapter.controller(deepcopy(inp),deepcopy(targets),list(spec['checkpoints']),work))
                    c.keys(tracking,('shared_sha256','targets_sha256','predicted','observed','raw_actions'))
                    require(tracking['shared_sha256']==c.digest(shared) and tracking['targets_sha256']==c.digest(targets),'same controller inputs/targets')
                    c.deviations(targets,tracking['observed'],spec['latent_dim'])
                    require(all(x is not None for x in tracking['observed']),'completed tracking needs actual observations')
                    if tracking['predicted'] is not None:c.deviations(targets,tracking['predicted'],spec['latent_dim'])
                    actions=tracking['raw_actions'];controller=c.bound(spec['assets']['controller']['path'],p['asset_hashes']['controller'])
                    require(len(actions)==spec['checkpoints'][-1],'fixed raw-action horizon')
                    for a in actions:
                        c.vector(a,len(controller['action_low']))
                        require(all(l<=x<=h for l,x,h in zip(controller['action_low'],a,controller['action_high'])),'no extra clip/normalization')
                    require(work.accounts[(slot,phase)]['returned']['physics_steps']==len(actions),'meter executing branch physical steps')
                    for key in ('predicted','observed','raw_actions'):row[key]=deepcopy(tracking[key])
                    row['deviation']=c.deviations(targets,tracking['observed'],spec['latent_dim'])
                    emit(dict(kind='tracking_saved',slot=slot,payload=tracking))
                else:
                    native=call(lambda:adapter.native(deepcopy(inp),deepcopy(tracking),work))
                    if native is not None:
                        c.keys(native,('success','evaluator_sha256','task_goal_sha256'))
                        require(type(native['success']) is bool and native['evaluator_sha256']==shared['native_success_evaluator']
                            and native['task_goal_sha256']==shared['task_goal']
                            and work.accounts[(slot,phase)]['returned']['native_evaluator_calls']>0,'native success independent of latent deviation')
                        row['native_success']=native['success']
                    emit(dict(kind='native_saved',slot=slot,payload=native))
            row['status']='completed'
    except BaseException as exc:
        if work.primary is not None:exc=work.primary
        errors.append(dict(type=type(exc).__name__,message=str(exc)))
        for row in rows:
            if row['status']=='attempted':row.update(status='failed',error=errors[0])
    finally:
        if not fixture and before is not None:
            try:
                after=adapter.preflight()
                require(after==before and after['cuda_only'] is after['frozen_eval'] is True,'model parameters/buffers/frozen/CUDA state changed')
            except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
        try:verify(p,c.digest(p))
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
    costs=[]
    for slot in range(len(rows)):
        for phase in PHASES:
            account=work.accounts.get((slot,phase))
            costs.append(dict(slot=slot,phase=phase,planned_upper=deepcopy(spec['phase_caps'][phase]),
                status='not_dispatched' if account is None else 'pending_or_failed' if account['pending'] else 'host_returns_observed',
                **(dict(reserved=None,returned=None,pending=None) if account is None else deepcopy(account))))
    result=dict(schema=NAME+'.report',status='failed_closed' if errors else 'callbacks_completed_NOT_scientific_pass',
        fixture_only=fixture,rows=rows,errors=errors,secondary_errors=work.secondary_errors,
        trial_denominator=p['trial_denominator'],arm_denominator=len(rows),
        costs=costs,model_before=before,model_after=after,
        full_cost_instrumentation_complete=False,scope='cooperative callback/primitive host entries; unhooked work unknown',
        P1_admission=False,scientific_pass=False,physical_execution_authorized=False)
    try:emit(dict(kind='terminal',payload=result))
    except BaseException as exc:result['status']='failed_closed';result['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
    return result


def run_fixture(p,expected,adapter,emit):
    verify(p,expected);require(not p['blockers'],'missing assets/controller: no callback allowed')
    adapter_binding(adapter,p,'RAM_METADATA');return execute_core(p,adapter,emit,True)


def run(p,expected,adapter,journal_path,authority_path,authority_sha):
    verify(p,expected);require(not p['blockers'],'missing assets/controller: no callback allowed')
    adapter_binding(adapter,p,'HOST_CUDA')
    a=c.bound(authority_path,authority_sha)
    c.keys(a,('schema','reviewer','prepared_sha256','P1_passed','freshness_passed','physical_run_authorized'))
    require(a==dict(schema=NAME+'.MAIN_authority',reviewer='MAIN',prepared_sha256=expected,
        P1_passed=True,freshness_passed=True,physical_run_authorized=True)
        and all(a[k] is True for k in ('P1_passed','freshness_passed','physical_run_authorized')),
        'P1/freshness/physical MAIN release missing')
    sink=Journal(journal_path)
    try:return execute_core(p,adapter,sink,False)
    finally:sink.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify'))
    p.add_argument('--input',required=True);p.add_argument('--sha256',required=True);a=p.parse_args(argv)
    value=c.bound(a.input,a.sha256)
    # CLI authenticates file BYTES first (including whitespace/newline). The
    # in-memory API/MAIN authority binds canonical prepared content separately.
    out=prepare(value) if a.command=='prepare' else verify(value,c.digest(value))
    print(c.encode(out).decode());return 0


if __name__=='__main__':raise SystemExit(main())
