"""MAIN-only exact known-trajectory replay with native-default target pairs.

Saved prepare/verify and mock kernels do not import physics or neural packages.
Never open a sourcepack, certificate or witness; replay sealed literal placement.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_constructive_plan_physical_20261027_v1 as physical
from experiments.spectral_reachability import p1_constructive_offline_fidelity_cuda_20261027_v1 as saved

base,enc,rgb=saved.base,saved.enc,saved.rgb
ROOT=base.ROOT
NAME='p1_constructive_live_overlay_fidelity_20261027_v1'
AUTH='MAIN_LIVE_OVERLAY_REPLAY_ONLY'
ACTUAL_SHA='d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300'
PHYSICAL_SHA='9dea58dfc6418dd01bf9ab28de6d712189ca1154a5cf29b165efa4b8ef65ea96'
SELECTED=(('warmup',5),('warmup',10),('plan',5),('plan',10))
ALL=tuple((p,s) for p in ('warmup','plan') for s in range(11))
DEFAULT_POSE=[256.,256.,float(np.pi/4)]
FLOOR,CAP,FINAL=200*1024**2,8*1024**2,256*1024
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical


def contract():
    return dict(branches=4,selected_pairs=16,selected_images=32,off_full_trajectory_images=88,
        caps=dict(reset=8,public_step=80,physics_step=816,set_state=16,goal_render=8,explicit_render=104,goal_bind=4),
        wall_seconds=600,output_cap_bytes=CAP,reserve_bytes=FLOOR,no_retry=True,
        pair_order=['off','on'],selected=[list(x) for x in SELECTED],goal_pose=DEFAULT_POSE,
        goal_pose_scope='source-derived native default, NOT custom task goal marker; old artifact did not record goal_pose',
        intervention='with_target only during on-render; restore False before next operation',
        recipe='original two resets + sealed literal placement, zero warmup10, original goal attachment, literal raw10',
        snapshot_scope='all space body dynamics/forces, space parameters, native state/contact/action, RNG, goal state/RGB/pose; renderer screen/color caches excluded',
        new_neural=0,training=False,P1_admission=False,independent_confirmation=False)


def single(events,kind,branch):
    rows=[e for e in events if e['kind']==kind and e.get('branch')==branch]
    require(len(rows)==1,'one original '+kind);return rows[0]


def assemble(report,events,p):
    """Extract only approved evaluator recipe and all88 off receipts, no outcomes selecting rows."""
    require(report['schema']==physical.NAME+'.report' and report['status']=='physical_calibration_recorded' and
        report['errors']==[] and events[0]['prepare']==p and events[-1]['status']==report['status'] and
        events[-1]['counts']==report['counts'],'complete original execution')
    expected=[(i,a) for i in range(2) for a in base.ARMS]
    require([(r['root_slot'],r['arm']) for r in report['rows']]==expected and
        [(r['root_slot'],r['arm']) for r in p['plans']]==expected,'original fixed4 known branches')
    require(report['contract']==physical.contract(4,600)==p['contract'],'original off execution contract')
    counts={k:dict(attempted=0,completed=0) for k in report['counts']};stack=[]
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            k=e['resource'];attempt=e['kind']=='operation_attempted';counts[k]['attempted' if attempt else 'completed']+=1
            if attempt:stack.append(k)
            else:require(stack and stack.pop()==k,'nested original operation receipts')
    require(not stack and counts==report['counts']=={k:dict(attempted=v,completed=v) for k,v in p['contract']['caps'].items()},'original actual full resource denominator')
    branches=[]
    for b,(row,plan) in enumerate(zip(report['rows'],p['plans'],strict=True)):
        require(row['full_horizon'] and row['own_returned']==row['own_attempted']==10 and row['warmup_returned']==10 and
            row['first_error'] is None and row['secondary_errors']==[] and row['PNG']==22,'original full branch retained')
        require(single(events,'branch_started',b)['plan']==plan and
            plan['selection_digest']==row['selection_digest']==physical.source.digest({k:v for k,v in plan.items() if k!='selection_digest'}),'original command seal')
        physical.commands(plan['commands'])
        reset=[e for e in events if e['kind']=='operation_attempted' and e.get('branch')==b and e['resource']=='reset']
        require(len(reset)==2 and reset[0]['kind_of_reset']=='original_seed' and reset[1]['kind_of_reset']=='placement' and reset[1]['seed']==0,'exact original reset recipe')
        place=single(events,'placement_request',b);attached=single(events,'goal_attached',b)
        receipts=[e['receipt'] for e in events if e['kind']=='fresh_observation' and e.get('branch')==b]
        require([(r['phase'],r['step']) for r in receipts]==list(ALL),'all22 original off receipts')
        guards=[e for e in events if e['kind']=='guard_receipt' and e.get('branch')==b]
        require(len(guards)==22 and all(g['accepted'] and g['render_preserved_dynamics'] for g in guards),'original guards')
        transitions=[e for e in events if e['kind']=='returned_transition' and e.get('branch')==b]
        require([(e['phase'],e['step']) for e in transitions]==[(phase,s) for phase in ('warmup','plan') for s in range(1,11)] and
            [e['command'] for e in transitions if e['phase']=='plan']==plan['commands'],'all original transitions/commands')
        require(attached['goal']==row['goal'] and all(e['returned_goal_state']==row['goal'] and
            e['returned_goal_rgb_sha256']==attached['goal_rgb_sha256'] for e in transitions if e['phase']=='plan'),'goal attachment chain')
        branches.append(dict(branch=b,source_slot=plan['root_slot'],arm=plan['arm'],commands=plan['commands'],
            selection_digest=plan['selection_digest'],reset_seed=reset[0]['seed'],placement_seed=0,
            original_goal=place['original_goal'],requested=place['requested'],goal=attached['goal'],
            goal_rgb_sha256=attached['goal_rgb_sha256'],root_sha256=p['source_slots'][plan['root_slot']]['root_sha256'],
            receipts=receipts,transitions=transitions))
    return branches


def source_files():
    return base.source_files()+[Path(__file__),Path(saved.__file__),Path(rgb.__file__),Path(physical.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build(physical_report,physical_sha):
    path=Path(physical_report).absolute();require(physical_sha==ACTUAL_SHA and sha(path)==physical_sha,'approved original4 physical report SHA')
    require(sha(physical.__file__)==PHYSICAL_SHA,'original frozen physical source')
    report=read(path);binding=read(path.parent/'execution_binding.json')
    require(sha(binding['prepared'])==binding['prepare_sha256'],'original prepare seal');p=read(binding['prepared'])
    jp=path.parent/'journal.jsonl';events=rgb.journal(jp,report['journal_sha256']);branches=assemble(report,events,p)
    bindings={str(path):physical_sha,str(jp):report['journal_sha256'],str(path.parent/'execution_binding.json'):sha(path.parent/'execution_binding.json'),
        str(Path(binding['prepared']).absolute()):binding['prepare_sha256']}
    for file,h in p['bindings'].items():
        # Do not hash/open sourcepack, certificates, source journal or witness artifacts.
        if Path(file).suffix in ('.py','.md'):
            require(sha(file)==h,'original source seal: '+file);bindings[str(Path(file).absolute())]=h
    for item in branches:
        slot=item['source_slot'];matches=[(f,h) for f,h in p['bindings'].items() if Path(f).name==f's{slot}_blind_inputs.npz']
        require(len(matches)==1,'explicit old blind goal asset only');file,h=matches[0]
        require(sha(file)==h,'old blind input SHA')
        with np.load(file,allow_pickle=False) as z:goal_rgb=z['goal_rgb'].copy()
        require(goal_rgb.dtype==np.uint8 and goal_rgb.shape==(224,224,3) and
            physical.builder.image_sha(goal_rgb)==item['goal_rgb_sha256'],'exact original goal attachment RGB')
        item['goal_rgb_file']=str(Path(file).absolute());item['goal_rgb_file_sha256']=h;bindings[item['goal_rgb_file']]=h
        for r in item['receipts']:
            f=rgb.local(path.parent,r['png']);rgb.png(f,r['file_sha256'],r['pixel_sha256']);bindings[str(f)]=r['file_sha256']
    bindings.update({str(Path(f).resolve()):sha(f) for f in source_files()})
    require(bindings[str(physical.builder.ENV)]==physical.builder.ENV_SHA,'native default pose source authority')
    return dict(schema=NAME+'.prepare',arguments=dict(physical_report=str(path),physical_sha=physical_sha),
        branches=branches,bindings=bindings,contract=contract())


def prepare(output,**args):
    p=build(**args);value=canonical(p);path=Path(output).absolute()
    require(not path.exists() and len(value)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(value),'exclusive small prepare/200MiB floor')
    with path.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(value))


def verify(path,expected):
    require(sha(path)==expected,'external new prepare SHA');p=read(path)
    require(p==build(**p['arguments']),'complete original recipe/PNG/source reconstruction');return p


def numeric(value):
    a=np.ascontiguousarray(value,dtype=np.float64)
    require(np.isfinite(a).all(),'finite snapshot');return dict(shape=list(a.shape),bytes=a.tobytes().hex())


def snapshot(body):
    def dyn(b):return {k:numeric(getattr(b,k)) for k in ('position','velocity','force','angle','angular_velocity','torque','center_of_gravity')}
    return dict(native=physical.builder.snapshot(body),bodies=[dyn(b) for b in body.space.bodies],
        space={k:numeric(getattr(body.space,k)) for k in ('gravity','damping','iterations','collision_slop','collision_bias','collision_persistence')},
        goal_pose=numeric(body.goal_pose),goal_state=numeric(body.goal_state),goal_rgb=physical.builder.image_sha(body._goal),
        latest_action=None if body.latest_action is None else numeric(body.latest_action),contacts=int(body.n_contact_points),
        rng=deepcopy(body.rng.bit_generator.state))


def default_pose(body):
    require(np.array_equal(np.asarray(body.goal_pose),np.asarray(DEFAULT_POSE)) and
        np.array_equal(body.variation_space['goal']['position'].value,[256.,256.]) and
        float(body.variation_space['goal']['angle'].value)==DEFAULT_POSE[2] and
        int(body.variation_space['rendering']['render_goal'].value)==1,'certified native default goal_pose/render_goal; never replace')


class Store(base.Store):
    @property
    def path(self):return self.output
    def check(self,size=4096):
        require(sum(f.stat().st_size for f in self.output.iterdir() if f.is_file())+size<=CAP-FINAL and
            shutil.disk_usage(self.output).free>=FLOOR+size,'predispatch8MiB/200MiB')
    def emit(self,kind,**fields):self.event(dict(kind=kind,**deepcopy(fields)))
    def frame(self,name,a):self._write(name,saved.png_bytes(a));return sha(self.output/name)


class Ops(physical.Ops):
    in_pair=False
    def call(self,kind,fn,**fields):
        require(not self.in_pair or kind=='explicit_render','NO reset/setter/step/physics/goal binding in pair')
        return super().call(kind,fn,**fields)


def pair(env,ops,off_callback,on_callback,snap=None,check=None):
    """Fixed off then on; retain each image before checking purity; always restore flag."""
    snap=snapshot if snap is None else snap;check=default_pose if check is None else check
    body=env.unwrapped;require(body.with_target is False,'off before pair');check(body)
    before=snap(body);counts=deepcopy(ops.counts);ops.in_pair=True
    try:
        off=ops.call('explicit_render',env.render);off_callback(np.asarray(off).copy())
        require(snap(body)==before,'off render mutated snapshot');check(body)
        body.with_target=True
        on=ops.call('explicit_render',env.render);on_callback(np.asarray(on).copy())
        require(snap(body)==before,'on render mutated snapshot');check(body)
    finally:body.with_target=False;ops.in_pair=False
    require(snap(body)==before and all(ops.counts[k]==v for k,v in counts.items() if k!='explicit_render'),'pair unchanged, no hidden steps')
    return dict(before=before,after=snap(body),goal_pose=DEFAULT_POSE,order=['off','on'],snapshot_equal=True)


@contextmanager
def instrument(env,ops):
    """Count reset-internal renders/setters/space.step, including newly created spaces."""
    body=env.unwrapped;undo=[];spaces=[];resetting=False
    old_set,old_render,old_reset=body._set_state,body.render,env.reset
    def install():
        s=body.space
        if any(x[0] is s for x in spaces):return
        old=s.step;spaces.append((s,'step' in s.__dict__,old))
        s.step=lambda *a,**kw:ops.call('physics_step',lambda:old(*a,**kw))
    def setter(value):install();return ops.call('set_state',lambda:old_set(value))
    def render():return ops.call('goal_render',old_render) if resetting else old_render()
    def reset(*a,**kw):
        nonlocal resetting
        resetting=True
        try:return old_reset(*a,**kw)
        finally:resetting=False
    try:
        for obj,key,value in ((body,'_set_state',setter),(body,'render',render),(env,'reset',reset)):
            undo.append((obj,key,key in obj.__dict__,getattr(obj,key)));setattr(obj,key,value)
        yield
    finally:
        for obj,owned,old in reversed(spaces):
            if owned:obj.step=old
            else:delattr(obj,'step')
        for obj,key,owned,old in reversed(undo):
            if owned:setattr(obj,key,old)
            else:delattr(obj,key)


def branch(env,item,row,ops,store):
    body=env.unwrapped;b=item['branch'];ops.where=dict(branch=b,source_slot=item['source_slot'],arm=item['arm'])
    old={(r['phase'],r['step']):r for r in item['receipts']}
    def capture(phase,step,obs):
        default_pose(body);key=(phase,step);r=old[key];pairrow=dict(branch=b,source_slot=item['source_slot'],arm=item['arm'],phase=phase,step=step,status='attempted',views={})
        if key in SELECTED:row['pairs'].append(pairrow)
        def off(a):
            name=r['png'];h=store.frame(name,a)
            fresh=dict(phase=phase,step=step,png=name,file_sha256=h,pixel_sha256=physical.builder.image_sha(a),
                state=np.asarray(body._get_obs()).tolist(),snapshot=physical.builder.snapshot(body),observation_state=np.asarray(obs['state']).tolist())
            row['last_observation']=fresh;store.emit('off_receipt',branch=b,receipt=fresh)
            require(fresh==r,'off full trajectory state/snapshot/PNG exact mismatch; retained')
            row['off_exact']+=1
            pairrow['views']['off']=dict(file=name,sha256=h,pixel_sha256=fresh['pixel_sha256'])
        def on(a):
            name=f'b{b}_{phase}_{step:02}_on.png';h=store.frame(name,a)
            pairrow['views']['on']=dict(file=name,sha256=h,pixel_sha256=physical.builder.image_sha(a))
            store.emit('on_frame',branch=b,phase=phase,step=step,receipt=pairrow['views']['on'])
        if key in SELECTED:
            proof=pair(env,ops,off,on);pairrow.update(status='paired',proof=proof);store.emit('pair_completed',receipt=pairrow)
        else:
            before=snapshot(body);a=ops.call('explicit_render',env.render);off(np.asarray(a).copy())
            require(snapshot(body)==before,'off render purity')
    _,info=ops.call('reset',lambda:env.reset(seed=item['reset_seed']))
    require(np.array_equal(info['goal_state'],item['original_goal']),'original sampled goal exact')
    obs,info=ops.call('reset',lambda:env.reset(seed=item['placement_seed'],options=dict(state=np.asarray(item['requested']),goal_state=np.asarray(item['original_goal']))))
    capture('warmup',0,obs)
    transitions={(e['phase'],e['step']):e for e in item['transitions']}
    def advance(phase,step,command):
        obs,reward,terminated,truncated,info=ops.call('public_step',lambda:env.step(command.copy()))
        row['public_returned']+=1
        actual=dict(state=np.asarray(obs['state']).tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated))
        row['last_transition']=dict(phase=phase,step=step,**actual);store.emit('returned_transition',branch=b,**row['last_transition'])
        expected=transitions[(phase,step)]
        require(actual=={k:expected[k] for k in actual},'original state/reward/stop exact')
        capture(phase,step,obs)
        require(not terminated and not truncated,'original full horizon required; no early replacement')
        if phase=='plan':require(np.array_equal(info['goal_state'],item['goal']) and physical.builder.image_sha(info['goal'])==item['goal_rgb_sha256'],'original task goal unchanged')
        return obs
    for step in range(1,11):obs=advance('warmup',step,np.zeros(2,np.float32))
    require(physical.source.root_hash(physical.source.vector(obs['state']),physical.source.vector(item['goal']))==item['root_sha256'],'old root identity')
    with np.load(item['goal_rgb_file'],allow_pickle=False) as z:goal_rgb=z['goal_rgb'].copy()
    before=snapshot(body);pose=deepcopy(body.goal_pose)
    ops.call('goal_bind',lambda:physical.builder.attach_goal(body,np.asarray(item['goal']),goal_rgb))
    # Original attachment changes goal_state/_goal, never goal_pose or dynamics.
    require(np.array_equal(body.goal_pose,pose) and snapshot(body)['bodies']==before['bodies'],'original attachment must preserve dynamics/default pose')
    capture('plan',0,obs)
    for step,command in enumerate(physical.commands(item['commands']),1):obs=advance('plan',step,command)
    require(row['off_exact']==22 and len(row['pairs'])==4 and all(p['status']=='paired' for p in row['pairs']),'full branch denominator')
    row['status']='replayed_and_paired'


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'explicit MAIN physics authorization');p=verify(prepared,expected);store=Store(output);ops=Ops(p,store)
    rows=[dict(branch=i['branch'],source_slot=i['source_slot'],arm=i['arm'],commands=i['commands'],status='not_started',off_exact=0,public_returned=0,pairs=[],first_error=None,secondary_errors=[]) for i in p['branches']]
    errors=[];source_ok=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected));store.emit('run_started',contract=p['contract'])
        for item,row in zip(p['branches'],rows,strict=True):
            env=None
            try:
                store.check();env=factory();physical.builder.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','same native renderer')
                row['status']='running'
                with instrument(env,ops):branch(env,item,row,ops,store)
            except BaseException as exc:physical.retain(row,exc,'branch')
            finally:
                if env is not None:
                    try:env.close()
                    except BaseException as exc:physical.retain(row,exc,'close')
                store.emit('branch_closed',receipt=row)
            if row['status']!='replayed_and_paired':break
    except BaseException as exc:errors.append(enc.error('execution',exc))
    finally:
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:errors.append(enc.error('final_source',exc))
        complete=source_ok and not errors and all(r['status']=='replayed_and_paired' for r in rows) and ops.counts=={k:dict(attempted=v,completed=v) for k,v in contract()['caps'].items()}
        try:store.emit('run_finished',counts=ops.counts,complete=complete);store.close()
        except BaseException as exc:errors.append(enc.error('journal_finalize',exc));complete=False
        report=dict(schema=NAME+'.report',status='live_overlay_pairs_sealed_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,counts=ops.counts,errors=errors,source_verified=source_ok,
            contract=p['contract'],artifact_sha256={f.name:sha(f) for f in store.output.iterdir() if f.is_file()},
            native_goal_pose_evidence='frozen native defaults plus runtime checks; no historical goal_pose receipt',
            training_overlay_provenance='unknown',P1_admission=False,training=False,new_neural=0)
        store.put('report.json',report,final=True)
    require(complete,'partial retained, no retries: '+str(errors));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for k in ('physical-report','physical-sha','prepared','prepare-sha','output'):parser.add_argument('--'+k)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.output,physical_report=a.physical_report,physical_sha=a.physical_sha)
    elif a.command=='verify':r=dict(status='saved_verified_only',branches=len(verify(a.prepared,a.prepare_sha)['branches']))
    else:
        require(a.authorize_physics,'MAIN CLI physics authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()
