"""MAIN-only known-root saved-prefix feedback physical replay, never P1/P4 admission.

Own branch/lifecycle, frozen moving physical tools only; no branch monkeypatch.
New compiler consume layout is explicitly mapped, never inferred or executed here.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import io
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_constructive_moving_history_physical_20261027_v1 as base

NAME='moving_history_observed_suffix_physical_20261027_v1'
AUTH='MAIN_MOVING_HISTORY_SAVED_PREFIX_FEEDBACK_PHYSICS_ONLY'
BASE_SHA='44d90599eab881c773d5efa09177dfe6b395fa3f4eef41070340681fea5ec96e'
ORIGINAL_SHA='a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf'
ROOT,source,moving=base.ROOT,base.source,base.moving
require,sha,read,raw=base.require,base.sha,base.read,base.raw
Store,metric,retain=base.Store,base.metric,base.retain
ARMS=['line_search_GN','full10_conditional_D2P']
COMPILER='moving_history_observed_suffix_cuda_20261027_v1'
CONSUME_STATUS='observed_suffix_commands_verified_not_admitted'


class Ops(base.Ops):
    """Unchanged frozen billing with lexical activity, not cumulative deficit as lock."""
    def __init__(self,p,store):super().__init__(p,store);self.active=[]
    def call(self,kind,fn,**fields):
        self.active.append(kind)
        try:return super().call(kind,fn,**fields)
        finally:require(self.active.pop()==kind,'operation scope balanced')


@contextmanager
def native_hooks(env,ops,row):
    """New hook scope: failed goal_bind stays charged but cannot poison later branches."""
    body=env.unwrapped;undo=[];spaces=[];resetting=False
    old_set,old_render,old_reset=body._set_state,body.render,env.reset
    def outside_goal():require('goal_bind' not in ops.active,'no physics/render during offline goal bind')
    def install():
        space=body.space
        if any(x[0] is space for x in spaces):return
        old=space.step;spaces.append((space,'step' in space.__dict__,old))
        def physics(*a,**kw):outside_goal();return ops.call('physics_step',lambda:old(*a,**kw))
        space.step=physics
    def setter(value):
        require(resetting,'post-reset state setter forbidden');install();return ops.call('set_state',lambda:old_set(value))
    def render():outside_goal();return ops.call('goal_render',old_render) if resetting else old_render()
    def reset(*a,**kw):
        nonlocal resetting
        resetting=True
        try:return old_reset(*a,**kw)
        finally:resetting=False
    try:
        for obj,k,v in ((body,'_set_state',setter),(body,'render',render),(env,'reset',reset)):
            undo.append((obj,k,k in obj.__dict__,getattr(obj,k)));setattr(obj,k,v)
        yield
    except BaseException as exc:retain(row,exc,'branch')
    finally:
        for obj,owned,old in reversed(spaces):
            try:
                if owned:obj.step=old
                else:delattr(obj,'step')
            except BaseException as exc:retain(row,exc,'restore_space')
        for obj,k,owned,old in reversed(undo):
            try:
                if owned:setattr(obj,k,old)
                else:delattr(obj,k)
            except BaseException as exc:retain(row,exc,'restore_hook')


def key(row):
    require(type(row['root_slot']) is int and row['root_slot'] in (0,1),'literal original root slot')
    return row['root_slot'],row['arm'],row['selection_digest']


def contract(wall):
    c=base.contract(wall)
    c.update(scope='known-root saved-prefix feedback calibration; same4 branches, old suffix endpoint saved only',
        prefix='original raw5; exact all warmup0..10 and plan0..5 RGB/state/body snapshots BEFORE new suffix',
        P1_admission=False,P4_admission=False,separate_old_control_reexecution=False,poststate_restore=False)
    return c


def assemble(report,p,events):
    """Saved receipts only; no environment/model imports or outcome-based choices."""
    order=[(s,a) for s in (0,1) for a in ARMS]
    require(report['schema']==base.NAME+'.report' and report['status']=='moving_history_physical_recorded' and
        report['errors']==[] and report['P1_admission'] is False and
        p['schema']==base.NAME+'.prepare' and p['contract']==base.contract(p['contract']['wall_seconds']) and
        report['contract']==p['contract'] and report['sourcepack_sha256']==p['sourcepack_sha256'],'original recorded protocol')
    require(events[0]['kind']=='run_started' and events[0]['prepare']==p and events[-1]['kind']=='run_finished' and
        events[-1]['status']==report['status'] and events[-1]['counts']==report['counts'] and events[-1]['errors']==[] and
        events[-1]['event_sha256']==report['journal_chain_tail'],'original durable begin/end')
    require([(r['root_slot'],r['arm']) for r in report['rows']]==order and
        [(r['root_slot'],r['arm']) for r in p['plans']]==order,'ALL four original branches, never select roots')
    counts={k:dict(attempted=0,completed=0) for k in p['contract']['caps']};stack=[]
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            k=e['resource'];field='attempted' if e['kind']=='operation_attempted' else 'completed'
            counts[k][field]+=1
            if field=='attempted':stack.append(k)
            else:require(stack and stack.pop()==k,'balanced original nested operations')
    require(not stack and counts==report['counts']=={k:dict(attempted=v,completed=v) for k,v in p['contract']['caps'].items()},'full original resource census')
    paired=[]
    for index,(row,plan) in enumerate(zip(report['rows'],p['plans'],strict=True)):
        require(key(row)==key(plan) and row['status'] in ('horizon_completed','native_success') and
            row['first_error'] is None and row['secondary_errors']==[] and row['initial_exact'] is True and
            row['full_horizon'] is True and row['warmup_returned']==row['own_returned']==row['own_attempted']==10 and
            row['PNG']==22 and row['truncated'] is False,'original full-horizon paired record, no substitutes')
        scoped=[e for e in events if e.get('branch')==index]
        require(scoped and all(key(e)==key(row) for e in scoped),'original branch identity scope')
        starts=[e for e in scoped if e['kind']=='branch_started'];closed=[e for e in events if e['kind']=='branch_closed' and key(e['receipt'])==key(row)]
        require(len(starts)==len(closed)==1 and starts[0]['plan']==plan and closed[0]['receipt']==row,'original plan/row seals')
        observed=[e['receipt'] for e in scoped if e['kind']=='fresh_observation'];guards=[e for e in scoped if e['kind']=='guard_receipt']
        order22=[(ph,n) for ph in ('warmup','plan') for n in range(11)]
        require([(r['phase'],r['step']) for r in observed]==order22 and [(e['phase'],e['step']) for e in guards]==order22 and
            all(e['accepted'] is True and e['render_preserved_dynamics'] is True for e in guards),'all original frames/guards')
        require(all(r['state']==r['observation_state']==r['snapshot']['state'] for r in observed),'original observation-state bridge')
        warm=base.raw_history(np.asarray(starts[0]['raw_warmup'],np.float32))
        wr=[e for e in scoped if e['kind']=='warmup_return'];transitions=[e for e in scoped if e['kind']=='returned_transition' and e['phase']=='plan']
        require(starts[0]['raw_float32_bits']==warm.tobytes().hex() and [e['step'] for e in wr]==list(range(1,11)) and
            [e['raw_float32_bits'] for e in wr]==[c.tobytes().hex() for c in warm] and
            moving.exact(np.asarray([e['raw_action'] for e in wr],np.float32),warm),'original actual nonzero history')
        base.commands(plan['commands'])
        require([e['step'] for e in transitions]==list(range(1,11)) and [e['command'] for e in transitions]==plan['commands'] and
            all(not e['terminated'] and not e['truncated'] for e in transitions[:9]),'original raw commands/chronology')
        goals=[e for e in scoped if e['kind']=='goal_attached']
        require(len(goals)==1 and goals[0]['goal']==row['goal'] and goals[0]['snapshot']==observed[11]['snapshot'] and
            all(e['returned_goal_state']==row['goal'] and e['returned_goal_rgb_sha256']==goals[0]['goal_rgb_sha256'] for e in transitions),'original goal throughout')
        require(row['checkpoints']==[observed[11+n] for n in (0,5,10)] and observed[11]['state']==row['root_state'] and
            observed[-1]['state']==row['final_state'] and all(transitions[n-1]['state']==observed[11+n]['state'] for n in range(1,11)),'original root/prefix/terminal joins')
        paired.append(dict(root_slot=row['root_slot'],arm=row['arm'],original_selection_digest=row['selection_digest'],
            original_commands=deepcopy(plan['commands']),warmup=deepcopy(observed[:11]),prefix=deepcopy(observed[11:17]),
            goal=deepcopy(row['goal']),goal_rgb_sha256=goals[0]['goal_rgb_sha256'],
            old_terminal=deepcopy(observed[-1]),old_final_metrics=deepcopy(row['final_metrics']),all_frames=deepcopy(observed)))
    return paired


def load_original(path,expected):
    require(expected==ORIGINAL_SHA and sha(base.__file__)==BASE_SHA,'MAIN exact old report/frozen moving tools')
    bindings={}
    def bind(path,h):
        path=Path(path).resolve();require(sha(path)==h and (str(path) not in bindings or bindings[str(path)]==h),'saved original SHA')
        bindings[str(path)]=h;return path
    path=bind(path,expected);report=read(path);bp=bind(path.parent/'execution_binding.json',sha(path.parent/'execution_binding.json'));b=read(bp)
    pp=bind(b['prepared'],b['prepare_sha256']);p=base.verify(pp,b['prepare_sha256'])
    m=read(p['manifest']);data=base.load_source(m['sourcepack'],m['sourcepack_sha256'],m['source_prepared'],m['source_prepare_sha256'])
    bindings.update(p['bindings']);bindings.update(data['bindings'])
    events=base.journal(bind(path.parent/'journal.jsonl',report['journal_sha256']),report['journal_chain_tail'])
    paired=assemble(report,p,events)
    from PIL import Image
    for item in paired:
        for r in item['all_frames']:
            f=bind(base.flat(path.parent,r['png']),r['file_sha256'])
            require(f.stat().st_size<1024**2,'bounded original PNG')
            with Image.open(io.BytesIO(f.read_bytes())) as im:
                require(im.mode=='RGB' and im.size==(224,224),'native original RGB224');a=np.asarray(im).copy()
            require(a.dtype==np.uint8 and moving.image_sha(a)==r['pixel_sha256'],'PNG byte/pixel seals')
    data.update(paired=paired,original_prepared=p,original_report=report,bindings=bindings)
    return data


def selection(m,data):
    """Explicit pointers into BOTH approved handoff and raw own-consume; no layout guess."""
    extras={'original_physical_report','original_physical_report_sha256','compiler_authority'}
    require(m['schema']==NAME+'.selection' and extras<=set(m) and m['arms']==ARMS,'NEW fixed4 selection manifest')
    adapted={k:deepcopy(v) for k,v in m.items() if k not in extras};adapted['schema']=base.NAME+'.selection'
    rows,bindings=base.selection(adapted,data)
    a=m['compiler_authority'];fields={'source','source_sha256','report','report_sha256','consume','consume_sha256','consume_schema','consume_status','mappings'}
    require(set(a)==fields and len(a['mappings'])==4,'explicit reviewed NEW compiler source/report/own consume authority')
    for f,h in (('source','source_sha256'),('report','report_sha256'),('consume','consume_sha256')):
        path=str(Path(a[f]).resolve());require(sha(path)==a[h],'external NEW compiler '+f+' seal');bindings[path]=a[h]
    require(Path(a['source']).name==COMPILER+'.py' and
        a['report_sha256']!=ORIGINAL_SHA,'compiler authority is not physical executor/result')
    consumed=read(a['consume'])
    require(consumed['schema']==a['consume_schema']==COMPILER+'.consume' and consumed['status']==a['consume_status']==CONSUME_STATUS and
        consumed['errors']==[] and consumed['new_physics']==0 and consumed['P1_admission'] is False and consumed['P4_admission'] is False,
        'actual NEW own-consume schema/status and non-admission flags, no fabricated MAIN metadata')
    require([(p['source_slot'],p['arm']) for p in consumed['plans']]==[(s,arm) for s in (0,1) for arm in ARMS] and
        all(p['whole_run_report_sha256']==a['report_sha256'] for p in consumed['plans']),'all4 actual consumed plans bind NEW report SHA')
    for index,(row,old,mapping) in enumerate(zip(rows,data['paired'],a['mappings'],strict=True)):
        require(Path(row['compiler_report']).resolve()==Path(a['consume']).resolve() and row['compiler_report_SHA']==a['consume_sha256'] and
            row['compiler_schema']==COMPILER+'.consume' and row['compiler_status']==CONSUME_STATUS,
            'compiler mapping directly references raw own-consume, not MAIN-added wrapper status/blind')
        require(set(mapping)=={'commands_pointer','source_slot_pointer','blind_sha_pointer','arm_pointer'},'exact own-consume mapping')
        require(mapping==dict(commands_pointer=f'/plans/{index}/commands',source_slot_pointer=f'/plans/{index}/source_slot',
            blind_sha_pointer=f'/plans/{index}/blind_sha256',arm_pointer=f'/plans/{index}/arm'),'final own-consume literal plan pointers')
        slot=base.pointer(consumed,mapping['source_slot_pointer'])
        require(type(slot) is int and slot==row['root_slot'] and base.pointer(consumed,mapping['arm_pointer'])==row['arm'] and
            base.pointer(consumed,mapping['commands_pointer'])==row['commands'] and
            base.pointer(consumed,mapping['blind_sha_pointer'])==data['references'][slot]['blind_sha256'],'commands also literal own consume output')
        require((row['root_slot'],row['arm'])==(old['root_slot'],old['arm']) and
            moving.exact(base.commands(row['commands'])[:5],base.commands(old['original_commands'])[:5]),'ALL original prefix5 raw bytes unchanged')
    return rows,bindings


def material(manifest,expected):
    require(sha(manifest)==expected,'external NEW selection SHA');m=read(manifest)
    data=load_original(m['original_physical_report'],m['original_physical_report_sha256'])
    old=read(data['original_prepared']['manifest'])
    require(all(m[k]==old[k] for k in ('sourcepack','sourcepack_sha256','source_prepared','source_prepare_sha256','roots','arms')),'same source/root/goals/arms, no substitution')
    rows,bindings=selection(m,data);bindings.update(data['bindings']);bindings[str(Path(manifest).resolve())]=expected
    files=[Path(__file__),Path(base.__file__),Path(base.low.__file__),Path(base.low.builder.__file__),Path(moving.__file__),Path(source.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings.update({str(f.resolve()):sha(f) for f in files})
    return dict(schema=NAME+'.prepare',manifest=str(Path(manifest).resolve()),manifest_sha256=expected,plans=rows,
        contract=contract(m['wall_seconds']),bindings=bindings,sourcepack_sha256=m['sourcepack_sha256'],
        original_physical_report_sha256=ORIGINAL_SHA,paired_receipts_sha256=source.digest(data['paired']))


def prepare(manifest,expected,output):
    p=material(manifest,expected);value=raw(p);path=Path(output)
    require(len(value)<256*1024 and shutil.disk_usage(path.parent).free>=base.FLOOR+len(value),'small exclusive prepare +200MiB floor')
    with path.open('xb') as f:require(f.write(value)==len(value),'complete prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path.resolve()),prepare_sha256=sha(path),bytes=len(value))


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=read(path)
    require(p==material(p['manifest'],p['manifest_sha256']),'complete current closure reconstruction');return p


def paired_capture(env,obs,phase,step,row,ops,store,index,paired):
    frame,snap=base.capture(env,obs,phase,step,row,ops,store,index)
    if phase=='warmup' or step<=5:
        r=paired['warmup' if phase=='warmup' else 'prefix'][step];actual=row['last_observation']
        equal=(actual['file_sha256']==r['file_sha256'] and actual['pixel_sha256']==r['pixel_sha256'] and actual['snapshot']==r['snapshot'] and
            moving.exact(np.asarray(actual['state']),np.asarray(r['state'])) and actual['observation_state']==r['observation_state'])
        store.emit('paired_prefix_exact',**ops.where,phase=phase,step=step,accepted=equal,
            original_png=r['png'],original_file_sha256=r['file_sha256'],original_pixel_sha256=r['pixel_sha256'],new_png=actual['png'])
        require(equal,'same-prefix replay drift: image/state/body bits; new observation already retained')
    return frame,snap


def branch(env,ref,source_plan,plan,paired,row,ops,store,index):
    """Independent replay kernel; never call or patch base.branch/execute."""
    body=env.unwrapped;warm=base.raw_history(ref['raw_history']);cmd=base.commands(plan['commands']);old=base.commands(paired['original_commands'])
    require((plan['root_slot'],plan['arm'])==(paired['root_slot'],paired['arm']),'same fixed original branch')
    require(moving.exact(cmd[:5],old[:5]),'exact original prefix before ANY dispatch')
    row['selected_suffix_matches_original']=moving.exact(cmd[5:],old[5:])
    ops.where=dict(branch=index,root_slot=plan['root_slot'],arm=plan['arm'],selection_digest=plan['selection_digest'])
    store.emit('branch_started',**ops.where,plan=plan,original_selection_digest=paired['original_selection_digest'],raw_warmup=warm.tolist(),raw_float32_bits=warm.tobytes().hex())
    _,info=ops.call('reset',lambda:env.reset(seed=ref['slot']['reset_seed']));original_goal=source.vector(info['goal_state'])
    requested=source.sample_state(original_goal,list(body.block.shapes),dict(root_seed=ref['slot']['root_seed']),0,0,source_plan)
    store.emit('placement_request',**ops.where,original_goal=original_goal.tolist(),requested=requested.tolist(),placement_seed=0)
    obs,_=ops.call('reset',lambda:env.reset(seed=0,options=dict(state=requested.copy(),goal_state=original_goal.copy())))
    frame,snap=paired_capture(env,obs,'warmup',0,row,ops,store,index,paired);history=[frame];states=[source.vector(obs['state'])];dynamics=[snap];sent=[]
    for step,c in enumerate(warm,1):
        require(not body.eval_state(original_goal,source.vector(obs['state']))[0],'terminal before warmup dispatch')
        obs,reward,terminated,truncated,info=ops.call('public_step',lambda c=c:env.step(c.copy()),phase='warmup',step=step,raw_action=c.tolist())
        row['warmup_returned']+=1
        store.emit('returned_transition',**ops.where,phase='warmup',step=step,state=source.vector(obs['state']).tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated))
        frame,snap=paired_capture(env,obs,'warmup',step,row,ops,store,index,paired)
        require(moving.exact(np.asarray(body.latest_action),c),'actual raw warmup command')
        sent.append(np.asarray(body.latest_action).copy());store.emit('warmup_return',**ops.where,step=step,raw_action=c.tolist(),raw_float32_bits=c.tobytes().hex())
        require(not terminated and not truncated,'warmup stopped; never dispatch prefix/suffix')
        if step in (5,10):history.append(frame);states.append(source.vector(obs['state']));dynamics.append(snap)
    require(moving.exact(np.asarray(sent),warm) and moving.exact(np.asarray(history),ref['history']) and
        moving.exact(np.asarray(states),ref['states']) and dynamics==ref['dynamics'],'same moving source0/5/10')
    goal,goal_rgb=ref['goal'],ref['goal_rgb'];root=source.vector(obs['state'])
    require(source.root_hash(root,goal)==ref['root_sha256'] and goal.tolist()==paired['goal'] and moving.image_sha(goal_rgb)==paired['goal_rgb_sha256'],'same original root/goal')
    before=moving.snapshot(body);counts=deepcopy(ops.counts);pose=deepcopy(getattr(body,'goal_pose',None))
    ops.call('goal_bind',lambda:moving.attach_goal(body,goal,goal_rgb))
    require(moving.snapshot(body)==before and np.array_equal(getattr(body,'goal_pose',None),pose) and
        all(v==ops.counts[k] for k,v in counts.items() if k!='goal_bind'),'offline goal attachment, no state restore/extra physics/render')
    frame,_=paired_capture(env,obs,'plan',0,row,ops,store,index,paired)
    require(moving.exact(frame,history[-1]) and np.array_equal(body.goal_state,goal) and np.array_equal(body._goal,goal_rgb),'bound same goal/root pixels')
    row.update(initial_exact=True,root_state=root.tolist(),goal=goal.tolist(),initial_metrics=metric(goal,root))
    require(not row['initial_metrics']['native_success'],'no already-solved root')
    store.emit('goal_attached',**ops.where,goal=goal.tolist(),goal_rgb_sha256=moving.image_sha(goal_rgb),snapshot=before)
    for step,c in enumerate(cmd,1):
        if step==6:require(row['prefix_exact'] is True,'prefix5 prerequisite BEFORE new suffix dispatch')
        def dispatch(c=c):row['own_attempted']+=1;return env.step(c.copy())
        obs,reward,terminated,truncated,info=ops.call('public_step',dispatch,phase='plan',step=step,raw_action=c.tolist())
        row['own_returned']+=1;state=source.vector(obs['state']);m=metric(goal,state);success,distance=body.eval_state(goal,state)
        row.update(final_state=state.tolist(),final_metrics=m,native_success=bool(success),terminated=bool(terminated),truncated=bool(truncated))
        store.emit('returned_transition',**ops.where,phase='plan',step=step,command=c.tolist(),state=state.tolist(),reward=float(reward),terminated=bool(terminated),truncated=bool(truncated),
            returned_goal_state=np.asarray(info['goal_state']).tolist(),returned_goal_rgb_sha256=moving.image_sha(info['goal']))
        paired_capture(env,obs,'plan',step,row,ops,store,index,paired)
        require(moving.exact(np.asarray(body.latest_action),c),'actual own command exact')
        require(np.array_equal(info['goal_state'],goal) and np.array_equal(info['goal'],goal_rgb) and
            np.array_equal(body.goal_state,goal) and np.array_equal(body._goal,goal_rgb),'same goal at prefix and suffix')
        require(bool(success)==m['native_success'] and float(distance)==m['native_distance'] and bool(terminated)==bool(success) and float(reward)==-float(distance),'native metric/reward/terminal semantics')
        if step<=5:require(not terminated and not truncated,'original prefix unexpectedly stopped; NO new suffix')
        if step==5:
            row['prefix_exact']=True;row['prefix_goal_exact']=True
            store.emit('suffix_admitted_after_exact_prefix',**ops.where,prefix_snapshot=moving.snapshot(body),goal=goal.tolist(),goal_rgb_sha256=moving.image_sha(goal_rgb))
        if terminated or truncated:row['status']='native_success' if success and terminated and not truncated else 'native_stopped';break
    else:row['status']='horizon_completed'
    row['full_horizon']=row['own_returned']==10;row['endpoint_scope']='native_terminal' if row['terminated'] else 'full10' if row['full_horizon'] else 'partial'
    row['paired_native_distance_delta']=row['final_metrics']['native_distance']-paired['old_final_metrics']['native_distance']


def execute(p,data,store,factory,*,authorization):
    require(authorization==AUTH,'MAIN-only explicit physics authorization')
    require(len(p['plans'])==len(data['paired'])==4 and len(data['references'])==2,'fixed4 branches, no root selection')
    rows=[dict(root_slot=x['root_slot'],arm=x['arm'],selection_digest=x['selection_digest'],original_selection_digest=o['original_selection_digest'],
        status='not_started',first_error=None,secondary_errors=[],warmup_returned=0,own_attempted=0,own_returned=0,
        initial_exact=False,prefix_exact=False,prefix_goal_exact=False,full_horizon=False,native_success=False,terminated=False,truncated=False,PNG=0,checkpoints=[],
        saved_old_terminal=deepcopy(o['old_terminal']),saved_old_final_metrics=deepcopy(o['old_final_metrics'])) for x,o in zip(p['plans'],data['paired'],strict=True)]
    result=dict(schema=NAME+'.report',status='running',rows=rows,errors=[],contract=p['contract'],sourcepack_sha256=p['sourcepack_sha256'],
        original_physical_report_sha256=ORIGINAL_SHA,P1_admission=False,P4_admission=False,new_neural_calls=0,training=False,new_independent_samples=0,
        global_freshness='known_root_saved_prefix_feedback_calibration_not_online_from_old_terminal',separate_old_control_reexecuted=False)
    ops=Ops(p,store)
    def error(stage,exc):result['errors'].append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
    try:
        store.emit('run_started',prepare=p)
        for i,(plan,row,paired) in enumerate(zip(p['plans'],rows,data['paired'],strict=True)):
            env=None
            try:
                store.check(262144);env=factory();moving.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size==224 and env.unwrapped.render_mode=='rgb_array','native target-free/no marker RGB224')
                row['status']='running'
                with native_hooks(env,ops,row):branch(env,data['references'][plan['root_slot']],data['plan'],plan,paired,row,ops,store,i)
            except BaseException as exc:retain(row,exc,'setup_or_branch')
            finally:
                if env is not None:
                    try:env.close()
                    except BaseException as exc:retain(row,exc,'close')
                store.emit('branch_closed',receipt=row)
    except BaseException as exc:error('execution',exc)
    try:
        for path,h in p['bindings'].items():require(sha(path)==h,'final closure SHA: '+path)
    except BaseException as exc:error('final_bindings',exc)
    clean=not result['errors'] and all(r['status'] in ('horizon_completed','native_success','native_stopped') and r['prefix_exact'] and r['prefix_goal_exact'] for r in rows)
    if clean:
        try:
            c=ops.counts
            require(all(v['attempted']==v['completed'] for v in c.values()) and c['reset']['completed']==8 and c['set_state']['completed']==16 and
                c['goal_render']['completed']==8 and c['goal_bind']['completed']==4 and
                c['public_step']['completed']==sum(r['warmup_returned']+r['own_returned'] for r in rows) and
                c['physics_step']['completed']==16+10*c['public_step']['completed'] and c['explicit_render']['completed']==sum(r['PNG'] for r in rows),'actual nested resource census')
        except BaseException as exc:error('final_census',exc);clean=False
    result.update(status='moving_history_observed_suffix_physical_recorded' if clean else 'partial_or_failed',counts=ops.counts,
        native_successes=sum(r['status']=='native_success' for r in rows),wall_seconds=time.monotonic()-ops.started)
    try:
        store.emit('run_finished',status=result['status'],counts=ops.counts,errors=result['errors'])
        result['journal_sha256']=sha(store.path/'journal.jsonl');result['journal_chain_tail']=store.previous
    except BaseException as exc:error('final_journal',exc);result['status']='partial_or_failed'
    try:store.put('report.json',result,final=True)
    except BaseException as exc:error('final_report',exc);result['status']='partial_or_failed'
    return result


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authorization');p=verify(prepared,expected);m=read(p['manifest'])
    data=load_original(m['original_physical_report'],m['original_physical_report_sha256'])
    require(source.digest(data['paired'])==p['paired_receipts_sha256'],'same paired reference receipts')
    store=Store(output);store.put('execution_binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
    return execute(p,data,store,factory,authorization=authorization)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for k in ('manifest','manifest-sha','prepared','prepare-sha','output'):parser.add_argument('--'+k)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.manifest,a.manifest_sha,a.output)
    elif a.command=='verify':r=dict(schema=verify(a.prepared,a.prepare_sha)['schema'])
    else:
        require(a.authorize_physics,'MAIN CLI physics authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
        if r['status']!='moving_history_observed_suffix_physical_recorded':print(source.canonical(r));raise SystemExit(2)
    print(source.canonical({k:v for k,v in r.items() if k!='rows'}))


if __name__=='__main__':main()
