"""Eight contextual full10 plans -> original moving native bridge; MAIN only.

Saved prepare/verify/consume do not instantiate an environment or load a model.
The conditional plan's prefix/suffix remain its OWN imagined-context selection.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np

from experiments.spectral_reachability import agent_p2_contextual_composition_saved_consume_20261028_v2 as saved
from experiments.spectral_reachability import p1_moving_overlay_recompile_physical_20261027_v1 as kernel
from experiments.spectral_reachability import p1_moving_extended_baseline_physical_20261027_v1 as helper

NAME='agent_p2_contextual_composition_physical_handoff_20261028_v1'
AUTH='MAIN_CONTEXTUAL_COMPOSITION_PHYSICS_ONLY'
ROOT=saved.ROOT
require,sha=saved.require,saved.sha
raw=saved.producer.canonical
ARMS=saved.ARMS
CAP,FLOOR=kernel.CAP,kernel.FLOOR
PREP_CAP=8*1024**2
Store=kernel.Store
RECEIPT=ROOT/'results/agent_p2_contextual_composition_saved_consume_20261028_v2_MAIN01.json'
RECEIPT_SHA='578c5bd93394ae9c3863cb420afe4cfd0c59633fbbea503b4012c029211e063f'


def read(path,cap=PREP_CAP):
    require(Path(path).is_file() and not Path(path).is_symlink(),'regular saved input')
    return saved.overlay.read(path,cap)


def bits(commands):
    return kernel.previous.commands(commands).astype('<f4',copy=False).tobytes().hex()


def merge(bindings,more):
    for p,h in more.items():
        p=str(Path(p).resolve()); kernel.hash_value(h)
        require(p not in bindings or bindings[p]==h,'nonconflicting closure')
        bindings[p]=h


def source_files():
    todo=[Path(__file__),Path(kernel.__file__),Path(saved.__file__),Path(helper.__file__)]
    seen=set()
    while todo:
        p=todo.pop().resolve()
        if p in seen:continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text())):
            names=[]
            if isinstance(n,ast.Import):names=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom) and n.module:names=[n.module]+[n.module+'.'+a.name for a in n.names]
            for name in names:
                if name.startswith('experiments.'):
                    f=ROOT.joinpath(*name.split('.')).with_suffix('.py')
                    if f.is_file() and f not in seen:todo.append(f)
    return sorted(seen|set(kernel.own_files())|set(saved.own_files())|
        {ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')})


def contract():
    c=kernel.contract()
    c.update(arms=list(ARMS),schema=NAME+'.contract',known_roots=2,scientific_slots=4,
        prefix='original root10; do NOT restore own imagined midcontext as physical state',
        conditional='execute selected full10 prefix+own-imagined-context-compiled suffix without replanning',
        compiler_candidates=480,compiler_predict=1328,total_FLOPs=None,
        old_equal144_or_teacher15_reuse=False,selection_uses_physics=False,
        new_independent_roots=0,P2_admission=False,physical_release=False,equal_FLOPs=False,
        success_contract='native whole agent+object XY<20 AND original native angle term<pi/9; not object-only coverage',
        warmup_goal='original reset goal until bind; never score warmup rewards against task goal')
    return c


def validate_plans(plans,data):
    require(type(plans) is list and len(plans)==8,'full8 physical denominator')
    for p,key in zip(plans,[(s,c,a) for s,c in saved.producer.ORDER for a in ARMS],strict=True):
        require(type(p['root_slot']) is int and (p['root_slot'],p['condition'],p['arm'])==key,'fixed slot/condition/arm')
        require(bits(p['commands'])==p['raw_float32_bits'] and
            hashlib.sha256(bytes.fromhex(p['raw_float32_bits'])).hexdigest()==p['raw_sha256'],'full10 original raw bits')
        root=data['roots'][key[0]]; export=data['exports'][key[:2]]
        require(p['root_sha256']==root['root_sha256'] and
            p['blind_path']==export['path'] and p['blind_sha256']==export['sha256'],'original source root and same-condition blind')
        require(p['selection_digest']==saved.digest({k:v for k,v in p.items() if k!='selection_digest'}),'own physical plan seal')


def material(planner_output,report_sha,receipt,receipt_sha):
    """No future SHA constants: entire real strict receipt is reconstructed."""
    kernel.hash_value(receipt_sha); kernel.hash_value(report_sha)
    output=Path(planner_output).resolve(); receipt=Path(receipt).resolve()
    authority=saved.verify_receipt(receipt,receipt_sha)
    require(authority['schema']==saved.STEM+'.receipt' and authority['status']=='strict_saved_verified_science_blocked'
        and authority['planner_output']==str(output) and authority['planner_report_sha256']==report_sha,
        'strict receipt belongs to this exact actual run')
    native=authority['native_strict_receipt']
    require(native['schema']==saved.producer.NAME+'.consume' and native['status']=='engineering_verified_science_blocked'
        and native['candidate_denominator']==480 and native['known_roots']==2 and native['scientific_slots']==4
        and native['total_predict']==1328 and native['total_FLOPs'] is None
        and native['P1_admission'] is native['P2_admission'] is native['physical_release'] is False,
        'full native strict authority, no scientific upgrade')
    report=read(output/'report.json'); prep=saved.producer.verify(report['prepared'],report['prepare_sha256'])
    data=kernel.source_material(); bindings={};merge(bindings,data['bindings']);merge(bindings,prep['bindings_sha256'])
    merge(bindings,authority['bindings_sha256'])
    merge(bindings,{str(output/'report.json'):report_sha,str(receipt):receipt_sha,report['prepared']:report['prepare_sha256']})
    merge(bindings,{str(output/n):h for n,h in report['artifact_sha256'].items()})
    plans=[]
    for original in native['plans']:
        s,c=original['root_slot'],original['condition']; slot=next(x for x in prep['slots'] if (x['root_slot'],x['condition'])==(s,c))
        ready,compiled=saved.producer.task_bindings(slot)
        export=data['exports'][s,c]; packet=data['roots'][s]['packets'][c]
        task=read(slot['task_file'])
        require(original['root_context']==ready['root']['context'] and original['goal']==compiled['goal_vectors'][0]
            and original['task_id']==slot['task_id'] and original['original_blind_sha256']==slot['blind_sha256']==export['sha256']
            and original['original_task_file']==slot['task_file'] and original['original_task_sha256']==slot['task_sha256'],
            'original prediction/goal/context/task and blind identity')
        require(bits(task['task']['root_raw_action_history'])==bits(packet['raw_action_history'].tolist())==bits(kernel.moving.warmup_commands().tolist()),
            'original physical warm10 raw history, no teacher15')
        pixels=task['task_contract']['pixel_sha256']
        require(pixels==[kernel.moving.image_sha(x) for x in list(packet['rgb_history'])+[packet['goal_rgb']]],
            'all3 root RGB and same-condition goalRGB literal pixels')
        require(original['selection_digest']==saved.digest({k:v for k,v in original.items() if k!='selection_digest'}),
            'original model-only selected plan digest')
        p=deepcopy(original)
        p.update(compiler_selection_digest=original['selection_digest'],raw_float32_bits=bits(original['commands']),
            root_sha256=data['roots'][s]['root_sha256'],blind_path=export['path'],blind_sha256=export['sha256'],
            source_report_sha256=kernel.SOURCE_SHA,strict_receipt_sha256=receipt_sha,
            encoder_identity={k:deepcopy(task['task'][k]) for k in ('model_sha256','model_config_sha256','action_mean','action_std')})
        del p['selection_digest'];p['selection_digest']=saved.digest(p);plans.append(p)
    validate_plans(plans,data)
    native_path,h=helper.native_constructor_contract();merge(bindings,{native_path:h})
    files=source_files()
    merge(bindings,{str(f.resolve()):sha(f) for f in files})
    for f,h in bindings.items():require(sha(f)==h,'complete physical handoff closure')
    p=dict(schema=NAME+'.prepare',contract=contract(),plans=plans,bindings=bindings,
        planner_output=str(output),planner_report_sha256=report_sha,strict_receipt=str(receipt),strict_receipt_sha256=receipt_sha,
        source_report_sha256=kernel.SOURCE_SHA,compiler_resources=saved.producer.protocol())
    return p,data


def prepare(output,planner_output,report_sha,receipt,receipt_sha):
    p,_=material(planner_output,report_sha,receipt,receipt_sha)
    path=Path(output).absolute();value=raw(p)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and
        len(value)<=PREP_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(value),'exclusive bounded prepare')
    with path.open('xb') as f:require(f.write(value)==len(value),'short prepare');f.flush();os.fsync(f.fileno())
    kernel.paired.fsync_dir(path.parent)
    return dict(prepared=str(path),prepare_sha256=sha(path),status='prepared_no_physics_no_admission')


def verify(path,expected):
    require(sha(path)==expected,'external own prepare SHA')
    p=read(path);now,data=material(p['planner_output'],p['planner_report_sha256'],p['strict_receipt'],p['strict_receipt_sha256'])
    require(raw(p)==raw(now) and sha(path)==expected,'entire own prepare reconstruction')
    return p,data


def execute(p,data,store,factory,*,authorization,binding,started=None):
    """Own dispatch loop; reuse native episode, not old-arm loader/execute."""
    require(authorization==AUTH and p['schema']==NAME+'.prepare' and p['contract']==contract(),'MAIN own physical dispatch authorization')
    validate_plans(p['plans'],data)
    require(binding['planner_report_sha256']==p['planner_report_sha256'] and binding['strict_receipt_sha256']==p['strict_receipt_sha256'],
        'execution authority binding')
    rows=[dict(index=i,root_slot=x['root_slot'],condition=x['condition'],arm=x['arm'],selection_digest=x['selection_digest'],
        task_id=x['task_id'],status='not_started',checkpoints=[],native_success=None,missing_checkpoints=[0,5,10],
        warmup_returned=0,own_attempted=0,own_returned=0,PNG=0,initial_exact=False,trajectory=[],last_observation=None,
        first_error=None,secondary_errors=[]) for i,x in enumerate(p['plans'])]
    meter=kernel.Meter(store,started); errors=[];seen=[];source_ok=False
    def fail(stage,exc,row=None):
        e=kernel.error(stage,exc);errors.append(e)
        if row is not None:
            if row['first_error'] is None:row['first_error']=deepcopy(e)
            else:row['secondary_errors'].append(deepcopy(e))
            row['status']='engineering_failed'
    try:
        store.put('execution_binding.json',binding)
        store.emit('run_started',prepare_sha256=binding['prepare_sha256'],plans=p['plans'],contract=contract())
        for row,plan in zip(rows,p['plans'],strict=True):
            meter.where={k:row[k] for k in ('root_slot','condition','arm','selection_digest')};meter.where['branch']=row['index']
            env=None;row['status']='running'
            try:
                env=meter.call('constructor',factory)
                require(all(env is not e and env.unwrapped is not e.unwrapped for e in seen),'fresh independent native instance')
                seen.append(env);body=env.unwrapped;kernel.validate_env(body)
                require(all(getattr(body,k,None) is None for k in ('space','render_buffer','latest_action')),'inert pristine constructor before hooks')
                require(body.render_action is False and body.render_size==224 and body.render_mode=='rgb_array','native renderer/no action marker')
                with kernel.native_hooks(env,meter,row['secondary_errors']):
                    kernel.episode(env,data['roots'][plan['root_slot']],data['source_plan'],plan,row,meter,store)
                require(not row['secondary_errors'],'native hook restoration')
            except kernel.NativeStop as e:row.update(status='native_stopped',stop_reason=str(e))
            except BaseException as e:fail('branch',e,row)
            finally:
                if env is not None:
                    try:meter.call('close',env.close,cleanup=True)
                    except BaseException as e:fail('close',e,row)
                if row['secondary_errors'] and row['first_error'] is None:fail('cleanup',ValueError('native restore errors'),row)
            store.emit('branch_closed',**meter.where,status=row['status'],own_returned=row['own_returned'],first_error=row['first_error'])
    except BaseException as e:fail('execution',e)
    try:
        for f,h in p['bindings'].items():require(sha(f)==h,'final source/model/task seal')
        source_ok=True
    except BaseException as e:fail('final_inputs',e)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started>=900:fail('final_wall',ValueError('900s cap exhausted'))
    if not errors:
        try:kernel.census(rows,meter.counts)
        except BaseException as e:fail('census',e)
    clean=not errors and source_ok and all(r['status'] in ('horizon_completed','native_success','native_stopped') for r in rows)
    report=dict(schema=NAME+'.report',status='contextual_physical_recorded_not_admitted' if clean else 'partial_or_failed',
        rows=rows,errors=errors,contract=contract(),execution_binding=binding,fixed_denominator=8,
        counts=deepcopy(meter.counts),inputs_final_verified=source_ok,P1_admission=False,P2_admission=False,physical_release=False,
        new_neural_calls=0,new_independent_roots=0,wall_seconds=time.monotonic()-meter.started)
    try:
        store.emit('run_finished',status=report['status'],counts=meter.counts,errors=errors)
        report.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous,
            execution_binding_sha256=sha(store.path/'execution_binding.json'))
    except BaseException as e:fail('final_journal',e);report['status']='partial_or_failed'
    report['wall_seconds']=time.monotonic()-meter.started
    if report['wall_seconds']>=900 and not any(e['stage']=='final_wall' for e in errors):
        fail('final_wall',ValueError('900s cap exhausted during finalization'));report['status']='partial_or_failed'
    try:store.put('report.json',report,final=True)
    except BaseException as e:fail('final_report',e);report['status']='partial_or_failed'
    return report


def run(prepared,expected,output,factory,*,authorization):
    require(authorization==AUTH,'MAIN authorization BEFORE I/O/factory')
    started=time.monotonic();p,data=verify(prepared,expected)
    require(Path(output).name.startswith(NAME+'_'),'own output prefix')
    binding=dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected,
        planner_report_sha256=p['planner_report_sha256'],strict_receipt_sha256=p['strict_receipt_sha256'])
    return execute(p,data,Store(output),factory,authorization=authorization,binding=binding,started=started)


def metrics(state,goal):
    s=np.asarray(state,np.float64);g=np.asarray(goal,np.float64)
    require(s.shape==g.shape==(7,) and np.isfinite(s).all() and np.isfinite(g).all(),'native7 finite state/goal')
    d=s-g;joint=float(np.linalg.norm(d[:4]));a=abs(float(d[4]));angle=min(a,2*np.pi-a)
    return dict(native_success=bool(joint<20 and angle<np.pi/9),native_reward=-float(np.linalg.norm(d)),
        joint_xy_error=joint,agent_xy_error=float(np.linalg.norm(d[:2])),object_xy_error=float(np.linalg.norm(d[2:4])),
        agent_velocity_error=float(np.linalg.norm(d[5:])),native_angle_term=angle,
        wrapped_angle_error=abs(float(np.arctan2(np.sin(d[4]),np.cos(d[4])))),native_coverage=None,object_only_coverage=None)


def assemble(r,events,p,data,image):
    """Strict saved native/raw/PNG/resource joins, no environment construction."""
    require(r['schema']==NAME+'.report' and r['contract']==p['contract']==contract() and r['fixed_denominator']==8
        and len(r['rows'])==8 and r['new_neural_calls']==r['new_independent_roots']==0
        and r['P1_admission'] is r['P2_admission'] is r['physical_release'] is False,'own eight-denominator nonadmission')
    require(events and events[0]['kind']=='run_started' and events[0]['plans']==p['plans'] and events[0]['contract']==contract()
        and events[0]['prepare_sha256']==r['execution_binding']['prepare_sha256']
        and events[-1]['kind']=='run_finished' and events[-1]['status']==r['status']
        and events[-1]['counts']==r['counts'] and events[-1]['errors']==r['errors'],'exact durable start/terminal')
    # Pair every nested reserve/entry/return, not just aggregate event hashes.
    counts={k:dict(attempted=0,completed=0) for k in contract()['caps']};stack=[];failed=[]
    for e in events:
        if e['kind'] not in ('operation_attempted','operation_completed','operation_failed'):continue
        k=e['resource'];require(k in counts,'known native operation')
        sig={x:v for x,v in e.items() if x not in ('index','previous_sha256','event_sha256','kind','type','message')}
        if e['kind']=='operation_attempted':
            counts[k]['attempted']+=1;require(counts[k]['attempted']<=contract()['caps'][k],'native predeclared cap');stack.append(sig)
        else:
            require(stack and stack.pop()==sig,'nested actual operation attempt/return identity')
            if e['kind']=='operation_completed':counts[k]['completed']+=1
            else:failed.append(e)
    require(counts==r['counts'] and not stack,'complete actual resource ledger, pending rejected')
    clean=r['status']=='contextual_physical_recorded_not_admitted'
    require(clean or r['status']=='partial_or_failed','terminal status only')
    if clean:
        require(r['errors']==[] and not failed and r['inputs_final_verified'] is True,'clean actual execution')
        kernel.census(r['rows'],counts)
        require(0<=r['wall_seconds']<900,'actual wall cap')
    scoped=[e for e in events if 'branch' in e]
    require(all(e['kind'] in {'run_started','run_finished','operation_attempted','operation_completed','operation_failed',
        'placement_recipe','fresh_observation','returned_transition','goal_attached','branch_closed','invalid_observation'}
        for e in events),'known original native bridge event ABI')
    require(all(type(e['branch']) is int and e['branch'] in range(8) for e in scoped) and
        [e['branch'] for e in scoped]==sorted(e['branch'] for e in scoped),'fixed branch order, no unknown branch')
    require(all('branch' in e for e in events[1:-1]),'no unscoped hidden work')
    rows=[];png_names=[]
    for i,(row,plan) in enumerate(zip(r['rows'],p['plans'],strict=True)):
        identity=dict(branch=i,root_slot=plan['root_slot'],condition=plan['condition'],arm=plan['arm'],selection_digest=plan['selection_digest'])
        require(row['index']==i and all(row[k]==v for k,v in identity.items() if k!='branch') and row['task_id']==plan['task_id'],'row/selected plan identity')
        own=[e for e in events if e.get('branch')==i]
        require(all(all(e[k]==v for k,v in identity.items()) for e in own),'all branch-scoped event identity')
        attached=[e for e in own if e['kind']=='goal_attached'];returns=[e for e in own if e['kind']=='returned_transition']
        observations=[e for e in own if e['kind']=='fresh_observation']
        if clean:
            closed=[e for e in own if e['kind']=='branch_closed']
            require(len(closed)==1 and own[-1]==closed[0] and closed[0]['status']==row['status']
                and closed[0]['own_returned']==row['own_returned'] and closed[0]['first_error']==row['first_error'],
                'durable branch terminal')
        require(len(observations)==row['PNG'],'all emitted PNG denominator')
        chosen=[];triplets={};reference=data['roots'][plan['root_slot']];packet=reference['packets'][plan['condition']]
        for e in observations:
            f=e['receipt']; a=image(f);png_names.append(f['png'])
            require(a.shape==(224,224,3) and a.dtype==np.uint8 and kernel.moving.image_sha(a)==f['pixel_sha256'],'RGB224 exact pixel binding')
            require(e['pre']==e['post'] and f['snapshot']==e['post']['snapshot'] and f['state']==f['snapshot']['state'],'render invariants and native snapshot')
            kernel.moving.guard(a,np.asarray(f['state']))
            key=f['phase'],f['step'];triplets.setdefault(key,[]).append((f,e))
            if f['overlay_index']==(0 if plan['condition']=='off' else 1):chosen.append(f)
            if f['phase']=='warmup' and f['overlay_index'] in (0,2):
                w=reference['warm'][f['step']]
                require(all(f[k]==w[k] for k in ('file_sha256','pixel_sha256','state','snapshot')),'exact original warmup source')
        if clean:
            require(row['status'] in ('horizon_completed','native_success','native_stopped') and row['first_error'] is None and row['secondary_errors']==[],
                'all legal native stops, not engineering failures')
            for key,fs in triplets.items():
                require([f['overlay_index'] for f,e in fs]==[0,1,2] and [f['condition'] for f,e in fs]==['off','on','off']
                    and fs[0][0]['file_sha256']==fs[2][0]['file_sha256']
                    and fs[0][1]['post']==fs[1][1]['post']==fs[2][1]['post'],'full off/on/off identity')
            require(chosen==row['trajectory'],'same-condition entire true RGB trajectory')
            expected_observed=[('warmup',n) for n in range(row['warmup_returned']+1)]
            if row['initial_exact']:expected_observed += [('plan',n) for n in range(row['own_returned']+1)]
            require(list(triplets)==expected_observed and len(observations)==3*len(expected_observed),'every returned state has complete RGB triplet')
            expected_attempts=[e for e in own if e['kind']=='operation_attempted' and e['resource']=='public_step' and e.get('phase')=='plan']
            require(row['own_attempted']==len(expected_attempts)==row['own_returned'],'actual action attempted/returned census')
        phases={phase:[e for e in returns if e['phase']==phase] for phase in ('warmup','plan')}
        actual_metrics=[];executed=[]
        for phase,rr in phases.items():
            expected=kernel.moving.warmup_commands() if phase=='warmup' else kernel.previous.commands(plan['commands'])
            require(len(rr)==row['warmup_returned' if phase=='warmup' else 'own_returned'] and
                [e['step'] for e in rr]==list(range(1,len(rr)+1)),'exact returned prefix with no holes/padding')
            stepops=[e for e in own if e['kind']=='operation_completed' and e['resource']=='public_step' and e.get('phase')==phase]
            require(len(stepops)==len(rr),'every completed public step has actual return telemetry')
            for e,op in zip(rr,stepops,strict=True):
                b=expected[e['step']-1].astype('<f4').tobytes().hex()
                require(e['requested_bits']==e['actual_bits']==op['raw_bits']==b and e['actual_dtype']=='<f4'
                    and e['state']==e['snapshot']['state'],'actual native rawfloat32 commands and state')
                if clean:
                    obs=triplets[phase,e['step']]
                    require(all(f['state']==e['state'] and f['snapshot']==e['snapshot'] for f,entry in obs),
                        'every returned native snapshot joined to actual rendered RGB')
                    require(events.index(op)<events.index(e)<events.index(obs[0][1]),'native return precedes fresh observation')
                m=metrics(e['state'],e['goal_state'])
                require(m['native_reward']==e['reward'] and m['native_success'] is e['terminated'],'original native reward and termination (warmup uses its OLD goal)')
                if phase=='plan':
                    require(len(attached)==1 and e['goal_state']==reference['goal'].tolist(),'task goal only after bind')
                    actual_metrics.append(dict(step=e['step'],**m));executed.append(expected[e['step']-1].tolist())
                    if e['terminated'] or e['truncated']:require(e is rr[-1],'no step after native stop')
        rgb=[]
        if row['initial_exact']:
            require(len(attached)==1 and attached[0]['goal']==reference['goal'].tolist() and attached[0]['goal_pose']==row['goal_pose']
                and attached[0]['goal_rgb_sha256']==row['goal_pixel_sha256']==kernel.moving.image_sha(packet['goal_rgb'])
                and kernel.source.root_hash(row['root_state'],reference['goal'])==reference['root_sha256'],'exact root10 and unchanged outline/taskgoal')
            require(row['warmup_returned']==10 and triplets['warmup',10][0][0]['snapshot']==attached[0]['snapshot']
                ==triplets['plan',0][0][0]['snapshot'],'goal bind has no dynamics effect at exact root10')
            require(events.index(triplets['warmup',10][-1][1])<events.index(attached[0])<events.index(triplets['plan',0][0][1]),
                'task goal attached AFTER warm10, before plan0')
            selected_warm=[f for f in chosen if f['phase']=='warmup' and f['step'] in (0,5,10)]
            require(len(selected_warm)==3 and all(np.array_equal(image(f),a) for f,a in zip(selected_warm,packet['rgb_history'],strict=True)),
                'all3 selected-condition real RGB source history')
            for e in observations:
                if e['receipt']['phase']=='plan':
                    require(e['post']['goal_state']==attached[0]['goal'] and e['post']['goal_pose']==attached[0]['goal_pose']
                        and e['post']['goal_rgb_sha256']==attached[0]['goal_rgb_sha256'],'fixed plan goal/pose/RGB')
            expected_cp=[{k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}
                for f in chosen if f['phase']=='plan' and f['step'] in (0,5,10)]
            require(row['checkpoints']==expected_cp and row['missing_checkpoints']==[x for x in (0,5,10) if x not in [f['step'] for f in expected_cp]],
                'all real checkpoints, no missing substitution')
            for cp in expected_cp:
                n=cp['step'];history=np.concatenate((kernel.moving.warmup_commands(),np.asarray(executed,np.float32).reshape(-1,2)[:n]))[-10:]
                rgb.append(dict(**cp,condition=plan['condition'],raw_action_history10=history.tolist(),raw_action_history_bits=bits(history.tolist()),
                    original_blind_path=plan['blind_path'],original_blind_sha256=plan['blind_sha256'],task_id=plan['task_id'],
                    original_task_file=plan['original_task_file'],original_task_sha256=plan['original_task_sha256'],
                    compiler_selection_digest=plan['compiler_selection_digest'],encoder_identity=deepcopy(plan['encoder_identity']),
                    model_bound_by_prepare=True,encoding_performed=False))
            if clean:
                if phases['plan']:
                    last=phases['plan'][-1]
                    require(row['native_success'] is actual_metrics[-1]['native_success'] and row['final_state']==last['state']
                        and row['terminated'] is last['terminated'] and row['truncated'] is last['truncated'],'final native row/return identity')
                    require((row['status']=='horizon_completed' and row['own_returned']==10 and not last['terminated'] and not last['truncated'])
                        or (row['status']=='native_success' and last['terminated'] and not last['truncated'])
                        or (row['status']=='native_stopped' and last['truncated']),'no invented full horizon or native success')
                else:require(row['native_success'] is metrics(row['root_state'],row['goal'])['native_success'],'native root stop')
        rows.append(dict(index=i,root_slot=plan['root_slot'],condition=plan['condition'],arm=plan['arm'],status=row['status'],
            selection_digest=plan['selection_digest'],compiler_selection_digest=plan['compiler_selection_digest'],
            own_returned=row['own_returned'],full_horizon=row['own_returned']==10,
            initial_metrics=metrics(row['root_state'],row['goal']) if row['initial_exact'] else None,
            actual_metrics=actual_metrics,final_metrics=actual_metrics[-1] if actual_metrics else None,
            missing_checkpoints=row['missing_checkpoints'],rgb_encoding_inputs=rgb))
    require(len(png_names)==len(set(png_names)),'unique original PNG artifacts')
    return dict(schema=NAME+'.consume',status='engineering_verified_science_blocked' if clean else 'partial_failclosed',
        rows=rows,counts=counts,PNG_files=png_names,fixed_denominator=8,known_roots=2,independent_new_roots=0,
        P1_admission=False,P2_admission=False,physical_release=False,new_neural_calls=0,encoding_performed=False)


def journal(path,tail):
    value=Path(path).read_bytes();require(0<len(value)<=CAP and value.endswith(b'\n'),'bounded complete physical journal')
    previous=None;events=[]
    for index,line in enumerate(value.splitlines()):
        e=saved.overlay.blind.decode(line)
        require(kernel.raw(e)==line+b'\n' and e['index']==index and e['previous_sha256']==previous
            and e['event_sha256']==kernel.source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'original canonical physical hashchain')
        previous=e['event_sha256'];events.append(e)
    require(previous==tail and events[-1]['kind']=='run_finished','actual physical terminal tail')
    return events


def consume(output,report_sha):
    output=Path(output).resolve();require(sha(output/'report.json')==report_sha,'external actual physical report SHA')
    r=read(output/'report.json');b=r['execution_binding'];p,data=verify(b['prepared'],b['prepare_sha256'])
    require(b['planner_report_sha256']==p['planner_report_sha256'] and b['strict_receipt_sha256']==p['strict_receipt_sha256']
        and sha(output/'execution_binding.json')==r['execution_binding_sha256'] and read(output/'execution_binding.json')==b,'exact execution binding')
    require(sha(output/'journal.jsonl')==r['journal_sha256'],'original physical journal bytes')
    require(sum(f.stat().st_size for f in output.iterdir())<=CAP,'physical output64MiB cap')
    events=journal(output/'journal.jsonl',r['journal_chain_tail'])
    def image(f):
        require(Path(f['png']).name==f['png'],'flat original PNG');return kernel.old_image(dict(path=str(output/f['png']),**f))
    result=assemble(r,events,p,data,image)
    require({f.name for f in output.iterdir()}==set(result['PNG_files'])|{'report.json','journal.jsonl','execution_binding.json'},'exact physical artifact census')
    require(sha(output/'report.json')==report_sha,'report unchanged after full consumption')
    for path,h in p['bindings'].items():require(sha(path)==h,'final consume source/model/authority closure')
    for row in result['rows']:
        for cp in row['rgb_encoding_inputs']:cp['path']=str(output/cp['png'])
    result.update(physical_output=str(output),physical_report_sha256=report_sha,planner_report_sha256=p['planner_report_sha256'],
        strict_receipt_sha256=p['strict_receipt_sha256'],prepared=b['prepared'],prepare_sha256=b['prepare_sha256'])
    return result


def write_receipt(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and
        len(payload)<=PREP_CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'exclusive own consume receipt/floor')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'short receipt');f.flush();os.fsync(f.fileno())
    kernel.paired.fsync_dir(path.parent)
    return dict(receipt=str(path),receipt_sha256=sha(path),status=value['status'],rows=len(value['rows']),P2_admission=False,physical_release=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','physical-report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--strict-receipt',default=str(RECEIPT));q.add_argument('--strict-receipt-sha',default=RECEIPT_SHA)
    q.add_argument('--planner-output',default=str(saved.DIRECTORY));q.add_argument('--planner-report-sha',default=saved.REPORT_SHA)
    q.add_argument('--authorize-physics',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=prepare(a.output,a.planner_output,a.planner_report_sha,a.strict_receipt,a.strict_receipt_sha)
    elif a.command=='verify':p,_=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',plans=len(p['plans']))
    elif a.command=='consume':
        result=consume(a.output,a.physical_report_sha)
        require(a.receipt,'consume requires exclusive durable receipt path');v=write_receipt(a.receipt,result)
    else:
        require(a.authorize_physics,'MAIN explicit physical flag')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        v=run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
    print(raw({k:x for k,x in v.items() if k!='rows'}).decode(),end='',flush=True)
    return 1 if v.get('status') in ('partial_or_failed','partial_failclosed') else 0


if __name__=='__main__':raise SystemExit(main())
