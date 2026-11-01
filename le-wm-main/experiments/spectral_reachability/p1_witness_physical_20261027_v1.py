"""Fixed eight-branch P1 calibration physical consumer; MAIN only, no neural work.

MAIN authorized 88 explicit renders: fresh RGB per command through the existing
guard, saving only 24 initial/step5/step10 PNGs. No equality-of-compute claim.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
import inspect
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_two_stage_witness_20261027_v1 as witness
from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard

source, core, screen, upstream = witness.source, witness.core, witness.screen, witness.upstream
NAME = 'p1_witness_physical_20261027_v1'
ARMS = ('zero', 'full_H2_GN', 'full_H2_gradient', 'two_stage')
ROOT = Path(__file__).resolve().parents[2]
CAP = 8 * 1024**2
FLOOR = 200 * 1024**2
require = source.require
sha = source.file_sha


def read(path):
    return json.loads(Path(path).read_text())


def contract():
    return dict(roots=2, arms=list(ARMS), order='source_plan_slot_order_then_fixed_arms', branches=8,
                resets=16, selected_steps=80, explicit_renders=88, internal_goal_renders=16,
                reset_internal_physics_steps=32, checkpoints=[0,5,10], neural_calls=0,
                guard='existing_RgbObservabilityGuard_fresh_initial_and_each_poststep; no_extra_precommand_render',
                evaluation='native_unchanged_and_separate_wrapped_allxy_success_and_object_pose; no_termination_override',
                initial='original_random_seed_then_original_sample_state_then_placement_seed',
                history='causal_padding_not_observed_transitions', artifact_cap_bytes=CAP,
                reserve_bytes=FLOOR, P1_admission=False, equal_compute=False, independent_confirmation=False)


def bindings_check(bindings):
    for path, expected in bindings.items():
        require(Path(path).is_file() and not Path(path).is_symlink() and sha(path)==expected, 'binding drift: '+str(path))


def raw_commands(value):
    a = np.asarray(value)
    require(a.shape==(10,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and (np.abs(a)<=1).all(), 'exact bounded 10x2 raw commands')
    require(np.array_equal(a,a.astype(np.float32)), 'saved commands must be exactly float32 representable; no conversion/clip repair')
    return a.astype(np.float32)


def direct_selection(task, report, method):
    require(report['status']=='direct_component_completed_not_admitted' and report['error'] is None and
            not report['secondary_errors'] and report['task_id']==core.task_id(task), 'complete same-task direct report')
    configuration = upstream.config(method)
    require(report['configuration']==configuration and report['configuration_sha256']==core.digest(configuration) and
            report['budget']==screen.budget(configuration), 'fixed direct optimizer/budget')
    require(len(report['direct_runs'])==1 and len(report['direct_runs'][0])==4, 'one full four-iterate direct trajectory')
    run=report['direct_runs'][0]; previous=None
    for i, step in enumerate(run):
        require(step['iteration']==i, 'ordered original direct iterations')
        core.check_prediction(task,step['incumbent'])
        if i==0:
            require(step['proposal'] is None and step['accepted'] is False and step['derivative'] is None and
                    step['incumbent']['logits']==configuration['initial_logits'][0], 'original zero initial incumbent')
        else:
            core.check_prediction(task,step['proposal']); core.check_derivative(task,step['derivative'],previous)
            accepted=step['proposal']['cost']<previous['cost']
            require(step['accepted'] is accepted and step['incumbent']==(step['proposal'] if accepted else previous), 'original strict incumbent update, not physical reselection')
        previous=step['incumbent']
    costs={k:dict(attempted=sum(e['samples'] for e in report['extended_journal'] if e['resource']==k),
                       completed=sum(e['samples'] for e in report['extended_journal'] if e['resource']==k and e['completed']))
           for k in report['budget']}
    require(all(costs[k]==dict(attempted=n,completed=n) for k,n in report['budget'].items()), 'complete actual direct costs')
    return previous, costs


def witness_selection(task, report):
    require(report['status']=='witness_completed_not_admitted' and report['error'] is None and not report['secondary_errors'] and
            report['task_id']==core.task_id(task) and report['protocol_sha256']==core.digest(witness.PROTOCOL), 'complete same-task witness')
    require(report['budget']==witness.budgets() and report['resources']==witness.counts(report['extended_journal']), 'actual witness costs')
    require(all(report['resources'][phase][k]==dict(attempted=n,completed=n)
                for phase, budget in witness.budgets().items() for k,n in budget.items()), 'complete scoped witness budgets')
    require(len(report['candidates'])==64 and len(report['prefix_compilations'])==16, 'all original 64 candidates/16 contexts')
    for i,c in enumerate(report['candidates']):
        require(c['index']==i and c['prefix_index']==i//4 and c['suffix_index']==i%4, 'full ordered candidate census')
        pred=dict(task_id=core.task_id(task),logits=c['logits'],commands=c['commands'],contexts=c['contexts'],
                  whole_replay=c['contexts'][-1],segmented_replay=c['contexts'],cost=c['cost'])
        # Candidate chronology/math only; independent winner replay checked below.
        core.check_prediction(task,pred)
        compilation=report['prefix_compilations'][i//4]
        require(compilation['index']==i//4 and compilation['input_context_sha256']==core.digest(c['contexts'][0]), 'own-prefix context binding')
    best=min(report['candidates'],key=lambda c:(c['cost'],c['index']))
    require(report['winner']==best, 'original stable argmin over all64, no physical selection')
    verified=report['witness']; core.check_prediction(task,verified)
    require(all(verified[k]==best[k] for k in ('commands','logits','contexts','cost')), 'independent H2 winner exact replay')
    return verified, report['resources']


def collect(witness_prepared, witness_prepare_sha, witness_report, witness_report_sha,
            direct_prepared, direct_prepare_sha, direct_report, direct_report_sha, sourcepack, sourcepack_sha):
    """Pure saved evidence validation; no environment/model construction."""
    wp, wr, dp, dr, sp = map(lambda x:Path(x).resolve(),(witness_prepared,witness_report,direct_prepared,direct_report,sourcepack))
    for value in (witness_prepare_sha,witness_report_sha,direct_prepare_sha,direct_report_sha,sourcepack_sha): source.hash_value(value)
    bindings={str(wr):witness_report_sha,str(dr):direct_report_sha,str(sp/'sourcepack.json'):sourcepack_sha,
              str(wp/'prepare.json'):witness_prepare_sha,str(dp/'prepare.json'):direct_prepare_sha}
    bindings_check(bindings)
    p=witness.verify(wp,witness_prepare_sha); direct=upstream.verify(dp,direct_prepare_sha)
    inputs=p['inputs']; w=read(wr); d=read(dr); s=read(sp/'sourcepack.json')
    require(Path(inputs['direct_prepared']).resolve()==dp and inputs['direct_prepare_sha256']==direct_prepare_sha and
            Path(inputs['task_directory']).resolve()==dr.parent, 'same original direct acquisition/prepare')
    require(Path(direct['sourcepack']).resolve()==sp and direct['sourcepack_sha256']==sourcepack_sha and
            direct['source_plan']==s['plan'], 'same original calibration2 sourcepack')
    require(w['status']=='cuda_completed_not_admitted' and not w['errors'] and w['source_verified'] is True and
            w['model_unchanged'] is True and w['model_hash_before']==w['model_hash_after'] and w['model_hash_before'] is not None and
            w['prepare_sha256']==witness_prepare_sha and w['algorithm_sha256']==core.digest(witness.PROTOCOL), 'real complete witness CUDA required')
    require(d['status']=='direct_calibration_completed' and not d['errors'] and d['exact_budget_verified'] is True and
            d['source_verified'] is True and d['model_unchanged'] is True and d['prepare_sha256']==direct_prepare_sha and
            d['model_hash_before']==d['model_hash_after']==w['model_hash_before'], 'real complete direct same weights required')
    require(len(w['roots'])==2 and len(d['direct'])==4 and len(inputs['tasks'])==2, 'complete original cohorts')
    tasks={x['root_sha256']:x for x in inputs['tasks']}
    roots={x['root_sha256']:x for x in w['roots']}
    methods={(x['root_sha256'],x['method']):x for x in d['direct']}
    require(len(tasks)==len(roots)==2 and len(methods)==4, 'unique stable root/method joins')
    source_roots={x['metadata']['root_sha256'] for x in s['slots']}
    require(set(tasks)==set(roots)==source_roots and set(methods)=={(h,m) for h in source_roots for m in upstream.METHODS}, 'exact source/root/method census')
    bindings.update(direct['bindings']); bindings.update(direct['source_plan']['source_closure'])
    bindings.update(inputs['bindings']); bindings.update(p['sources_sha256']); bindings.update(direct['sources_sha256'])
    plans=[]
    for source_row in s['slots']:
        root=source_row['metadata']['root_sha256']; item=tasks[root]; task=item['task']; core.validate_task(task)
        wrt=roots[root]
        require(wrt['task_id']==item['task_id']==core.task_id(task), 'stable root/task identity')
        sub=Path(wrt['report_path']).resolve(); journal=Path(wrt['journal_path']).resolve()
        require(sub.parent==wr.parent and journal.parent==wr.parent, 'witness artifacts must belong to actual run')
        bindings[str(sub)]=wrt['report_sha256']; bindings[str(journal)]=wrt['journal_sha256']; bindings_check({str(sub):bindings[str(sub)],str(journal):bindings[str(journal)]})
        subreport=read(sub)
        winner,cost=witness_selection(task,subreport)
        require(wrt['status']==subreport['status'] and wrt['error'] is None and wrt['resources']==cost,
                'top-level witness root completion/cost binding')
        selections={'two_stage':(winner,cost,str(sub))}
        for method in upstream.METHODS:
            ref=methods[root,method]; sub=Path(ref['report_path']).resolve()
            require(sub.parent.parent==dr.parent and ref['status']=='direct_component_completed_not_admitted', 'direct artifact same actual run')
            bindings[str(sub)]=ref['report_sha256']; bindings_check({str(sub):ref['report_sha256']})
            prediction,work=direct_selection(task,read(sub),method); selections[method]=(prediction,work,str(sub))
        selections['zero']=(None,dict(incremental_search=0,source='predeclared_zero_raw_not_model_selected'),None)
        for arm in ARMS:
            prediction,work,origin=selections[arm]
            commands=np.zeros((10,2),np.float32) if prediction is None else raw_commands(prediction['commands'])
            plan=dict(root_sha256=root,task_id=item['task_id'],candidate_id=source_row['slot']['candidate_id'],arm=arm,
                      source_metadata=source_row['metadata'],raw_source_file=str(sp/source_row['raw_source_file']),
                      raw_source_sha256=source_row['raw_source_sha256'],commands=commands.tolist(),
                      prediction_sha256=None if prediction is None else source.digest(prediction),origin=origin,search_cost=work,
                      goal_Q_context_sha256=source.digest({k:task[k] for k in ('goal','projection','root_context','action_mean','action_std','model_sha256')}))
            plan['selection_sha256']=source.digest(plan); plans.append(plan)
    bindings_check(bindings)
    return dict(arguments=dict(witness_prepared=str(wp),witness_prepare_sha=witness_prepare_sha,witness_report=str(wr),witness_report_sha=witness_report_sha,
                direct_prepared=str(dp),direct_prepare_sha=direct_prepare_sha,direct_report=str(dr),direct_report_sha=direct_report_sha,sourcepack=str(sp),sourcepack_sha=sourcepack_sha),
                plans=plans,source_plan=s['plan'],bindings=bindings,
                inherited_cost=inputs['inherited'],direct_total_cost=d['actual_direct_resources'],witness_total_cost=[x['resources'] for x in w['roots']])


class Store:
    """One exclusive directory, bounded JSONL/PNG/report bytes, fsync each write."""
    def __init__(self, path):
        self.path=Path(path); self.used=0; self.previous=None; self.events=0
        require(shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'200MiB plus full8MiB reserve')
        self.path.mkdir(exist_ok=False)

    def check(self, size=65536):
        require(self.used+size<=CAP,'8MiB global cap')
        require(shutil.disk_usage(self.path).free>=FLOOR+size,'200MiB floor')

    def put(self, name, raw, *, append=False):
        require(Path(name).name==name,'flat output name')
        self.check(len(raw))
        with (self.path/name).open('ab' if append else 'xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        self.used+=len(raw)

    def json(self, name, value): self.put(name,(source.canonical(value)+'\n').encode())

    def emit(self, kind, **value):
        event=dict(index=self.events,previous_sha256=self.previous,kind=kind,**deepcopy(value))
        event['event_sha256']=source.digest(event)
        self.put('journal.jsonl',(source.canonical(event)+'\n').encode(),append=True)
        self.previous=event['event_sha256']; self.events+=1

    def png(self, name, image):
        from PIL import Image
        image=np.asarray(image)
        require(image.dtype==np.uint8 and image.shape==(224,224,3),'official224 uint8 RGB')
        stream=BytesIO(); Image.fromarray(image).save(stream,format='PNG')
        self.put(name,stream.getvalue())
        return dict(path=str(self.path/name),file_sha256=sha(self.path/name),rgb_sha256=core.sha(image.tobytes()))


def serial(value):
    if isinstance(value,np.ndarray): return serial(value.tolist())
    if isinstance(value,np.generic): return serial(value.item())
    if isinstance(value,dict): return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [serial(x) for x in value]
    require(value is None or type(value) in (str,bool,int,float),'JSON official info only')
    if type(value) is float: require(np.isfinite(value),'finite official values')
    return value


def pose_diagnostics(goal, state):
    """Pure saved-state arithmetic; independent diagnostic, not new admission.

    Native angular formula is deliberately preserved, including its negative
    result beyond one turn. Never use it to replace/override env termination.
    """
    goal=source.vector(goal); state=source.vector(state)
    delta=state-goal; absolute=float(abs(delta[4]))
    native_angle=float(min(absolute,2*np.pi-absolute))
    wrapped=float(abs(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))))
    allxy=float(np.linalg.norm(delta[:4])); native_success=bool(allxy<20 and native_angle<np.pi/9)
    wrapped_success=bool(allxy<20 and wrapped<np.pi/9)
    wrapped_delta=delta.copy();wrapped_delta[4]=float(np.arctan2(np.sin(delta[4]),np.cos(delta[4])))
    return dict(native_formula=dict(success=native_success,raw_angle_difference=absolute,
                angle_term=native_angle,state_distance=float(np.linalg.norm(delta)),negative_angle_term=native_angle<0),
                independent=dict(wrapped_angle_error=wrapped,all_xy_position_error=allxy,
                object_position_error=float(np.linalg.norm(delta[2:4])),agent_position_error=float(np.linalg.norm(delta[:2])),
                wrapped_state_delta=wrapped_delta.tolist(),wrapped_state_distance=float(np.linalg.norm(wrapped_delta)),
                same_allxy_threshold_wrapped_success=wrapped_success,
                label='diagnostic_only_not_official_or_admission'),
                native_vs_wrapped_success_disagreement=native_success!=wrapped_success)


def branch(plan, source_plan, env, store, counts):
    """Exact reset reconstruction then ten fixed float32 commands; no clipping."""
    row=dict(root_sha256=plan['root_sha256'],arm=plan['arm'],selection_sha256=plan['selection_sha256'],status='not_started',
             trace=[],checkpoints=[],initial_exact=False,attempted_steps=0,returned_steps=0,
             full_horizon=False,final_state=None,official_info=None,first_error=None,secondary_errors=[])
    body=env.unwrapped; raw=raw_commands(plan['commands']); active=plan['selection_sha256']
    guard=RgbObservabilityGuard(root_id=plan['root_sha256'],agent_rgb=(65,105,225))
    def emit(kind, **value): store.emit(kind,selection_sha256=active,**value)
    def retain(exc, phase):
        err=dict(phase=phase,type=type(exc).__name__,message=str(exc))
        if row['first_error'] is None: row['first_error']=err
        else: row['secondary_errors'].append(err)
        row['status']='failed'
    def call(kind, fn):
        store.check()
        counts[kind]['attempted']+=1; emit(kind+'_attempted',attempt=counts[kind]['attempted'])
        value=fn(); counts[kind]['completed']+=1
        emit(kind+'_returned',completed=counts[kind]['completed'])
        return value
    def reset(seed, options=None):
        value=call('reset',lambda:env.reset(seed=seed,**({} if options is None else {'options':options})))
        emit('reset_observation',seed=seed,options=serial(options),observation=serial(value[0]),info=serial({k:v for k,v in value[1].items() if k!='goal'}))
        return value
    def capture(step):
        image=source.rgb(call('render',env.render),source_plan)
        rec=dict(step=step,state=source.vector(body._get_obs()).tolist())
        rec['rgb_sha256']=core.sha(image.tobytes())
        emit('fresh_rgb_observed',receipt=rec)
        if step in (0,5,10):
            row['checkpoints'].append(rec)
            rec.update(store.png(f'{plan["candidate_id"]}_{plan["arm"]}_{step}.png',image))
            emit('checkpoint_saved',receipt=rec)
        guard.observe(image,primitive_step=step)
        emit('rgb_guard_observed',step=step,receipt=guard.receipt())
        return image
    try:
        row['status']='running'; emit('branch_started',plan=plan)
        require(sha(plan['raw_source_file'])==plan['raw_source_sha256'],'raw source binding')
        packet=source.load_raw_source(plan['raw_source_file'],source_plan)
        metadata=packet['metadata']; require(metadata==plan['source_metadata'] and metadata['root_sha256']==plan['root_sha256'],'source identity')
        slot=next(s for s in source_plan['slots'] if s['candidate_id']==plan['candidate_id'])
        selected=metadata['selected_attempt']; proposal=metadata['source']['proposal_index']; attempt=selected['attempt_index']
        seed=slot['reset_seed_span'][0]+proposal*source_plan['reset_attempts_per_proposal']+attempt
        require(seed==selected['random_reset_seed'] and selected['placement_reset_seed']==source_plan['placement_reset_seed'],'original seeds')
        initial=reset(seed); goal=source.vector(initial[1]['goal_state'])
        requested=source.sample_state(goal,list(body.block.shapes),slot,proposal,attempt,source_plan)
        emit('original_placement_request',state=requested.tolist(),goal=goal.tolist())
        placed=reset(source_plan['placement_reset_seed'],dict(state=requested.copy(),goal_state=goal.copy()))
        actual=source.vector(placed[0]['state']); actual_goal=source.vector(placed[1]['goal_state'])
        image=capture(0)
        exact=dict(state=np.array_equal(actual,packet['state']),goal=np.array_equal(actual_goal,packet['goal']),
                   rgb=np.array_equal(image,packet['rgb_history'][0]),goal_rgb=np.array_equal(placed[1]['goal'],packet['goal_rgb']),
                   root=source.root_hash(actual,actual_goal)==plan['root_sha256'],state_at_render=np.array_equal(actual,body._get_obs()))
        row['initial_checks']={k:bool(v) for k,v in exact.items()}; row['initial_exact']=all(exact.values())
        emit('initial_identity',checks=row['initial_checks'])
        require(row['initial_exact'],'initial state/goal/RGB/root mismatch: no dispatch')
        row['initial_pose_diagnostics']=pose_diagnostics(actual_goal,actual)
        emit('initial_pose_diagnostics',receipt=row['initial_pose_diagnostics'])
        for i,command in enumerate(raw,1):
            require(row['returned_steps']==i-1 and row['attempted_steps']==i-1,'sequential single dispatch only')
            require(command.dtype==np.float32 and command.shape==(2,) and np.isfinite(command).all() and (abs(command)<=1).all(),'bounded raw command guard')
            store.check(131072)
            authorized=guard.dispatch(command)
            require(np.array_equal(authorized,command),'guard cannot alter selected raw command')
            entry=dict(step=i,command=command.tolist(),attempted=True,returned=False);row['trace'].append(entry)
            row['attempted_steps']+=1;counts['step']['attempted']+=1
            emit('step_attempted',receipt=entry)
            transition=env.step(authorized)
            counts['step']['completed']+=1;row['returned_steps']+=1;entry['returned']=True
            emit('step_returned',step=i)
            require(isinstance(transition,(tuple,list)) and len(transition)==5,'five-field Gymnasium transition')
            obs,reward,terminated,truncated,info=transition
            entry.update(reward=float(reward),terminated=bool(terminated),truncated=bool(truncated))
            emit('termination_returned',step=i,reward=entry['reward'],terminated=entry['terminated'],truncated=entry['truncated'])
            entry['observation']=serial(obs)
            emit('returned_observation',step=i,observation=entry['observation'])
            entry['actual_state']=source.vector(body._get_obs()).tolist()
            row['final_state']=entry['actual_state']
            emit('actual_successor',step=i,state=entry['actual_state'])
            entry['official_info']=serial({k:v for k,v in info.items() if k!='goal'})
            row['official_info']=entry['official_info']
            emit('official_info_returned',step=i,info=entry['official_info'])
            entry['pose_diagnostics']=pose_diagnostics(actual_goal,entry['actual_state'])
            native_success,native_dist=body.eval_state(actual_goal,np.asarray(entry['actual_state']))
            entry['native_official_evaluation']=dict(success=bool(native_success),state_distance=float(native_dist))
            row['pose_diagnostics']=entry['pose_diagnostics'];row['official_evaluation']=entry['native_official_evaluation']
            emit('successor_evaluation',step=i,native=entry['native_official_evaluation'],diagnostics=entry['pose_diagnostics'])
            row.update(final_state=entry['actual_state'],official_info=entry['official_info'])
            entry['termination_protocol_risk']=dict(native_vs_wrapped_success_disagreement=entry['pose_diagnostics']['native_vs_wrapped_success_disagreement'],
                 negative_native_angle_term=entry['pose_diagnostics']['native_formula']['negative_angle_term'],
                 native_eval_vs_formula_disagreement=bool(native_success)!=entry['pose_diagnostics']['native_formula']['success'],
                 terminated_vs_eval_disagreement=bool(terminated)!=bool(native_success),
                 policy='honor_returned_terminated_truncated_no_override')
            emit('step_observation',receipt=entry)
            require(np.array_equal(obs['state'],entry['actual_state']),'returned/actual successor mismatch')
            capture(i)
            if terminated or truncated:
                row['status']='environment_terminated';break
        else: row['status']='completed'
        row['full_horizon']=row['returned_steps']==10
        if row['final_state'] is not None:
            state=np.asarray(row['final_state']); success,dist=body.eval_state(actual_goal,state)
            delta=state-actual_goal;delta[4]=(delta[4]+np.pi)%(2*np.pi)-np.pi
            row['official_evaluation']=dict(success=bool(success),state_distance=float(dist))
            row['wrapped_state_delta']=delta.tolist()
    except (Exception,KeyboardInterrupt) as exc:
        retain(exc,'branch')
        try: emit('branch_failed',first_error=row['first_error'],partial=row)
        except (Exception,KeyboardInterrupt) as secondary: retain(secondary,'failure_journal')
    row['endpoint_scope']='full_H2' if row['full_horizon'] and row['status']=='completed' else 'partial_or_unavailable'
    row['guard_receipt']=guard.receipt()
    row['endpoint_rgb_fresh']=bool(row['checkpoints'] and row['checkpoints'][-1].get('path') and
                                  row['checkpoints'][-1]['step']==row['returned_steps']==row['attempted_steps'])
    try: emit('branch_finished',receipt=row)
    except (Exception,KeyboardInterrupt) as exc: retain(exc,'final_journal')
    return row


def own_sources():
    import PIL
    paths={Path(__file__).resolve(),Path(source.__file__).resolve(),Path(witness.__file__).resolve(),Path(core.__file__).resolve(),
           Path(screen.__file__).resolve(),Path(upstream.__file__).resolve(),Path(PIL.__file__).resolve(),
           Path(inspect.getsourcefile(RgbObservabilityGuard)).resolve(),
           ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')}
    return {str(p):sha(p) for p in sorted(paths)}


def prepare(arguments, output):
    data=collect(**arguments)
    p=dict(schema=NAME+'.prepare',contract=contract(),inputs=data,sources=own_sources())
    store=Store(output); store.json('prepare.json',p)
    return p


def verify(folder, expected_sha):
    folder=Path(folder); require(sha(folder/'prepare.json')==expected_sha,'external prepare SHA')
    p=read(folder/'prepare.json'); require(p['schema']==NAME+'.prepare' and p['contract']==contract(),'fixed physical contract')
    bindings_check(p['sources']); bindings_check(p['inputs']['bindings'])
    require(collect(**p['inputs']['arguments'])==p['inputs'],'source/selection reconstruction')
    return p


def run(prepared, prepare_sha, output, factory, *, authorize_physics=False):
    require(authorize_physics is True,'MAIN explicit physics authorization')
    p=verify(prepared,prepare_sha); store=Store(output)
    counts={k:dict(attempted=0,completed=0) for k in ('reset','render','step')}
    rows=[dict(root_sha256=x['root_sha256'],arm=x['arm'],selection_sha256=x['selection_sha256'],status='not_started',trace=[]) for x in p['inputs']['plans']]
    result=dict(schema=NAME+'.report',prepare_sha256=prepare_sha,contract=contract(),status='running',rows=rows,counts=counts,
                errors=[],neural_calls=0,P1_admission=False,equal_compute=False,independent_confirmation=False,
                inherited_cost=p['inputs']['inherited_cost'],search_cost_by_arm=[dict(root_sha256=x['root_sha256'],arm=x['arm'],search_cost=x['search_cost']) for x in p['inputs']['plans']])
    try:
        store.emit('run_started',prepare_sha256=prepare_sha)
        for i,plan in enumerate(p['inputs']['plans']):
            env=None
            try:
                store.check(262144); env=factory(); body=env.unwrapped
                backend=inspect.getsourcefile(type(body))
                require(backend is not None and p['inputs']['source_plan']['source_closure'].get(str(Path(backend).resolve()))==p['inputs']['source_plan']['environment_sha256'],'actual native constructor source')
                require(body.relative is True and body.action_scale==100 and body.render_mode=='rgb_array' and body.render_size==224,'original native raw mapping/render contract')
                rows[i]=branch(plan,p['inputs']['source_plan'],env,store,counts)
            except (Exception,KeyboardInterrupt) as exc:
                rows[i].update(status='failed',first_error=dict(phase='branch_setup',type=type(exc).__name__,message=str(exc)))
            finally:
                if env is not None:
                    try: env.close()
                    except (Exception,KeyboardInterrupt) as exc:
                        rows[i].setdefault('secondary_errors',[]).append(dict(phase='close',type=type(exc).__name__,message=str(exc))); rows[i]['status']='failed'
                store.emit('branch_closed',receipt=rows[i])
        result['status']='physical_calibration_recorded' if all(x['status']=='completed' for x in rows) else 'partial_or_failed'
        if result['status']=='physical_calibration_recorded':
            require(counts=={k:dict(attempted=n,completed=n) for k,n in dict(reset=16,render=88,step=80).items()},'complete global physical census')
    except (Exception,KeyboardInterrupt) as exc:
        result['errors'].append(dict(phase='execution',type=type(exc).__name__,message=str(exc)));result['status']='partial_or_failed'
    finally:
        try: verify(prepared,prepare_sha)
        except (Exception,KeyboardInterrupt) as exc:
            result['errors'].append(dict(phase='final_bindings',type=type(exc).__name__,message=str(exc)));result['status']='partial_or_failed'
        semantics=p['inputs']['source_plan']['reset_semantics']
        result['reset_internal_physics_steps']=dict(completed_reset_implied=counts['reset']['completed']*semantics['internal_physics_steps_per_reset'],
            attempted_upper_bound=counts['reset']['attempted']*semantics['internal_physics_steps_per_reset'],actual_observed=None)
        result['internal_goal_renders_completed_reset_implied']=counts['reset']['completed']*semantics['internal_renders_per_reset']
        try:
            store.emit('run_finished',status=result['status'],counts=counts,errors=result['errors'])
            result['journal_sha256']=sha(store.path/'journal.jsonl')
            store.json('report.json',result)
        except (Exception,KeyboardInterrupt) as exc:
            result['errors'].append(dict(phase='final_write',type=type(exc).__name__,message=str(exc)));result['status']='partial_or_failed'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for name in ('witness-prepared','witness-prepare-sha','witness-report','witness-report-sha','direct-prepared','direct-prepare-sha','direct-report','direct-report-sha','sourcepack','sourcepack-sha','prepared','prepare-sha','output'):
        parser.add_argument('--'+name)
    parser.add_argument('--authorize-physics',action='store_true');a=parser.parse_args()
    if a.command=='prepare':
        args={k:getattr(a,k) for k in ('witness_prepared','witness_prepare_sha','witness_report','witness_report_sha','direct_prepared','direct_prepare_sha','direct_report','direct_report_sha','sourcepack','sourcepack_sha')}
        prepare(args,a.output);print(sha(Path(a.output)/'prepare.json'))
    elif a.command=='verify': print(verify(a.prepared,a.prepare_sha)['schema'])
    else:
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224)
        result=run(a.prepared,a.prepare_sha,a.output,factory,authorize_physics=a.authorize_physics)
        print(source.canonical(result))
        if result['status']!='physical_calibration_recorded':raise SystemExit(2)


if __name__=='__main__':main()
