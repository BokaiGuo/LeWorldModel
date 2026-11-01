"""Pure saved U/D/S/N moving-two-root model/physical descriptive assembler.

No model loading, optimizer, encoding, simulator, render or task selection.
Input authority is the scheduler's own saved consume and an external-SHA-bound
Meitner physical report. Missing RGB is never replaced with a geometry proxy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import moving_history_budget_scheduler_cuda_20261027_v1 as scheduler

NAME='moving_budget_scheduler_comparison_20261027_v1'
PHYSICAL='moving_budget_scheduler_physical_20261027_v1'
SCHEDULER_SHA='f4f4d5e013e3fc3aac896da58faaafadcf5e2d4e53f5f2f7b61203707e09ff9d'
ROOT=scheduler.ROOT
METHODS=scheduler.METHODS
core,require,sha,read,canonical=scheduler.core,scheduler.require,scheduler.sha,scheduler.read,scheduler.canonical
CAP,FLOOR=16*1024**2,200*1024**2
FLAGS=dict(P1_admission=False,P2_admission=False,P3_admission=False,independent_confirmation=False,
    full_C3_demonstrated=False,compute_advantage=False,walltime_advantage=False)
RESOURCES=('reset','public_step','explicit_render','goal_render','set_state','physics_step','goal_bind','environment_construct','environment_close')
LEGAL_STATUS=('horizon_completed','native_success','native_stopped')
MISSING_RGB=dict(status='missing_not_encoded',actual_same_Q=None,actual_native192=None,
    predicted_actual_same_Q_rank=None,reason='No externally sealed RGB encoding receipt is in this v1 contract; no new neural calls.')


def digest(value):return core.digest(value)


def exact_commands(a,b):
    a=np.asarray(a);b=np.asarray(b)
    require(a.shape==b.shape==(10,2) and np.isfinite(a).all() and np.max(np.abs(a))<=1 and
        np.array_equal(a,a.astype(np.float32)) and np.array_equal(a,b),'literal float32 selected raw10, no repair/reselection')


def identity(plan,slot_key):
    slot=plan[slot_key]
    require(type(slot) is int and slot in (0,1) and plan['arm'] in METHODS,'strict fixed root slot/arm')
    for k in ('root_sha256','task_id','blind_sha256'):core.hash_value(plan[k])
    return slot,plan['root_sha256'],plan['task_id'],plan['blind_sha256'],plan['arm']


def index_unique(rows,slot_key):
    out={}
    for row in rows:
        key=identity(row,slot_key);require(key not in out,'duplicate stable root/task/blind/arm');out[key]=row
    require(len(out)==8 and {(k[0],k[-1]) for k in out}=={(s,m) for s in (0,1) for m in METHODS},'ALL eight fixed root/strategy cells')
    return out


def metrics(goal,state):
    goal=np.asarray(goal,np.float64);state=np.asarray(state,np.float64)
    require(goal.shape==state.shape==(7,) and np.isfinite(goal).all() and np.isfinite(state).all(),'actual finite native state7/goal7')
    d=state-goal;angle=float(abs(d[4]));native_angle=min(angle,2*np.pi-angle)
    wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))));xy=float(np.linalg.norm(d[:4]));wrapped_d=d.copy();wrapped_d[4]=wrapped
    return dict(native_success=bool(xy<20 and native_angle<np.pi/9),native_distance=float(np.linalg.norm(d)),
        native_angle_term=float(native_angle),negative_native_angle=bool(native_angle<0),
        wrapped_allxy_success=bool(xy<20 and wrapped<np.pi/9),wrapped_object_angle_error=wrapped,
        object_position_error=float(np.linalg.norm(d[2:4])),object_pose=state[2:5].tolist(),object_success=None,
        native_xy_l2=xy,wrapped_joint_state_l2=float(np.linalg.norm(wrapped_d)),
        remaining_native_state_squared_error=float(d[5:]@d[5:]),
        native_wrapped_success_disagree=bool((xy<20 and native_angle<np.pi/9)!=(xy<20 and wrapped<np.pi/9)))


def check_recorded_metrics(goal,state,recorded):
    out=metrics(goal,state)
    require(set(recorded)=={'native_success','native_distance','native_angle_term','negative_native_angle',
        'wrapped_allxy_success','wrapped_object_angle_error','object_position_error','object_pose','object_success'},'exact native kernel metric schema')
    require(recorded=={k:out[k] for k in recorded},'saved native/wrapped metrics recomputed without simulation')
    return out


def reconcile_journal(events,report,prepared):
    """Saved operation/frame binding only, never call a native environment."""
    require(events and events[0]['kind']=='run_started' and events[0]['prepare']==prepared and
        events[0]['execution_binding']==dict(prepared=report['prepared'],prepare_sha256=report['prepare_sha256']), 'physical execution prepare binding')
    require(events[-1]['kind']=='run_finished' and events[-1]['counts']==report['counts'] and events[-1]['errors']==[] and events[-1]['status']==report['status'],'terminal physical census')
    rows={r['branch_index']:r for r in report['rows']}
    require(len(rows)==8 and set(rows)==set(range(8)) and all(type(k) is int for k in rows),'eight unique physical branch IDs')
    totals={k:dict(attempted=0,completed=0) for k in RESOURCES}
    branch={i:{k:dict(attempted=0,completed=0) for k in RESOURCES} for i in rows}
    step_counts={i:{p:dict(attempted=0,completed=0) for p in ('warmup','plan')} for i in rows}
    internal={i:{p:dict(attempted=0,completed=0) for p in ('reset','warmup','plan')} for i in rows}
    transitions={i:[] for i in rows};observations={i:[] for i in rows};guards={};closed=set();started=set();stacks={i:[] for i in rows}
    for e in events:
        if e['kind']=='branch_closed':
            # Producer deliberately emits an unscoped terminal receipt.
            # Identity is inside receipt.branch_index, NEVER its array position.
            b=e['receipt']['branch_index'];require(type(b) is int and b in rows and b not in closed and e['receipt']==rows[b] and not stacks[b],'complete branch close receipt')
            closed.add(b);continue
        if 'branch' not in e:continue
        b=e['branch'];require(type(b) is int and b in rows,'literal physical journal branch')
        row=rows[b];plan=row['selected_plan']
        require(e['root_slot']==row['root_slot'] and e['arm']==row['arm'] and e['selection_digest']==row['selection_digest'],'journal stable branch/selection identity')
        kind=e['kind']
        if kind=='branch_started':
            require(b not in started and e['plan']==plan,'frozen selected plan before dispatch');started.add(b)
            require(np.array_equal(e['raw_warmup'],scheduler.moving.fixed_raw()) and e['raw_float32_bits']==scheduler.moving.fixed_raw().tobytes().hex(),'original moving warmup raw bits')
        elif kind in ('operation_attempted','operation_completed'):
            resource=e['resource'];require(resource in totals,'known physical counter')
            stage='attempted' if kind=='operation_attempted' else 'completed';totals[resource][stage]+=1;branch[b][resource][stage]+=1
            if stage=='attempted':
                if resource=='physics_step':
                    parents=[x for x in stacks[b] if x['resource'] in ('reset','public_step')]
                    require(len(parents)==1,'internal step has an observed native parent')
                    parent=parents[0];phase='reset' if parent['resource']=='reset' else parent['phase']
                    internal[b][phase]['attempted']+=1
                stacks[b].append(e)
            else:
                require(stacks[b] and stacks[b][-1]['resource']==resource,'nested native attempt/return, no silent retries')
                previous=stacks[b].pop();require({k:v for k,v in previous.items() if k not in ('kind','index','previous_sha256','event_sha256')}==
                    {k:v for k,v in e.items() if k not in ('kind','index','previous_sha256','event_sha256')},'same native operation on return')
                if resource=='physics_step':
                    parent=next(x for x in reversed(stacks[b]) if x['resource'] in ('reset','public_step'))
                    internal[b]['reset' if parent['resource']=='reset' else parent['phase']]['completed']+=1
            if resource=='public_step':
                phase=e['phase'];require(phase in ('warmup','plan'),'warmup vs selected explicitly counted')
                step_counts[b][phase][stage]+=1;step=e['step'];require(type(step) is int and 1<=step<=10,'native actual step index')
                wanted=scheduler.moving.fixed_raw()[step-1] if phase=='warmup' else np.asarray(plan['commands'],np.float32)[step-1]
                require(np.array_equal(e['raw_action'],wanted),'every actual raw dispatched command, no clip/substitution')
        elif kind=='returned_transition':transitions[b].append(e)
        elif kind=='fresh_observation':observations[b].append(e['receipt'])
        elif kind=='guard_receipt':
            key=(b,e['phase'],e['step']);require(key not in guards,'unique observation guard');guards[key]=e
            require(e['accepted'] is True and e['render_preserved_dynamics'] is True and e['canonical_agent_pixels']>0,'actual fresh RGB/state/pure-render guard')
    require(closed==started==set(rows) and totals==report['counts'],'ALL8 actual branches and full native operation ledger')
    for b,row in rows.items():
        n=row['own_returned'];require(step_counts[b]['warmup']==dict(attempted=10,completed=10) and step_counts[b]['plan']==dict(attempted=n,completed=n),'real warmup vs selected dispatch denominator')
        require(internal[b]['reset']==dict(attempted=4,completed=4) and internal[b]['warmup']==dict(attempted=100,completed=100) and internal[b]['plan']==dict(attempted=10*n,completed=10*n),'reset/warmup/selected internal physics separately observed')
        plan_steps=[e for e in transitions[b] if e['phase']=='plan'];warm=[e for e in transitions[b] if e['phase']=='warmup']
        require([e['step'] for e in warm]==list(range(1,11)) and [e['step'] for e in plan_steps]==list(range(1,n+1)),'every returned native transition')
        for e in plan_steps:
            require(e['returned_goal_state']==row['goal'] and e['command']==row['selected_plan']['commands'][e['step']-1],'returned target/raw selected command unchanged')
            actual=metrics(row['goal'],e['state']);require(e['reward']==-actual['native_distance'] and e['terminated']==actual['native_success'],'native reward vs success, no velocity threshold added')
        require(plan_steps[-1]['state']==row['final_state'] and bool(plan_steps[-1]['terminated'])==row['terminated'] and bool(plan_steps[-1]['truncated'])==row['truncated'],'actual last returned state and endpoint flags')
        obs=observations[b];expected=[('warmup',s) for s in range(11)]+[('plan',s) for s in range(n+1)]
        require([(o['phase'],o['step']) for o in obs]==expected and len(obs)==row['PNG'],'all actual fresh frames; no padded early-stop endpoint')
        for o in obs:
            require((b,o['phase'],o['step']) in guards and o['state']==o['observation_state'] and o['snapshot']['state']==o['state'],'observed state vs saved snapshot')
        require([o for o in obs if o['phase']=='plan' and o['step'] in (0,5,10)]==row['checkpoints'],'checkpoint identity, never old endpoint')
        require(next(o for o in obs if o['phase']=='plan' and o['step']==0)['state']==row['root_state'],'actual initial native state')
    return dict(totals=totals,by_branch=branch,public_steps_by_phase=step_counts,internal_physics_by_phase=internal,observations=observations)


def descriptive(rows,all_full10):
    keys=('model_Q','native_distance','object_position_error','wrapped_object_angle_error','wrapped_joint_state_l2')
    means={};differences={}
    for method in METHODS:
        selected=[r for r in rows if r['arm']==method]
        means[method]={k:float(np.mean([r['model_Q'] if k=='model_Q' else r['actual_final'][k] for r in selected])) for k in keys}
        means[method]['native_success_count']=sum(r['actual_final']['native_success'] for r in selected)
        means[method]['wrapped_success_count']=sum(r['actual_final']['wrapped_allxy_success'] for r in selected)
        means[method]['root_denominator']=2
    for reference in (METHODS[0],METHODS[2],METHODS[3]):
        per_root=[]
        for s in (0,1):
            d=next(r for r in rows if r['source_slot']==s and r['arm']==METHODS[1]);r=next(r for r in rows if r['source_slot']==s and r['arm']==reference)
            per_root.append(dict(source_slot=s,root_sha256=d['root_sha256'],model_Q_D_minus_reference=d['model_Q']-r['model_Q'],
                actual_native_distance_D_minus_reference=d['actual_final']['native_distance']-r['actual_final']['native_distance'],
                actual_object_position_error_D_minus_reference=d['actual_final']['object_position_error']-r['actual_final']['object_position_error'],
                actual_wrapped_angle_D_minus_reference=d['actual_final']['wrapped_object_angle_error']-r['actual_final']['wrapped_object_angle_error'],
                native_success_D_minus_reference=int(d['actual_final']['native_success'])-int(r['actual_final']['native_success']),
                initial_minus_final_native_gain_D_minus_reference=d['native_distance_reduction']-r['native_distance_reduction']))
        differences[reference]=dict(per_root=per_root,means={k:float(np.mean([r[k] for r in per_root])) for k in per_root[0] if k not in ('source_slot','root_sha256')},
            actual_same_Q_difference=deepcopy(MISSING_RGB))
    return dict(status='two_known_roots_descriptive_only',method_means=means,D_minus_reference=differences,
        endpoint_semantics='all8_fresh_full10' if all_full10 else 'native_guarded_endpoints_include_early_stop_not_common_H2_terminal',
        full10_paired_summary='available' if all_full10 else 'not_available_do_not_mix_native_stop_with_full10',
        uncertainty='n=2 reused roots; no p-value, CI, independent confirmation or generalization claim',
        scale_warning='Model Q is NOT pose/native reward/success or measured physical gain; do not pool units.')


def assemble_saved(receipt,scheduler_report,scheduler_prepare,panel,physical,physical_prepare,events):
    """Pure array/JSON assembly after external hashes and own-consume validation."""
    require(receipt['schema']==scheduler.NAME+'.consume' and receipt['status']=='four_policy_commands_verified_not_admitted' and receipt['errors']==[],'scheduler own complete consume')
    require(physical['schema']==PHYSICAL+'.report' and physical['status']=='moving_budget_scheduler_physical_recorded' and physical['errors']==[] and
        physical['fixed_branch_denominator']==8 and physical['new_neural_calls']==0 and physical['training'] is False,'complete fixed8 physical engineering report, not all8 task success')
    require(physical_prepare['schema']==PHYSICAL+'.prepare' and physical['contract']==physical_prepare['contract'] and
        physical['sourcepack_sha256']==scheduler.SOURCEPACK_SHA==physical_prepare['sourcepack_sha256'] and physical['source_prepare_id']==physical_prepare['source_prepare_id'],'same moving source and physical contract')
    for k in ('P1_admission','P3_admission','independent_confirmation','compute_advantage'):require(physical[k] is False and receipt[k] is False,'no scientific upgrade')
    require(scheduler_report['resources']=={k:dict(attempted=v,completed=v) for k,v in scheduler.MAXIMA.items()} and scheduler_prepare['protocol']==scheduler.protocol(),'complete520/2J384/2SVD contract')
    model_index=index_unique(receipt['plans'],'source_slot');prepared_index=index_unique(physical_prepare['plans'],'root_slot')
    selected_index=index_unique([r['selected_plan'] for r in physical['rows']],'root_slot')
    require(set(model_index)==set(prepared_index)==set(selected_index),'stable root/task/blind/arm join; never array order')
    ledger=reconcile_journal(events,physical,physical_prepare);rows=[]
    for method in METHODS:
        blocks=panel['allocations'][method]
        for s in (0,1):
            plan=next(p for k,p in model_index.items() if k[0]==s and k[-1]==method);key=identity(plan,'source_slot')
            pp=prepared_index[key];require(pp==selected_index[key],'physical selected plan identical to pre-frozen prepare')
            content={k:v for k,v in plan.items() if k not in ('selection_sha256','whole_run_report_sha256')}
            require(digest(content)==plan['selection_sha256'] and pp['compiler_selection_sha256']==plan['selection_sha256'] and pp['own_plan_sha256']==digest(plan),'model to physical selection seal')
            require(pp['selection_digest']==digest({k:v for k,v in pp.items() if k!='selection_digest'}) and pp['compiler_authority']==scheduler.NAME and pp['compiler_report_sha256']==plan['whole_run_report_sha256'],'frozen scheduler provenance, not old baseline endpoint')
            exact_commands(pp['commands'],plan['commands'])
            actual=[r for r in physical['rows'] if identity(r['selected_plan'],'root_slot')==key];require(len(actual)==1,'unique actual branch')
            actual=actual[0];b=actual['branch_index']
            require(actual['root_slot']==s and actual['arm']==method and actual['selection_digest']==pp['selection_digest'] and actual['selected_plan']==pp,'actual physical row identity')
            require(actual['initial_exact'] is True and actual['first_error'] is None and actual['secondary_errors']==[] and actual['status'] in LEGAL_STATUS and
                actual['constructor_internal_work']=='sealed_native_pristine_no_physics' and actual['own_attempted']==actual['own_returned'] and 1<=actual['own_returned']<=10,'no failed/unknown branch hidden in pair')
            require(actual['full_horizon']==(actual['own_returned']==10),'full10 vs native-stop denominator')
            initial=check_recorded_metrics(actual['goal'],actual['root_state'],actual['initial_metrics']);final=check_recorded_metrics(actual['goal'],actual['final_state'],actual['final_metrics'])
            require(actual['native_success']==final['native_success'],'native success flag vs actual state')
            resources=scheduler_report['scope_resources'][method][str(s)]
            require(all(resources[k]==dict(attempted=6*blocks[s]+4,completed=6*blocks[s]+4) for k in scheduler.NEURAL),'selected root actual quota+replays')
            rows.append(dict(source_slot=s,root_sha256=plan['root_sha256'],task_id=plan['task_id'],blind_sha256=plan['blind_sha256'],arm=method,
                model_selection_sha256=plan['selection_sha256'],physical_selection_digest=pp['selection_digest'],candidate_index=plan['candidate_index'],
                model_Q=plan['model_cost'],allocated_blocks=blocks[s],evaluated_H2_candidates=blocks[s]*3,
                planner_exclusive_resources=resources,physical_resources=ledger['by_branch'][b],
                actual_public_steps=ledger['public_steps_by_phase'][b],actual_internal_physics=ledger['internal_physics_by_phase'][b],
                native_steps=actual['own_returned'],full_horizon=actual['full_horizon'],endpoint_scope=actual['endpoint_scope'],
                native_status=actual['status'],terminated=actual['terminated'],truncated=actual['truncated'],
                actual_initial=initial,actual_final=final,native_distance_reduction=initial['native_distance']-final['native_distance'],
                raw_goal=actual['goal'],raw_initial_state=actual['root_state'],raw_final_state=actual['final_state'],
                checkpoint_references=actual['checkpoints'],actual_RGB_sameQ=deepcopy(MISSING_RGB)))
    equivalence=[]
    for s in (0,1):
        rr=[r for r in rows if r['source_slot']==s]
        require(all(r['raw_initial_state']==rr[0]['raw_initial_state'] and r['raw_goal']==rr[0]['raw_goal'] and r['root_sha256']==rr[0]['root_sha256'] for r in rr),'four methods same actual root and goal')
        commands=[next(p['commands'] for k,p in model_index.items() if k[0]==s and k[-1]==method) for method in METHODS]
        equivalence.append(dict(source_slot=s,root_sha256=rr[0]['root_sha256'],
            blocks={r['arm']:r['allocated_blocks'] for r in rr},
            same_literal_commands=all(c==commands[0] for c in commands),same_model_Q=all(r['model_Q']==rr[0]['model_Q'] for r in rr),
            same_actual_final_state=all(r['raw_final_state']==rr[0]['raw_final_state'] for r in rr),
            independent_physical_dispatches=4))
    return dict(schema=NAME+'.report',status='saved_comparison_assembled_not_admitted',errors=[],known_root_calibration=True,fixed_branch_denominator=8,
        rows=rows,descriptive=descriptive(rows,all(r['full_horizon'] for r in rows)),
        resources=dict(new_comparison_neural_calls=0,new_comparison_physics_calls=0,
            scheduler_actual=scheduler_report['resources'],scheduler_scope=scheduler_report['scope_resources'],
            shared_signal_once=scheduler_report['scope_resources']['shared'],
            standalone_logical_not_actual_extra=scheduler_prepare['protocol']['standalone_logical_predict'],
            original_encoder_acquisition=scheduler_prepare['encoder_receipt'],
            original_anchor_search_receipts=scheduler_prepare['historical_anchor_receipts'],
            scheduler_limits=scheduler_report['limits'],physical_actual=ledger['totals'],
            total_internal_physics_scope=physical['total_internal_physics_scope'],
            physical_wall_seconds=physical['wall_seconds'],
            processed_action_context_tokens=1560,engine_backward=scheduler_report['actual_backward_observation'],
            cost_warning='520 is actual panel work, shared8 once; U/D/S logical136 vs N128 not equal total vectors. Historical288predict and8visual acquisition remain separate. No FLOP/wall equivalence.'),
        allocations=deepcopy(panel['allocations']),signals=deepcopy([x['signal'] for x in panel['signals']]),policy_equivalence=equivalence,
        RGB_status=deepcopy(MISSING_RGB),
        interpretation=dict(native_reward='negative raw joint-state L2, including remaining native state coordinates; not a success probability',
            native_success='original all-four-xy norm<20 and original angle expression<pi/9; NO velocity success condition',
            native_angle_risk='native min(abs(delta),2pi-abs(delta)) may be negative; wrapped metrics reported separately without changing env/thresholds',
            zero_success='0/8 cannot establish a theoretical impossibility or eliminate model/proxy/protocol causes',
            quota_collision='When all integer allocations coincide, same-bank same-budget policies may dispatch identical commands. This is not evidence of missed dispatch or a P3 theoretical negative.',
            study_scope='static fixed-H2 cross-root scheduler; not full C3, online reconfiguration, independent task confirmation or P3 scientific admission'),**FLAGS)


def inputs(scheduler_output,scheduler_report_sha,physical_output,physical_report_sha,physical_source_sha):
    """Read only saved artifacts; physical source is hashed, NEVER imported."""
    for h in (scheduler_report_sha,physical_report_sha,physical_source_sha):core.hash_value(h)
    scheduler_output=Path(scheduler_output).resolve();physical_output=Path(physical_output).resolve();bindings={}
    def bind(path,h):
        path=Path(path);require(not path.is_symlink(),'regular saved binding, not symlink');path=path.resolve();require(sha(path)==h,'external/closure SHA: '+str(path))
        require(str(path) not in bindings or bindings[str(path)]==h,'nonconflicting artifact binding');bindings[str(path)]=h;return path
    bind(scheduler.__file__,SCHEDULER_SHA)
    receipt=scheduler.consume(scheduler_output,scheduler_report_sha)
    sr=read(bind(scheduler_output/'report.json',scheduler_report_sha));sp=read(bind(sr['prepared'],sr['prepare_sha256']))
    panel=read(bind(scheduler_output/'panel.json.gz',sr['artifact_sha256']['panel.json.gz']),32*1024**2)
    physical_source=ROOT/'experiments/spectral_reachability'/(PHYSICAL+'.py');bind(physical_source,physical_source_sha)
    pr=read(bind(physical_output/'report.json',physical_report_sha),2*1024**2);pp=read(bind(pr['prepared'],pr['prepare_sha256']),256*1024)
    require(pp['bindings'][str(physical_source.resolve())]==physical_source_sha,'actual physical prepare pins external final producer source')
    manifest=read(bind(pp['manifest'],pp['manifest_sha256']),256*1024)
    require(manifest['schema']==PHYSICAL+'.selection' and manifest['arms']==list(METHODS) and manifest['selection_fixed_before_physics'] is True and manifest['physical_outcome_selection'] is False and
        manifest['authorities']==[dict(authority=scheduler.NAME,output=str(scheduler_output),report_sha256=scheduler_report_sha)],'one actual scheduler authority frozen before physics')
    require(read(bind(physical_output/'execution_binding.json',sha(physical_output/'execution_binding.json')))==dict(prepared=pr['prepared'],prepare_sha256=pr['prepare_sha256']),'physical binding receipt')
    for path,h in sp['bindings_sha256'].items():bind(path,h)
    for path,h in pp['bindings'].items():bind(path,h)
    for name,h in sr['artifact_sha256'].items():
        require(Path(name).name==name,'flat scheduler artifact');bind(scheduler_output/name,h)
    journal=bind(physical_output/'journal.jsonl',pr['journal_sha256']);require(journal.stat().st_size<=CAP,'bounded actual physical journal')
    previous=None;events=[]
    with journal.open('rb') as f:
        for index,raw in enumerate(f):
            require(raw.endswith(b'\n'),'complete durable event line');e=scheduler.blind.decode(raw)
            require(type(e['index']) is int and e['index']==index and e['previous_sha256']==previous and e['event_sha256']==digest({k:v for k,v in e.items() if k!='event_sha256'}),'physical durable event chain')
            previous=e['event_sha256'];events.append(e)
    require(previous==pr['journal_chain_tail'],'physical terminal journal hashchain')
    # Hash saved fresh frames, do not render/decode/re-encode or copy them.
    for e in events:
        if e['kind']=='fresh_observation':
            frame=e['receipt'];name=frame['png'];require(Path(name).name==name and name not in ('','.','..'),'flat fresh PNG receipt');bind(physical_output/name,frame['file_sha256'])
    result=assemble_saved(receipt,sr,sp,panel,pr,pp,events)
    for path,h in bindings.items():require(sha(path)==h,'final saved-input TOCTOU')
    result['inputs']=dict(scheduler_output=str(scheduler_output),scheduler_report_sha256=scheduler_report_sha,
        physical_output=str(physical_output),physical_report_sha256=physical_report_sha,physical_source_sha256=physical_source_sha,
        scheduler_prepare_sha256=sr['prepare_sha256'],physical_prepare_sha256=pr['prepare_sha256'],bindings_sha256=bindings)
    result['implementation_sha256']={str(p.resolve()):sha(p) for p in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'))}
    return result


def write_new(output,result):
    output=Path(output).absolute();payload=canonical(result)
    require(not output.exists() and len(payload)<=CAP and shutil.disk_usage(output.parent).free>=FLOOR+max(CAP,len(payload)), 'NEW16MiB bounded output +200MiB floor')
    output.mkdir()
    with (output/'report.json').open('xb') as f:require(f.write(payload)==len(payload),'short comparison write');f.flush();os.fsync(f.fileno())
    fd=os.open(output,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(output=str(output),report_sha256=sha(output/'report.json'),bytes=len(payload),status=result['status'],**FLAGS)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scheduler-output','scheduler-report-sha','physical-output','physical-report-sha','physical-source-sha','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();arguments=vars(a);output=arguments.pop('output')
    result=inputs(**arguments);print(canonical(write_new(output,result)).decode(),end='')


if __name__=='__main__':main()
