"""Saved-only four moving sampling plans/physical labels; no neural or physics run."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_moving_overlay_recompile_comparison_20261027_v1 as math_saved

NAME='agent_baselines_resume_20261027_sampling_v2'
PLANNER='moving_prediction_matched_sampling_cuda_20261027_v1'
PHYSICAL='moving_prediction_matched_sampling_physical_20261027_v1'
GN_PLANNER='moving_history_planners_cuda_20261027_v1'
GN_PHYSICAL='p1_constructive_moving_history_physical_20261027_v1'
ARMS=('CEM','kernel_faithful_mppi_nominal')
GN_ARMS=('line_search_GN','full10_conditional_D2P')
RESOURCES=('predict_API','predictor','action_encoder','pred_proj')
CAPS=dict(environment_construct=4,environment_close=4,reset=8,set_state=16,goal_render=8,
          goal_bind=4,public_step=80,physics_step=816,explicit_render=88)
ROOT=Path(__file__).resolve().parents[2]
HELPER_SHA='f4ecb7b318771becb0d411ef594b9303c0981b653ea3a9af6fc11e3be6fb1ac5'
require,sha,parse,canonical=math_saved.require,math_saved.sha,math_saved.parse,math_saved.canonical
metrics,difference=math_saved.metrics,math_saved.difference
REFS=('physical_report','physical_prepare','physical_binding','physical_journal','physical_manifest',
      'physical_source','planner_report','planner_prepare','planner_source','planner_tests','planner_doc',
      'sourcepack','source_journal')
GN_REFS=('physical_report','physical_prepare','physical_binding','physical_journal','planner_handoff',
         'planner_report','planner_prepare','planner_source','planner_tests','planner_doc')
METRIC_ID='native7_distance_reward_and_original_success_plus_wrapped_object_diagnostics_v1'


def ident(value):return sha(canonical(value))


def keys(arms):return tuple((s,a) for s in (0,1) for a in arms)


def index(rows,arms,slot='root_slot'):
    require(isinstance(rows,list) and len(rows)==4,'fixed TWO roots x TWO arms, all four retained')
    out={}
    for r in rows:
        s=r[slot];k=(s,r['arm'])
        require(type(s) is int and k in keys(arms) and k not in out,'stable root/arm identity, no subset/duplicate')
        out[k]=r
    return out


def commands(value):
    a=np.asarray(value)
    require(a.shape==(10,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and
        (abs(a)<=1).all() and np.array_equal(a,a.astype(np.float32)),'literal raw float32 full10 commands, no repair')
    return a.astype(np.float32)


def budget(n):return {r:dict(attempted=n,completed=n) for r in RESOURCES}


def read_refs(refs,names,read):
    require(isinstance(refs,dict) and set(refs)==set(names),'exact external file reference set')
    values={}
    for name,ref in refs.items():
        require(set(ref)=={'file','sha256'} and isinstance(ref['file'],str) and Path(ref['file']).is_absolute() and
            '..' not in Path(ref['file']).parts,'explicit absolute external file/SHA')
        value=read(ref['file']);require(sha(value)==math_saved.digest(ref['sha256']),'external artifact SHA: '+name)
        values[name]=value
    return values


def read_json(data):
    require(len(data)<=4*1024**2,'bounded saved metadata JSON')
    return parse(data)


def own_consume(refs,planner,read):
    """Only the producer's saved NumPy consumer. No optimize/run/build experiment."""
    for key,path in (('planner_source',ROOT/'experiments/spectral_reachability'/(planner+'.py')),
                     ('planner_tests',ROOT/'tests'/('test_'+planner+'.py')),
                     ('planner_doc',ROOT/'innerdocs'/(planner+'.md'))):
        require(refs[key]['file']==str(path),'whitelisted own-consume source/test/doc path')
        require(sha(read(str(path)))==refs[key]['sha256'],'source seal BEFORE own consume import')
    module=importlib.import_module('experiments.spectral_reachability.'+planner)
    require(str(Path(module.__file__).resolve())==refs['planner_source']['file'],'whitelisted saved own consumer')
    return module.consume(Path(refs['planner_report']['file']).parent,refs['planner_report']['sha256'])


def planner_material(refs,blobs,consume,read):
    r=read_json(blobs['planner_report']);p=read_json(blobs['planner_prepare'])
    require(r['schema']==PLANNER+'.report' and r['status']=='matched_sampling_plans_sealed_not_admitted' and
        r['errors']==[] and r['first_error'] is None and r['commands_released'] is True and
        r['source_verified'] is True and r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'], 'complete actual sampling planner, same frozen model')
    require(p['schema']==PLANNER+'.prepare' and r['prepared']==refs['planner_prepare']['file'] and
        r['prepare_sha256']==refs['planner_prepare']['sha256'] and r['contract']==p['contract'], 'external planner prepare/contract')
    c=p['contract']
    require(c['roots']==2 and c['groups']==4 and c['arms']==list(ARMS) and
        c['iterations']==3 and c['samples']==24 and c['candidates_per_group']==72 and
        c['horizon']==2 and c['raw_steps']==10 and c['batch_size']==1 and
        all(c[f]==0 for f in ('extra_replay','padding','nominal_evaluations','AD','J','VJP','SVD','new_encoder','new_image_projector','new_physics')),
        'fixed 72 H2 only; no extra evaluations/replay/padding/AD')
    require(c['source_authority_sha256']==math_saved.SOURCEPACK_SHA and c['with_target'] is False and
        c['Q']=='original identity192 terminal mean192 squared error' and
        c['training'] is False and c['Qfit'] is False and c['MATH'] is True and c['TF32'] is False and
        c['autocast'] is False and c['CPU_neural_fallback'] is False,'original off source/goal/identity-Q and frozen CUDA numerics')
    require(r['resources']==budget(576) and r['predictor_samples']==dict(attempted=576,completed=576),
        'whole run actual API144x4 AND predictor samples144x4, not aggregate module sum')
    rows=index(r['rows'],ARMS,'source_slot')
    for row in rows.values():
        require(row['status']=='model_plan_sealed' and row['resources']==budget(144) and
            row['predictor_samples']==dict(attempted=144,completed=144),'each group exactly144 API/144 samples, no cross-group balancing')
    receipt=consume(refs,PLANNER,read)
    require(receipt['schema']==PLANNER+'.consume' and receipt['resources']==r['resources'] and
        receipt['predictor_samples']==r['predictor_samples'] and receipt['P1_admission'] is False and
        receipt['official_MPPI'] is False and receipt['equal_total_compute'] is False and
        receipt['prediction_calls_and_samples_matched_between_sampling_arms'] is True and
        receipt['inherited_encoding_once']==r['inherited_encoding_once']==c['inherited_encoding_once'], 'producer saved own-consume receipt')
    plans=index(receipt['rows'],ARMS,'source_slot');slots={s['source_slot']:s for s in p['slots']}
    require(set(slots)=={0,1} and len(p['slots'])==2,'two fixed task slots')
    for k,plan in plans.items():
        s,a=k;row=rows[k]
        require(plan['schema']==PLANNER+'.model_plan' and
            ident({f:v for f,v in plan.items() if f not in ('selection_sha256','whole_run_report_sha256')})==
                plan['selection_sha256']==row['selection_sha256'] and
            plan['whole_run_report_sha256']==refs['planner_report']['sha256'] and plan['commands']==row['commands'],
            'actual selected command seal tied to planner row/report')
        require(all(plan[f]==slots[s][f] for f in ('task_id','blind_sha256','task_export','task_sha256','task_contract_sha256')) and
            all(plan[f] is False for f in ('with_target','physical_execution','selection_uses_physics','P1_admission','official_MPPI')),
            'original blind task identities, prephysical frozen no outcomes')
        commands(plan['commands'])
        require(type(plan['selected_index']) is int and 0<=plan['selected_index']<72 and
            type(plan['model_terminal_cost']) in (int,float) and np.isfinite(plan['model_terminal_cost']), 'already scored finite selected candidate, not unscored nominal')
    return r,p,plans


def operation_counts(report,prepared,events,arms,legacy):
    """FINAL ABI: nested intent/terminal receipt, including environment_close."""
    if legacy:
        from experiments.spectral_reachability import moving_prediction_matched_sampling_comparison_20261027_v1 as old
        require(sha(Path(old.__file__).read_bytes())=='3948b036127ef292af2bf0f2baaa5e5c15f4a76b07043f1c5c1abd7b2650cbee',
                'frozen legacy reference helper')
        return old.operation_counts(report,prepared,events,arms,True)
    require(prepared['contract']['caps']==CAPS and set(report['counts'])==set(CAPS),'exact actual budget axes')
    intents={};terminals={};stack=[]
    counts={k:dict(attempted=0,completed=0) for k in CAPS}
    for e in events:
        kind=e['kind']
        require(kind not in ('cleanup_intent','cleanup_returned','operation_attempted','operation_completed'),
                'obsolete lifecycle ABI rejected, never synthesize legacy events')
        if kind not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        r=e['receipt'];i=r['operation_index']
        require(type(i) is int and r['resource'] in counts,'literal resource/operation_index')
        require(type(r['root_slot']) is int and (r['root_slot'],r['arm']) in keys(arms) and
                type(r['branch']) is int and r['branch']==list(keys(arms)).index((r['root_slot'],r['arm'])),
                'every operation original branch/root identity')
        if kind=='operation_dispatch_intent':
            require(i==len(intents) and r['status']=='pending','sequential pending operation index')
            intents[i]=r;stack.append(i)
        else:
            require(i in intents and i not in terminals and stack and stack.pop()==i and
                    r['status']==('returned' if kind=='operation_returned' else 'failed'),
                    'exact once nested terminal status/order')
            original={f:v for f,v in r.items() if f not in ('error_type','message')}
            require(dict(original,status='pending')==intents[i],'unchanged intent-to-terminal receipt identity')
            terminals[i]=r
    require(not stack and len(intents)==len(terminals),'no unverifiable pending operations')
    actual=[terminals[i] for i in range(len(terminals))]
    require(report['actual_operation_receipts']==actual,'actual receipts equal ALL journal terminals including close')
    for r in actual:
        c=counts[r['resource']];c['attempted']+=1;c['completed']+=int(r['status']=='returned')
    require(counts==report['counts'],'actual resource attempts/returns recount')
    for k,c in counts.items():
        require(all(type(c[f]) is int for f in ('attempted','completed')) and
                0<=c['completed']<=c['attempted']<=CAPS[k],'bounded literal callback counts')
    if report['status']!='partial_or_failed':
        require(all(c['attempted']==c['completed'] for c in counts.values()),'complete requires every actual return')
    return dict(actual=deepcopy(counts),journal_entry_return_lower_bounds=deepcopy(counts),
                journal_complete_returns=True,constructor_internal_scope=report['total_internal_physics_scope'],
                overlapping_axes_not_additive=True,operation_receipt_ABI='nested_operation_index_including_close')


def physical_material(report,prepared,events,targets,arms,frame,legacy=False):
    name=GN_PHYSICAL if legacy else PHYSICAL
    recorded='moving_history_physical_recorded' if legacy else 'matched_sampling_physical_recorded_not_admitted'
    require(report['schema']==name+'.report' and prepared['schema']==name+'.prepare' and
        report['status'] in (recorded,'partial_or_failed') and report['contract']==prepared['contract'] and
        report['sourcepack_sha256']==prepared['sourcepack_sha256']==math_saved.SOURCEPACK_SHA and
        report['contract']['with_target'] is False and report['contract']['own_horizon']==10 and
        report['contract']['warmup']==10,'NEW physical schema and original source/off/fullH2')
    rows=index(report['rows'],arms);plans=index(prepared['plans'],arms)
    require(events[0]['kind']=='run_started' and events[0]['prepare']==prepared and events[-1]['kind']=='run_finished' and
        all(events[-1][f]==report[f] for f in ('status','counts','errors')),'complete physical lifecycle/census/error preservation')
    closes={};starts={};observed={};guards={};transitions={};public={};pixels={k:0 for k in rows}
    for e in events:
        kind=e['kind']
        if kind=='branch_closed':
            r=e['receipt'];k=(r['root_slot'],r['arm']);require(k in rows and k not in closes and r==rows[k], 'actual whole row close receipt')
            closes[k]=r;continue
        if kind in ('branch_started','fresh_observation','guard_receipt','returned_transition','goal_attached','warmup_return'):
            k=(e['root_slot'],e['arm']);require(k in rows and e['selection_digest']==plans[k]['selection_digest'], 'stable journal branch selection')
        if kind=='branch_started':
            require(k not in starts and e['plan']==plans[k],'presealed plan dispatched once')
            starts[k]=e
            require(np.array_equal(commands(e['raw_warmup']),fixed_raw()) and e['raw_float32_bits']==fixed_raw().tobytes().hex(),'actual nonzero warmup bits')
        if kind=='goal_attached':require(e['goal']==targets[k[0]]['goal'],'original actual source goal, no endpoint goal fitting')
        if kind=='warmup_return':require(e['raw_float32_bits']==fixed_raw()[e['step']-1].tobytes().hex(),'actual warmup raw bits')
        if kind=='fresh_observation':
            r=e['receipt'];pk=(k,r['phase'],r['step']);require(pk not in observed,'unique actual observation')
            math_saved.png(frame(r['png']),r);math_saved.vector(r['state'])
            require(r['snapshot']['state']==r['state']==r['observation_state'],'true native7 observation/snapshot/RGB label bridge')
            observed[pk]=r;pixels[k]+=1
        if kind=='guard_receipt':
            pk=(k,e['phase'],e['step']);require(pk not in guards,'unique actual guard');guards[pk]=e
        if kind=='returned_transition':
            pk=(k,e['phase'],e['step']);require(pk not in transitions,'no repeated returned transition');transitions[pk]=e
        op=None
        if legacy and kind=='operation_attempted' and e['resource']=='public_step':op=e
        if not legacy and kind=='operation_dispatch_intent' and e['receipt']['resource']=='public_step':op=e['receipt']
        if op is not None:
            k=(op['root_slot'],op['arm']);phase,t=op['phase'],op['step'];require(k in plans and phase in ('warmup','plan') and
                type(t) is int and 1<=t<=10 and op['selection_digest']==plans[k]['selection_digest'],'command dispatch identity/step')
            expected=fixed_raw()[t-1] if phase=='warmup' else commands(plans[k]['commands'])[t-1]
            require(np.asarray(op['raw_action'],np.float32).tobytes()==expected.tobytes() and
                op['raw_action']==expected.astype(float).tolist(),'exact raw command intent, no rescale/clip/retry')
            pk=(k,phase,t);require(pk not in public,'no duplicate dispatch intent');public[pk]=op
    costs=operation_counts(report,prepared,events,arms,legacy)
    if report['status']!='partial_or_failed':
        counts={r:v['completed'] for r,v in report['counts'].items()}
        require(report['errors']==[] and counts['reset']==8 and counts['set_state']==16 and counts['goal_render']==8 and
            counts['goal_bind']==4 and counts['public_step']==sum(r['own_returned']+r['warmup_returned'] for r in rows.values()) and
            counts['physics_step']==16+10*counts['public_step'] and counts['explicit_render']==sum(r['PNG'] for r in rows.values()),
            'actual full/early-stop census, never selected-only counts')
    out=[]
    for k in keys(arms):
        row,plan=rows[k],plans[k];s,a=k;target=targets[s]
        require(row['selection_digest']==plan['selection_digest'] and ident({f:v for f,v in plan.items() if f!='selection_digest'})==plan['selection_digest'],
                'frozen physical selection seal')
        if not legacy:require(row['selected_plan']==plan,'selected plan report copy')
        if not legacy:
            own=[r for r in report['actual_operation_receipts'] if r['resource']=='public_step' and
                 (r['root_slot'],r['arm'])==k]
            require(all(r['selection_digest']==plan['selection_digest'] and r['branch']==row['branch_index'] for r in own),
                    'actual per-branch operation receipt identity')
            require(row['own_attempted']==sum(r['phase']=='plan' for r in own) and
                row['own_returned']==sum(r['phase']=='plan' and r['status']=='returned' for r in own) and
                row['warmup_returned']==sum(r['phase']=='warmup' and r['status']=='returned' for r in own),
                'per-branch actual callback attempted/returned counts, not inferred from transition logs')
        require(row['status'] in ('not_started','running','failed','horizon_completed','native_success','native_stopped'),'explicit physical row status')
        if k not in closes:require(row['status']=='not_started' and not row['checkpoints'],'unstarted denominator retained, not silently lost closure')
        if row['initial_exact']:require(row['goal']==target['goal'] and row['root_state']==target['initial_state'],'same original goal/root native7')
        require(row['PNG']==pixels[k],'all actual observation PNGs retained')
        cp={}
        for r in row['checkpoints']:
            t=r['step'];pk=(k,'plan',t)
            require(type(t) is int and t in (0,5,10) and t not in cp and r==observed.get(pk) and
                guards.get(pk,{}).get('accepted') is True and guards[pk]['render_preserved_dynamics'] is True,
                'actual checkpoint equals journal and accepted original RGB guard')
            if t==0:require(row['initial_exact'] and r['state']==target['initial_state'] and r['snapshot']==target['initial_snapshot'],'exact original initial body')
            else:require(pk in transitions and transitions[pk]['state']==r['state'],'checkpoint equals actual returned native state')
            cp[t]=r
        return_gaps={}
        for phase in ('warmup','plan'):
            ts=sorted(t for kk,ph,t in transitions if kk==k and ph==phase)
            charged=row['warmup_returned' if phase=='warmup' else 'own_returned']
            require(ts==list(range(1,len(ts)+1)) and len(ts)<=charged and
                (len(ts)==charged or report['status']=='partial_or_failed'), 'available returned trajectory vs actual callback denominator')
            return_gaps[phase]=charged-len(ts)
            for t in ts:
                pk=(k,phase,t);require(pk in public,'return needs predeclared raw command intent')
                if phase=='plan':
                    e=transitions[pk];require(e['command']==plan['commands'][t-1] and e['returned_goal_state']==target['goal'], 'actual returned command/original goal')
        require(row['own_returned']<=row['own_attempted']<=10 and row['full_horizon']==(row['own_returned']==10),'actual horizon not inferred from status')
        if row['status']=='horizon_completed':require(set(cp)=={0,5,10} and row['full_horizon'],'horizon completed requires actual checkpoints')
        if 10 in cp:require(row['full_horizon'],'no padded terminal label')
        values={t:metrics(target['goal'],cp[t]['state']) if t in cp else None for t in (0,5,10)}
        flag=row['native_success'];require(flag is None or type(flag) is bool,'explicit native flag or unavailable null')
        last=row.get('final_state');last_metrics=None if last is None else metrics(target['goal'],last)
        returned=transitions.get((k,'plan',row['own_returned']))
        if last is not None and returned is not None:require(last==returned['state'],'last returned native state equals actual journal label')
        out.append(dict(root_slot=s,arm=a,root_sha256=target['root_sha256'],sourcepack_sha256=report['sourcepack_sha256'],
            physical_selection_digest=row['selection_digest'],goal=target['goal'],initial_state=target['initial_state'],blind_sha256=target['blind_sha256'],horizon=2,raw_steps=10,
            condition='off',physical_metric=METRIC_ID,initial0=values[0],prefix5=values[5],terminal10=values[10],
            checkpoint_refs=[{f:r[f] for f in ('step','png','file_sha256','pixel_sha256')} for r in row['checkpoints']],
            missing_checkpoints=sorted({0,5,10}-set(cp)),source_status=row['status'],full_horizon=row['full_horizon'],
            native_success_reported=flag,last_returned_state=last,last_returned_metrics=last_metrics,
            last_returned_state_journal_verified=last is not None and returned is not None,
            native_flag_recomputation_mismatch=None if flag is None or last_metrics is None else flag!=last_metrics['native_success'],
            own_attempted=row['own_attempted'],own_returned=row['own_returned'],warmup_returned=row['warmup_returned'],
            actual_return_without_transition_label=return_gaps,
            first_error=row['first_error'],secondary_errors=row['secondary_errors'],
            actual_latent_Q=None,RGB_status='missing_not_encoded',
            prefix_retreat_native=difference(None if values[5] is None else values[5]['native_distance'],
                                             None if values[0] is None else values[0]['native_distance'])))
    census=dict(fixed_branch_denominator=4,complete_terminal10=sum(r['terminal10'] is not None for r in out),
        native_flags_true=sum(r['native_success_reported'] is True for r in out),
        missing_native_flags=sum(r['native_success_reported'] is None for r in out),
        full_horizons=sum(r['full_horizon'] for r in out),engineering_failures=sum(r['source_status']=='failed' for r in out),
        not_started=sum(r['source_status']=='not_started' for r in out),
        early_stops=sum(r['source_status'] in ('native_success','native_stopped') and not r['full_horizon'] for r in out))
    return dict(rows=out,physical_budget=costs,errors=report['errors'],source_status=report['status'],census=census,
                branch_denominator=4,root_denominator=2)


def fixed_raw():return np.asarray([[.1,0]]*5+[[0,.1]]*5,np.float32)


def paired(rows,arms):
    by=index(rows,arms);out=[]
    for s in (0,1):
        a,b=by[s,arms[0]],by[s,arms[1]]
        delta={f:difference(None if b['terminal10'] is None else b['terminal10'][f],
                            None if a['terminal10'] is None else a['terminal10'][f]) for f in
               ('native_distance','object_xy_l2','wrapped_angle_error','wrapped_joint_state_l2')}
        out.append(dict(root_slot=s,second_minus_first_terminal=delta,first_arm=arms[0],second_arm=arms[1],
            native_success_difference=difference(None if b['terminal10'] is None else int(b['terminal10']['native_success']),
                                                None if a['terminal10'] is None else int(a['terminal10']['native_success'])),
            actual_latent_Q_difference=None))
    return out


def relative(rows,reference):
    if reference is None:return dict(status='missing_reference_not_compared',pairs=None)
    ref=index(reference,GN_ARMS);pairs=[]
    fields=('root_sha256','sourcepack_sha256','goal','initial_state','blind_sha256','horizon','raw_steps','condition','physical_metric',
            'task_sha256','task_id','task_contract_sha256')
    for r in rows:
        for a in GN_ARMS:
            other=ref[r['root_slot'],a];mismatched=[f for f in fields if f not in r or f not in other or r[f]!=other[f]]
            pairs.append(dict(root_slot=r['root_slot'],arm=r['arm'],reference_arm=a,
                status='source_horizon_metric_mismatch_not_compared' if mismatched else 'descriptive_only_compute_unmatched',
                mismatched_fields=mismatched,budget_match=False,equal_FLOPs=False,
                native_distance_minus_reference=None if mismatched else difference(
                    None if r['terminal10'] is None else r['terminal10']['native_distance'],
                    None if other['terminal10'] is None else other['terminal10']['native_distance']),
                actual_latent_Q_difference=None))
    return dict(status='fixed_eight_descriptive_reference_pairs',pairs=pairs)


def frame_reader(refs,read,hashes):
    parent=Path(refs['physical_report']['file']).parent
    def frame(name):
        require(isinstance(name,str) and Path(name).name==name and name not in ('','.','..'),'flat PNG')
        path=str(parent/name);data=read(path);hashes[path]=sha(data);return data
    return frame


def lifecycle(refs,blobs,targets,arms,read,legacy=False):
    r=read_json(blobs['physical_report']);p=read_json(blobs['physical_prepare']);binding=read_json(blobs['physical_binding'])
    require(binding==dict(prepared=refs['physical_prepare']['file'],prepare_sha256=refs['physical_prepare']['sha256']), 'external physical execution binding')
    require(r['journal_sha256']==refs['physical_journal']['sha256'],'external physical journal file SHA')
    ev=math_saved.chain(blobs['physical_journal'],r['journal_chain_tail'])
    if not legacy:require(r['prepared']==binding['prepared'] and r['prepare_sha256']==binding['prepare_sha256'] and
                         ev[0]['execution_binding']==binding,'physical report/start actual prepare binding')
    frames={};result=physical_material(r,p,ev,targets,arms,frame_reader(refs,read,frames),legacy)
    result['frame_sha256']=frames
    return result,p


def reference_material(refs,targets,consume,read):
    b=read_refs(refs,GN_REFS,read);r=read_json(b['planner_report']);p=read_json(b['planner_prepare'])
    require(r['schema']==GN_PLANNER+'.report' and r['status']=='moving_history_plans_sealed_not_admitted' and
        r['errors']==[] and r['commands_released'] is True and r['model_unchanged'] is True and
        r['prepared']==refs['planner_prepare']['file'] and r['prepare_sha256']==refs['planner_prepare']['sha256'] and
        r['contract']==p['contract'],'original GN/D2P actual planner report/prepare')
    receipt=consume(refs,GN_PLANNER,read)
    require(receipt['schema']==GN_PLANNER+'.consume' and receipt['status']=='moving_commands_verified_not_admitted', 'original GN/D2P saved own consume')
    plans=index(receipt['plans'],GN_ARMS,'source_slot');handoff=read_json(b['planner_handoff'])
    require(handoff['schema']=='moving_history_planner_handoff.v1' and handoff['status']=='commands_verified_before_physics' and
        handoff['errors']==[] and handoff['projection']['source_report_sha256']==refs['planner_report']['sha256'], 'handoff SHA is NOT original CUDA report SHA')
    projected=index(handoff['data']['plans'],GN_ARMS,'source_slot')
    result,physical= lifecycle(refs,b,targets,GN_ARMS,read,legacy=True)
    frozen=index(physical['plans'],GN_ARMS);resources=index(r['rows'],GN_ARMS,'source_slot')
    for row in result['rows']:
        k=row['root_slot'],row['arm'];plan=plans[k];ph=frozen[k]
        require(projected[k]=={f:v for f,v in plan.items() if f!='prediction'} and
            ph['compiler_report']==refs['planner_handoff']['file'] and ph['compiler_report_SHA']==refs['planner_handoff']['sha256'] and
            physical['bindings'].get(ph['compiler_report'])==ph['compiler_report_SHA'] and
            ph['commands']==plan['commands'],'original full GN/D2P consume -> projected handoff -> physical commands')
        require(plan['blind_sha256']==row['blind_sha256'],'reference same original blind root')
        row.update({f:plan[f] for f in ('task_sha256','task_id','task_contract_sha256')})
        row['planner_budget']=dict(resources=resources[k]['resources'],AD_API_resources=resources[k]['AD_API_resources'],
            mathematical_VJP=r['contract']['mathematical_vjp_per_root'][row['arm']],equal_total_compute=False)
    result['external_files']=refs
    return result


def material(binding,binding_sha,read=math_saved.load_local,consume=own_consume):
    require(sha(Path(math_saved.__file__).read_bytes())==HELPER_SHA,'frozen pure math helper, never modify old comparison')
    raw=read(binding);require(sha(raw)==math_saved.digest(binding_sha),'external comparison binding SHA')
    b=read_json(raw);require(set(b)=={'schema','files','reference','rgb'} and b['schema']==NAME+'.binding','own exact binding schema')
    refs=b['files'];blobs=read_refs(refs,REFS,read)
    require(refs['sourcepack']['sha256']==math_saved.SOURCEPACK_SHA,'fixed original source authority')
    pack=read_json(blobs['sourcepack']);events=math_saved.chain(blobs['source_journal'],pack['journal_chain_tail'])
    targets=math_saved.source_targets(pack,events)
    planner,prepared,plans=planner_material(refs,blobs,consume,read)
    physical=read_json(blobs['physical_prepare']);manifest=read_json(blobs['physical_manifest'])
    require(physical['manifest']==refs['physical_manifest']['file'] and physical['manifest_sha256']==refs['physical_manifest']['sha256'] and
        manifest['schema']==PHYSICAL+'.selection' and manifest['sourcepack']==refs['sourcepack']['file'] and
        manifest['sourcepack_sha256']==refs['sourcepack']['sha256'] and manifest['selection_fixed_before_physics'] is True and
        manifest['physical_outcome_selection'] is False and manifest['arms']==list(ARMS),'original source and prephysical manifest')
    authority=manifest['authorities'];require(len(authority)==1 and authority[0]['authority']==PLANNER and
        authority[0]['output']==str(Path(refs['planner_report']['file']).parent) and
        authority[0]['report_sha256']==refs['planner_report']['sha256'] and
        all(authority[0][f]==refs[key]['sha256'] for f,key in
            (('source_sha256','planner_source'),('tests_sha256','planner_tests'),('innerdoc_sha256','planner_doc'))), 'manifest actual planner/code authority')
    for key in ('physical_source','planner_source','planner_tests','planner_doc','planner_report','planner_prepare','sourcepack','source_journal'):
        require(physical['bindings'].get(refs[key]['file'])==refs[key]['sha256'],'physical prepare frozen dependency '+key)
    result,physical=lifecycle(refs,blobs,targets,ARMS,read)
    frozen=index(physical['plans'],ARMS)
    for row in result['rows']:
        k=row['root_slot'],row['arm'];plan=plans[k];ph=frozen[k]
        require(ph['root_sha256']==row['root_sha256'] and ph['blind_sha256']==row['blind_sha256'] and
            ph['compiler_authority']==PLANNER and ph['compiler_report']==refs['planner_report']['file'] and
            ph['compiler_report_sha256']==refs['planner_report']['sha256'] and ph['compiler_selection_sha256']==plan['selection_sha256'] and
            ph['own_plan_sha256']==ident(plan) and all(ph[f]==plan[f] for f in
                ('commands','task_id','blind_sha256','task_export','task_sha256','task_contract_sha256')), 'exact original root -> planner winner -> physical plan join')
        row.update({f:plan[f] for f in ('task_sha256','task_id','task_contract_sha256','model_terminal_cost','selected_index')})
        row['planner_budget']=dict(resources=budget(144),predictor_samples=dict(attempted=144,completed=144),batch_size=1,
            candidates_H2=72,AD=0,J=0,mathematical_VJP=0,SVD=0,solve=0,new_encoder=0,new_image_projector=0,
            action_encoder_context_rows=432,pred_proj_context_rows=432,extra_replay=0,nominal_evaluations=0)
    reference=None if b['reference'] is None else reference_material(b['reference'],targets,consume,read)
    result.update(schema=NAME+'.report',status='saved_sampling_comparison_not_admitted',
        paired_sampling=paired(result['rows'],ARMS),relative_GN_D2P=relative(result['rows'],None if reference is None else reference['rows']),
        GN_D2P_reference=reference,planner_resources=planner['resources'],predictor_samples=planner['predictor_samples'],
        inherited_encoding_once=planner['inherited_encoding_once'],planner_contract=planner['contract'],
        sampling_prediction_API_and_samples_matched=True,equal_total_FLOPs=False,equal_total_compute=False,
        P1_admission=False,P3_admission=False,independent_confirmation=False,new_neural_calls=0,new_physics_calls=0,
        actual_latent_Q_status='missing_not_encoded',external_files=refs,binding=dict(file=binding,sha256=binding_sha),
        source_sha256=sha(Path(__file__).read_bytes()),
        MPPI_dispatch='NEW nominal-guided sampling: best of72 evaluated samples, NOT final nominal; not official MPPI',
        action_verification_scope='saved dispatch raw and returned command; actual latest_action equality remains producer native check')
    # Reject concurrent mutations of all explicit external inputs before release.
    read_refs(refs,REFS,read)
    if b['reference'] is not None:read_refs(b['reference'],GN_REFS,read)
    frames=dict(result['frame_sha256'])
    if reference is not None:frames.update(reference['frame_sha256'])
    for path,h in frames.items():require(sha(read(path))==h,'final saved PNG SHA, no concurrent mutation')
    attach_rgb(result,b['rgb'],read)
    result['evidence_matrix']=evidence_matrix(result,read)
    return result


# Saved-only extensions. No optimizer, environment factory, torch.load or CUDA entry.
RGB_NAME='moving_prediction_matched_sampling_rgb_cuda_20261027_v2'
RGB_SOURCE_SHA='dde4a59a92583161e488c906aaffe64f5f1e33c3be6bfcae9c46bd8172c83acc'
REPORTED_SHA='52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1'
LOCAL_SHA='96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e'
PAPER_PROTOCOL={
 'grasp':dict(model='DINO-WM for PushT; exact checkpoint unknown',observations='RGB goal; exact history unknown',
              action='5-action prediction blocks; table H mapping unknown',feedback='open_loop',
              metric_domain='learned_world_model_only',training='world model pretrained; per-env/H optimizer sweep',
              sample_unit='500 planning trials/setting Table1; Table3/4 denominator not inferred'),
 'hilewm':dict(model='learned high level + frozen low LeWM; exact table checkpoint unknown',
               observations='current/goal RGB',action='macro latent32; max chunk15; low CEM',
               feedback='staged OR online per row configuration',metric_domain='PushT-v1/OGBCube-v0 success',
               training='12.5M trained high +18M frozen low; continuous15epochs; VQ50',
               sample_unit='50 episodes/config, usually3 seeds; Table4 SD across seeds'),
 'lewm_v3':dict(model='LeWM or PLDM per row; exact checkpoint unknown',observations='PushT RGB history3 general protocol',
                action='5-action block',feedback='general H5/25-step execute then replan; cell binding unknown',
                metric_domain='PushT task SR',training='world model trained; not our invocation',
                sample_unit='Table10 seed count/pm semantics unknown; do not inherit Table5'),
 'planwam':dict(model='DINOv2 ViT-S frozen base + rank32 LoRA and planning modules trained',
                observations='4history x4cameras 1148x672',action='64 trajectories x8waypoints/4seconds',
                feedback='NAVSIM proposal OR HUGSIM closed loop',metric_domain='EPDMS/PDMS/RC/HDScore not PushT success',
                training='two stages50epochs each,16xA800',sample_unit='NAVSIM12146 navtest; HUGSIM436 scenario weighted')
}


def attach_rgb(result,ref,read):
    if ref is None:
        result['RGB_evidence']=dict(status='missing',actual_latent_Q=None)
        return
    require(set(ref)=={'file','sha256'},'explicit actual RGB report reference')
    path=Path(ref['file']);require(path.is_absolute(),'absolute RGB report')
    raw=read(str(path));require(sha(raw)==math_saved.digest(ref['sha256']),'external RGB report SHA')
    r=read_json(raw);require(r['schema']==RGB_NAME+'.report' and
        r['input_binding']['physical_report']==result['external_files']['physical_report']['file'] and
        r['input_binding']['physical_sha']==result['external_files']['physical_report']['sha256'],
        'RGB belongs to exact new physical invocation')
    modulepath=ROOT/'experiments/spectral_reachability'/(RGB_NAME+'.py')
    require(sha(read(str(modulepath)))==RGB_SOURCE_SHA,'frozen RGB saved consumer BEFORE import')
    module=importlib.import_module('experiments.spectral_reachability.'+RGB_NAME)
    receipt=module.consume(str(path.parent),ref['sha256']) # pure saved arrays only
    attach_rgb_rows(result,receipt)
    result['RGB_evidence'].update(external_report=ref,source_sha256=RGB_SOURCE_SHA)
    require(sha(read(str(path)))==ref['sha256'],'final RGB report unchanged')


def attach_rgb_rows(result,receipt):
    require(receipt['schema']==RGB_NAME+'.consume' and receipt['P1_admission'] is False and
            receipt['new_independent_samples']==0 and
            receipt['counts']=={k:dict(attempted=14,completed=14) for k in ('encoder','projector')},
            'actual14 encodings:12 checkpoints+2 original goals, not twelve total')
    by=index(receipt['rows'],ARMS)
    for row in result['rows']:
        k=row['root_slot'],row['arm'];rgb=by[k]
        # selection is recorded below by physical_material, never joined by array ordinal.
        require(rgb['selection_digest']==row['physical_selection_digest'] and rgb['task_id']==row['task_id'],
                'RGB original physical selection/task identity')
        cp={c['step']:c for c in rgb['checkpoints']}
        require(set(cp)=={0,5,10} and all(c['status']=='encoded' and type(c['Q']) in (int,float) and
                np.isfinite(c['Q']) and c['Q']>=0 for c in cp.values()),'all true checkpoint Q, no terminal substitution')
        row['actual_latent_Q']={str(t):cp[t]['Q'] for t in (0,5,10)}
        row['RGB_status']='actual_saved_native192_Q_recomputed'
    for pair in result['paired_sampling']:
        s=pair['root_slot'];rows=index(result['rows'],ARMS)
        pair['actual_latent_Q_difference']=rows[s,ARMS[1]]['actual_latent_Q']['10']-rows[s,ARMS[0]]['actual_latent_Q']['10']
    result['actual_latent_Q_status']='saved_RGB_v2_full_artifact_and_raw_NPY_verified'
    result['RGB_evidence']=dict(status='actual_saved_verified',resources=receipt['counts'],
                               checkpoint_images=12,source_goal_images=2,new_encoding_calls=0)


def matrix_from_reported(paper,local_rows,comparison):
    helper=ROOT/'experiments/spectral_reachability/d2p_baseline_reported_values_20261027_v1.py'
    require(sha(helper.read_bytes())=='30b1694e6537f15364ec1cc1da5b038fe749f8783800c60d20e6d5ab8a7461fb',
            'frozen reported-cell validator')
    from experiments.spectral_reachability import d2p_baseline_reported_values_20261027_v1 as reported
    reported.validate_report(paper)
    authors=[]
    for row in paper['rows']:
        rec=row['record'];s=row['source']
        authors.append(dict(id=row['id'],paper=rec['paper'],method=row['method'],
            author_reported=dict(value=None if rec['value']=='unknown' else rec['value'],unit=rec['unit'],
                uncertainty=rec['uncertainty'],literal=s['reported_token']),
            local_actual=None,protocol=dict(record=rec,supplement=PAPER_PROTOCOL[row['paper_key']]),
            provenance=s,version=rec['version'],classification='context_only',
            improvement_over_local=None,protocol_match=False,paired_inference=False))
    require(len(authors)==238 and len({x['id'] for x in authors})==238,'whole existing author coverage, no cherry picking')
    wanted=('moving_gradient_open_loop','CEM','kernel_faithful_mppi_nominal',
            'fixed_H2_kinodynamic_beam16x4','GRASP_H3_bounded','joint_H2_spectral_packet_mapped')
    require(len(local_rows)==12 and {(r['source_slot'],r['arm']) for r in local_rows}==
            {(s,a) for s in (0,1) for a in wanted},'six prior local variants x two known roots')
    local=[]
    for row in local_rows:
        local.append(dict(root_slot=row['source_slot'],arm=row['arm'],author_reported=None,
            local_actual={k:row[k] for k in ('model_Q','native_success','full_horizon','physical_returned_steps',
                'realQ_root','realQ_prefix5','realQ_terminal10','model_status','physical_status','rgb_status','resources')},
            identity={k:row[k] for k in ('blind_sha256','task_id','task_sha256','commands_sha256')},
            implementation_identity='local_variant_not_official_paper_reproduction',
            validation_scope='pinned prior saved assembler output; not a new full upstream replay',
            paper_gain=None,protocol_match=False))
    return dict(author_reported_cells=authors,prior_local_actual=local,
        current_prediction_matched_local_actual=comparison['rows'],
        local_official_paper_reproductions={p:None for p in ('GRASP','Hi-LeWM','LeWM','PlanWAM')},
        mismatched_axes=['root/goal distribution','model/checkpoint','observation and rendering',
            'action space and horizon','feedback','success/metric domain','training','budget and sampling unit'],
        eligible_cross_paper_improvements=0,improvement_values=None,
        missing=['official Hi-LeWM high/low paired weights+normalizer and native loader not executed',
                 'official GRASP protocol not reproduced by local bounded variant',
                 'Table10 per-solver full resource and statistical binding unknown',
                 'PlanWAM driving metric is not native PushT task success',
                 'fresh independent roots and P1 mechanism admission absent'],
        independent_unit='two known roots; four current arms are NOT four independent roots',
        official_MPPI=False,equal_total_FLOPs=False,P1_admission=False)


def evidence_matrix(comparison,read):
    paper_path=ROOT/'results/p2_reported_baseline_table_20261027_v3/export/report.json'
    local_path=ROOT/'results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json'
    paper_raw=read(str(paper_path));local_raw=read(str(local_path))
    require(sha(paper_raw)==REPORTED_SHA and sha(local_raw)==LOCAL_SHA,'fixed paper export and prior actual snapshot')
    # Verify cached primary bytes used by every cell, rather than trusting a transcribed value.
    paper=read_json(paper_raw)
    archive=paper_path.parent.parent;checked={}
    for row in paper['rows']:
        s=row['source'];pairs=[(s['html_file'],s['html_sha256'])]
        if row['paper_key']!='planwam':
            pairs.append(('sources/'+{'grasp':'grasp_pdf.pdf','hilewm':'hilewm_pdf.pdf',
                          'lewm_v3':'lewm_v3_pdf.pdf'}[row['paper_key']],s['pdf_sha256']))
        for file,h in pairs:
            require(not Path(file).is_absolute() and '..' not in Path(file).parts,'safe archived primary path')
            if file not in checked:
                require(sha(read(str(archive/file)))==h,'archived primary file hash');checked[file]=h
    local=read_json(local_raw)
    require(local['schema']=='p1_moving_baseline_comparison_20261027_v3.comparison' and
            local['status']=='comparison_recorded_not_admitted' and
            local['official_reproduction'] is local['P1_admission'] is local['equal_total_compute'] is False,
            'prior local evidence not paper reproduction')
    result=matrix_from_reported(paper,local['rows'],comparison)
    result['inputs']=dict(paper_export=dict(file=str(paper_path),sha256=REPORTED_SHA),
                         prior_local=dict(file=str(local_path),sha256=LOCAL_SHA),primary_files=checked)
    for path,h in ((paper_path,REPORTED_SHA),(local_path,LOCAL_SHA)):
        require(sha(read(str(path)))==h,'final matrix input snapshot unchanged')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binding',required=True);p.add_argument('--binding-sha',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);result=material(a.binding,a.binding_sha)
    h=math_saved.write_new(a.output,result)
    print(canonical(dict(path=a.output,sha256=h,status=result['status'],P1_admission=False)).decode())
    return 0


if __name__=='__main__':raise SystemExit(main())

