"""Source-bound five-arm saved gradient-MPC comparison; arithmetic only.

No new neural/physics work, outcome selection or composition admission.
AD overhead and the gradient planner's explicit MATH policy are NOT erased.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p2_sra_matched_comparison_20261027_v1 as previous
from experiments.spectral_reachability import p2_gradient_mpc_physical_20261027_v1 as physical
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

TAG = 'p2_gradient_mpc_matched_comparison_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
GRADIENT_RUN = 'p2_gradient_mpc_physical_20261027_v1_cuda_main_v1'
planner = physical.planner
GRADIENT_ARM = planner.ARM
BASE_ARMS = ('cem','random',previous.previous.mppi.sampling.ARM,previous.SRA_ARM)
ARMS = (GRADIENT_ARM,*BASE_ARMS)
METRICS,STATE_COORDINATES = previous.METRICS,previous.STATE_COORDINATES
evidence,require,index = previous.evidence,previous.require,previous.index
MATH_POLICY = 'CUDA SDPA MATH for all candidates and winner replay'


def load_inputs(project,reader):
    """Only the five explicit known calibration arms; never glob future runs."""
    bundle = previous.load_inputs(project,reader)
    directory = Path(project)/'results'/GRADIENT_RUN
    report,run_protocol = (reader.load(directory/name) for name in ('report.json','protocol.json'))
    require(report['schema']==physical.TAG+'.report','gradient physical schema mismatch')
    prepared_dir = Path(run_protocol['prepared'])
    protocol,prepared = (reader.load(prepared_dir/name) for name in ('protocol.json','prepared.json'))
    reader.bindings(protocol['bindings'])
    reader.seals(prepared['source_seals'])
    verified,sealed,context = physical.load_prepared(prepared_dir)
    require(verified==protocol and sealed==prepared,'gradient physical prepare changed')
    require(run_protocol['prepared_protocol_sha256']==report['prepared_protocol_sha256']==
        prepared['protocol_sha256']==physical.digest(protocol),'gradient run/prepare digest mismatch')
    require(report['source_seals']==prepared['source_seals']
        and run_protocol['plans']==protocol['plans']==context[5]
        and run_protocol['encode_terminals'] is True,'gradient source/plan/encoding request mismatch')
    planning_dir = Path(protocol['planning'])
    planning,planning_protocol,planning_prepared = (reader.load(planning_dir/name)
        for name in ('report.json','protocol.json','prepared.json'))
    reader.bindings(planning_protocol['bindings'])
    reader.seals(planning_prepared['source_seals'])
    require(context[2]==bundle['compilation'],'shared compilation identity required')
    reference = bundle['sra']['planning_protocol']['context']
    require(planning_protocol['context']['normalization']==reference['normalization']
        and planning_protocol['context']['objective']==reference['objective'],
        'gradient normalization/goal0Q mismatch')
    journals = {}
    planned = index(planning['rows'],('root_sha256',))
    for case in planning_protocol['context']['cases']:
        key,i = case['root_sha256'],case['root_index']
        require(reader.load(planning_dir/f'root_{i}.json')==planned[key,],
            'persisted gradient planning row differs')
        path = planning_dir/f'root_{i}_journal.jsonl'
        reader.bind(path)
        journals[key] = [json.loads(line,object_pairs_hook=evidence.strict_object)
            for line in path.read_text().splitlines()]
    bundle['gradient'] = dict(directory=directory,report=report,run_protocol=run_protocol,
        protocol=protocol,prepared=prepared,plans=context[5],planning=planning,
        planning_protocol=planning_protocol,planning_prepared=planning_prepared,journals=journals)
    return bundle


def validate_planning(group,compilation):
    protocol,prepared,report = group['planning_protocol'],group['planning_prepared'],group['planning']
    require(protocol['schema']==planner.TAG+'.protocol' and protocol['schedule']==planner.SCHEDULE
        and protocol['schedule']['attention_backend']==MATH_POLICY
        and protocol['boundary']==planner.BOUNDARY and protocol['frozen_before_gradient_cuda'] is True,
        'frozen gradient algorithm/MATH policy changed')
    required = {'all_four_retained','all_complete','actual336','actual_encoder336','actual_context1008',
        'backward156','updates156','model_unchanged','frozen_eval'}
    require(report['schema']==planner.TAG+'.report' and report['status']=='calibration_completed'
        and report['protocol_sha256']==physical.digest(protocol)
        and report['source_seals']==prepared['source_seals']
        and required<=report['gates'].keys() and all(v is True for v in report['gates'].values()),
        'complete source-bound gradient planning gates required')
    cases = index(protocol['context']['cases'],('root_sha256',))
    rows = index(report['rows'],('root_sha256',))
    compiled = index(compilation['rows'],('root_sha256',))
    plans = index(group['plans'],('root_sha256','method'))
    require(len(rows)==4 and rows.keys()==cases.keys()==compiled.keys()
        and set(plans)=={(key[0],GRADIENT_ARM) for key in rows},'gradient stable four-root census mismatch')
    require(group['protocol']['planning_accounting']==group['report']['planning_accounting']==physical.ACCOUNTING,
        'gradient physical AD accounting differs')
    for (key,),row in rows.items():
        case,saved = cases[key,],compiled[key,]['compiled']
        initial = saved['input_context']
        expected = dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
            action_dim=10,latent_history=initial['emb'][0],action_history=initial['actions'][0][1:])
        require(case['root']['context']==expected
            and case['metric_sha256']==saved['scoring']['metric']['sha256']
            and case['goal_vector_sha256']==physical.digest(saved['goal_vectors'][0])
            and case['projection_sha256']==physical.digest(saved['scoring']['metric']['projection']),
            'original full root/goal/Q binding required')
        actions = planner.verify_result(row,protocol,prepared,saved)
        result = row['result']
        rebound = physical.plan_for(case,actions,result['selected_candidate_id'],
            result['witness']['receipt']['receipt_sha256'])
        require(rebound==plans[key,GRADIENT_ARM],'gradient plan differs from stable saved winner')
        for field,value in planner.BOUNDARY.items():
            require(row[field]==result[field]==report[field]==value,'gradient claim boundary changed')
        require(row['predict_calls']==row['action_encoder_calls']==84
            and row['processed_context_token_samples']==252,'gradient actual work mismatch')
        require(result['selection_evaluations']==result['attempted_candidate_evaluations']==40
            and result['planning_predict_samples']==80 and result['replay_predict_samples']==4,
            'all forty gradient candidates / explicit80+4 counters required')
    for field,value in dict(actual_predict_calls=336,actual_action_encoder_calls=336,
            actual_processed_context_token_samples=1008,completed_autograd_grad_calls=156,
            completed_scalar_objective_vjp_directions=156,actual_optimizer_steps=156).items():
        require(report[field]==value,'gradient aggregate accounting mismatch: '+field)
    return rows


def validate_ad_journal(row,events):
    """All40 costs/packet dispatches and39 backward request/completion pairs."""
    attempts = row['result']['attempts']
    require([e for e in events if 'candidate_id' in e]==attempts,
        'gradient journal candidate records differ')
    packets = [e for e in events if e.get('kind')=='candidate_packet']
    require(len(packets)==80 and [{k:v for k,v in e.items() if k not in ('kind','candidate_index')}
        for e in packets]==row['result']['ledger'][:80],'gradient journal forward ledger differs')
    costs = [e for e in events if e.get('kind')=='candidate_cost']
    require(costs==[dict(kind='candidate_cost',candidate_index=i,cost=e['cost'])
        for i,e in enumerate(attempts)],'gradient journal candidate costs differ')
    derivatives = [e for e in events if e.get('kind')=='autograd_grad']
    require(len(derivatives)==78,'exact39 AD request/completion pairs required')
    for i in range(39):
        requested,completed = derivatives[2*i:2*i+2]
        require(requested==dict(kind='autograd_grad',candidate_index=i,completed=False,
            scalar_objective_vjp_directions_requested=1,extra_forward_requested=0)
            and completed==attempts[i]['derivative'] and completed['completed'] is True,
            'gradient journal backward operation differs')
    # The order itself is evidence: two packets, score, optional AD pair, then
    # the retained candidate. No delayed/extra unused final backward is valid.
    cursor = 0
    for i in range(40):
        segment = events[cursor:cursor+(6 if i<39 else 4)]
        kinds = [e.get('kind','candidate') for e in segment]
        expected = ['candidate_packet','candidate_packet','candidate_cost']
        if i<39: expected += ['autograd_grad','autograd_grad']
        expected += ['candidate']
        require(kinds==expected,'gradient planning event chronology differs')
        cursor += len(expected)
    require(cursor==len(events),'unaccounted gradient journal work')


def audit_artifacts(bundle,reader):
    previous.audit_artifacts(bundle,reader)
    group = bundle['gradient']
    previous.previous.audit_artifacts(group,reader)
    report = group['report']
    required = {'all_four_retained','branches_durable','frozen_eval','model_unchanged',
        'sources_inputs_unchanged','trace_matches_selected'}
    require(required<=report['gates'].keys() and report['full_H2_all'] is True
        and report['actual_physical_steps']==40 and report['attempted_terminal_encode_samples']==4,
        'complete fixed-H2 gradient physics/encodings required; no guarded endpoint substitution')
    physical_report = reader.load(group['directory']/'physical_report.json')
    require(physical_report['schema']==physical.TAG+'.physical_before_encoding'
        and physical_report['planning_accounting']==report['planning_accounting']==physical.ACCOUNTING,
        'durable gradient overhead differs')
    for field in ('P1_admission','P2_admission','physical_efficacy_verified'):
        require(report[field] is False,'gradient physical claim boundary changed')
    for row in group['planning']['rows']:
        validate_ad_journal(row,group['journals'][row['root_sha256']])


def build_rows(bundle):
    rows = previous.build_rows(bundle)
    # Retain the old four-arm arithmetic and resource checks without changing it.
    previous.compare_rows(rows)
    group = bundle['gradient']
    planned = validate_planning(group,bundle['compilation'])
    compiled = index(bundle['compilation']['rows'],('root_sha256',))
    plans = index(group['plans'],('root_sha256','method'))
    branches = index(group['report']['rows'],('root_sha256','method'))
    encodings = index(group['report']['encodings'],('root_sha256','method'))
    require(plans.keys()==branches.keys()==encodings.keys(),'gradient plan/branch/encoding census mismatch')
    for (key,arm),plan in plans.items():
        branch = branches[key,arm]
        require(arm==GRADIENT_ARM and branch['root_index']==plan['root_index'],'gradient root index/arm mismatch')
        row = previous.previous.validate_saved_branch(branch,plan,encodings[key,arm],
            compiled[key,]['compiled'],planned[key,]['result']['selected_model_cost'])
        row['budget'].update(compilation_predict_samples=0,total_predict_samples=84,
            autograd_grad_calls=39,scalar_objective_vjp_directions=39,action_optimizer_steps=39,
            model_optimizer_steps=0,jacobian_calls=0,svd_calls=0,scalar_output_vjp_directions=None)
        row['backend_policy'] = MATH_POLICY
        row['saved_planning_wall_time_s'] = evidence.finite(planned[key,]['actual_root_wall_time_s'])
        rows.append(row)
    report = group['report']
    require(report['full_H2_all'] is True and report['actual_physical_steps']==40
        and report['attempted_terminal_encode_samples']==4,'gradient fixed-H2 totals mismatch')
    for row in rows:
        if row['arm']!=GRADIENT_ARM:
            row['budget'].update(autograd_grad_calls=None,scalar_objective_vjp_directions=None,
                action_optimizer_steps=None,model_optimizer_steps=0)
            row['backend_policy'] = 'not explicitly forced/recorded MATH in reference protocol; actual backend unknown'
    return rows


def compare_rows(rows):
    lookup = index(rows,('root_sha256','arm'))
    roots = sorted({r['root_sha256'] for r in rows})
    require(len(roots)==4 and set(lookup)=={(r,a) for r in roots for a in ARMS},
        'four stable roots / five arms required; no filtering')
    old_rows = [r for r in rows if r['arm']!=GRADIENT_ARM]
    previous.compare_rows(old_rows)
    pairs = []
    shared = dict(full_H2_candidates=40,planning_predict_samples=80,replay_predict_samples=4,
        physical_env_steps=10,terminal_encode_samples=1)
    gradient_budget = dict(compilation_predict_samples=0,total_predict_samples=84,action_encoder_calls=84,
        processed_context_tokens=252,autograd_grad_calls=39,scalar_objective_vjp_directions=39,
        action_optimizer_steps=39,model_optimizer_steps=0,jacobian_calls=0,svd_calls=0,
        scalar_output_vjp_directions=None)
    for root in roots:
        arms = {a:lookup[root,a] for a in ARMS}
        for field in ('root_binding_sha256','action_domain','metric_sha256','goal_vectors_sha256','physical_identity'):
            require(all(arms[a][field]==arms[GRADIENT_ARM][field] for a in ARMS),
                'cross-arm '+field+' mismatch')
        for arm,row in arms.items():
            require(all(row['budget'][k]==v for k,v in shared.items()),'shared H2/predict/replay/physical units mismatch')
            if arm==GRADIENT_ARM:
                require(all(row['budget'][k]==v for k,v in gradient_budget.items()),'gradient AD overhead accounting mismatch')
                require(row['backend_policy']==MATH_POLICY,'gradient MATH backend policy changed')
            else:
                require(row['budget']['autograd_grad_calls'] is None
                    and row['budget']['scalar_objective_vjp_directions'] is None
                    and row['backend_policy']=='not explicitly forced/recorded MATH in reference protocol; actual backend unknown',
                    'unmeasured reference AD/backend cannot be invented')
        pairs.append(dict(root_sha256=root,arms=deepcopy(arms),
            gradient_minus={a:{m:arms[GRADIENT_ARM][m]-arms[a][m] for m in METRICS} for a in BASE_ARMS},
            gradient_minus_state_absolute_error={a:(np.asarray(arms[GRADIENT_ARM]['state_absolute_error'])-
                np.asarray(arms[a]['state_absolute_error'])).tolist() for a in BASE_ARMS}))
    return dict(schema=TAG+'.report',status='saved_calibration_comparison_completed',evidence_integrity_valid=True,
        paired_roots=4,branches=20,rows=pairs,
        paired_summaries={a:{m:previous.previous.describe([r['gradient_minus'][a][m] for r in pairs],
            higher_better=m in ('official_success','object_only_success')) for m in METRICS} for a in BASE_ARMS},
        paired_state_error_summaries={a:{c:previous.previous.describe([
            r['gradient_minus_state_absolute_error'][a][i] for r in pairs])
            for i,c in enumerate(STATE_COORDINATES)} for a in BASE_ARMS},
        state_absolute_error_coordinates=list(STATE_COORDINATES),
        state_error_scope='raw seven-state absolute coordinate errors; raw angle not circular, mixed units not pooled',
        arm_means={a:{m:float(np.mean([lookup[r,a][m] for r in roots])) for m in METRICS} for a in ARMS},
        official_successes={a:sum(lookup[r,a]['official_success'] for r in roots) for a in ARMS},
        cost_accounting=dict(per_root={a:deepcopy(lookup[roots[0],a]['budget']) for a in ARMS},
            four_root_total_predict_samples={a:sum(lookup[r,a]['budget']['total_predict_samples'] for r in roots) for a in ARMS},
            gradient_autograd_grad_calls=156,gradient_scalar_objective_vjp_directions=156,
            gradient_action_optimizer_steps=156,gradient_model_optimizer_steps=0,
            sra_jacobian_calls=4,sra_svd_calls=8,sra_scalar_output_vjp_directions=768,
            vjp_scope='gradient156 scalar-objective VJP directions vs SRA768 scalar-output directions: NOT backward-kernel or FLOP-equivalent counts',
            array_verification=deepcopy(previous.sra.ACCOUNTING['verification_work']),
            unknown_is_not_zero=True),
        backend_contracts={a:lookup[roots[0],a]['backend_policy'] for a in ARMS},
        same_backend_established=False,backend_scope='gradient explicit MATH search/replay is a sealed policy, other actual backends are unknown; no bitwise cross-arm or runtime-equivalence claim',
        gradient_label='internal projected-gradient open-loop H2 MPC baseline, not official paper implementation',
        official_gradient_implementation=False,official_MPPI=False,mppi_label='MPPI-inspired shooting, not official MPPI',
        delta_convention='gradient minus named reference; lower better except success; exact ties, no new thresholds',
        uncertainty_scope='sample SD/range over four already-used calibration roots; no CI, p-value or independent-population inference',
        budget_scope='80 candidate predicts =40 complete H2 programs +4 winner replay common; SRA adds4 compile predicts/root; gradient adds39 AD calls/root; NOT equal total budget',
        unmeasured_or_unmatched=deepcopy(planner.BOUNDARY['unmeasured'])+[
            'reference runtime attention backend','proposal distributions/support','AD kernel work versus Jacobian/SVD work'],
        composition_scope='No composition arm or matched composition ablation; cannot infer composition benefit or open P2',
        P1_admission=False,P2_admission=False,composition_admission=False,fair_total_budget_comparison=False,
        prediction_budget_matched=False,physical_efficacy_verified=False,new_neural_calls=0,new_physical_steps=0,
        future_heldout_results_opened=False)


def markdown(result):
    lines = ['# Gradient MPC / CEM / random / MPPI-inspired / SRA saved comparison','',
        f'Status: {result["status"]}; P2_admission=false; composition_admission=false.','']
    if not result['evidence_integrity_valid']:
        return '\n'.join(lines+result['integrity_errors'])+'\n'
    lines += ['Four already-used calibration roots, stable identity joins; saved arithmetic only.',
        'Internal gradient MPC, not an official paper implementation; MPPI-inspired, not official MPPI.',
        'Gradient MATH backend search/replay policy explicit; reference actual backends unknown, NOT established equal.',
        '40 complete H2 candidates/80 predicts +4 replay; AD and SRA compile overhead are additional work.','',
        '| Arm | Mean pose | Mean official state distance | Mean real encoder Q MSE | Official success | Predict total |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        means = result['arm_means'][arm]
        lines.append(f'| {arm} | {means["object_pose_cost"]:.6f} | {means["official_state_distance"]:.6f} | {means["real_encoder_q_mse"]:.8f} | {result["official_successes"][arm]}/4 | {result["cost_accounting"]["four_root_total_predict_samples"][arm]} |')
    lines += ['', 'Gradient minus reference: negative favors gradient on costs, not a method-selection rule.','',
        '| Reference | Metric | Mean delta | Sample SD | Gradient W/T/L |','| --- | --- | ---: | ---: | --- |']
    for arm in BASE_ARMS:
        for metric in ('object_pose_cost','official_state_distance','real_encoder_q_mse'):
            stats = result['paired_summaries'][arm][metric]
            lines.append(f'| {arm} | {metric} | {stats["mean"]:.8f} | {stats["sample_sd"]:.8f} | {stats["left_wins"]}/{stats["ties"]}/{stats["left_losses"]} |')
    lines += ['',result['uncertainty_scope'],result['composition_scope'],
        'Gradient:336 predicts +156 autograd.grad calls/156 action updates; model updates0.',
        'SRA:352 predicts +4 Jacobians/8 SVD/768 scalar-output VJP directions; not backward-kernel-equivalent to gradient.',
        'SRA read-only verification separately adds8 SVD/4 bounded solves per physical.inputs invocation.',
        'No equal total budget, shared backend, shared runtime cap or independent-population efficacy claim.']
    return '\n'.join(lines)+'\n'


def run(project,output):
    project,output = Path(project).resolve(),Path(output).resolve()
    require(TAG in output.name,'new dedicated tagged output required')
    output.mkdir(exist_ok=False)
    reader = evidence.EvidenceReader()
    own = [Path(__file__),project/'tests'/f'test_{TAG}.py',project/'innerdocs'/f'{TAG}.md',
        Path(previous.__file__),Path(previous.previous.__file__),Path(physical.__file__),Path(planner.__file__),
        Path(evidence.__file__),Path(previous.previous.labels.__code__.co_filename),
        Path(previous.previous.metric.__code__.co_filename)]
    try:
        for path in own: reader.bind(path)
        seal = create_seal(files=own,args_json=json.dumps(dict(scope='arithmetic-only five-arm known-root gradient comparison',
            P2_admission=False,composition_admission=False)),output_dir=output/'source_seal')
        bundle = load_inputs(project,reader)
        audit_artifacts(bundle,reader)
        result = compare_rows(build_rows(bundle))
        reader.unchanged()
        result['source_seal'] = seal
        result['source_verification'] = verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        result = dict(schema=TAG+'.report',status='invalid_evidence',evidence_integrity_valid=False,
            integrity_errors=[f'{type(exc).__name__}: {exc}'],P2_admission=False,composition_admission=False,
            fair_total_budget_comparison=False,new_neural_calls=0,new_physical_steps=0)
    result['input_and_source_sha256'] = reader.hashes
    result['hash_scope'] = 'audit-time observed bytes/current and archived sources; not independent proof of execution, attention dispatch, timing or prospective independence'
    evidence.write_new(output/'report.json',result)
    with (output/'report.md').open('x') as stream: stream.write(markdown(result))
    print(json.dumps(dict(status=result['status'],bound_files=len(reader.hashes),output=str(output),
        P2_admission=False,composition_admission=False)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    return 0 if run(args.project,args.output)['evidence_integrity_valid'] else 1


if __name__=='__main__': raise SystemExit(main())

