"""Read-only root-paired MPPI-inspired/CEM/random calibration comparison.

Only saved bytes, NumPy arithmetic and receipt reconstruction. No model/physics
execution, outcome-based selection, independent-population inference or P2 pass.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics

import numpy as np

from experiments.spectral_reachability import p2_composition_admission_20261027_v1 as evidence
from experiments.spectral_reachability import p2_full20_mppi_physical_20261027_v1 as mppi
from experiments.spectral_reachability import p2_full20_physical_20261027_v2 as baseline
from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
from experiments.spectral_reachability.p2_objective_decomposition_20261027 import metric
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p2_mppi_matched_comparison_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
MPPI_RUN = 'p2_full20_mppi_physical_20261027_v1_cuda_main_v1'
BASE_RUN = 'p2_full20_physical_20261027_v2_cuda_v1'
ARMS = (mppi.sampling.ARM, 'cem', 'random')
METRICS = ('object_pose_cost', 'object_xy_distance', 'object_angle_distance',
    'official_state_distance', 'official_position_error', 'agent_position_error',
    'object_position_error', 'official_angle_error', 'official_success', 'object_only_success',
    'real_encoder_q_mse', 'predicted_q_mse')
STATE_COORDINATES = ('agent_x','agent_y','object_x','object_y','raw_object_angle','agent_vx','agent_vy')
require = evidence.require


def index(rows, keys):
    return evidence.index(rows, keys)


def array(value, shape):
    raw = np.asarray(value)
    require(raw.dtype.kind in 'fiu' and raw.shape == shape and np.isfinite(raw).all(),
            'finite numeric state/vector with exact shape required')
    return raw.astype(float)


def close(actual, expected, name):
    # Original saved evaluator/metric verification tolerances, never admission gates.
    require(type(actual) in (int,float) and np.isfinite(actual)
            and np.isclose(actual,expected,rtol=0,atol=1e-10), 'saved '+name+' mismatch')


def validate_budget(arm, root_id, protocol, prepared, task_id):
    evidence.entries_match(arm)
    require(len(arm['attempts']) == 40 and arm['planning_predict_samples'] == 80
            and arm['replay_predict_samples'] == 4 and arm['fixture_only'] is False,
            '40 full H2 candidates / actual80+4 required')
    require(all(e['context_tokens'] == 3 for e in arm['actual_model_entries']), 'processed context mismatch')
    budget = arm['actual_prediction_budget']
    require(budget['binding'] == dict(protocol_id=prepared['source_seals'][0]['manifest_sha256'],
        task_id=task_id,root_id=root_id,frozen_model_id=evidence.CHECKPOINT_SHA)
        and budget['method'] == arm['arm'] and budget['predict_sample_limit'] == 84
        and budget['attempted_predict_samples'] == 84 and budget['blocked_calls'] == 0,
        'planning budget identity or limit mismatch')
    require([(p['phase'],p['predict_calls'],p['predict_samples']) for p in budget['phases']] ==
            [('planning',80,80),('evaluation',4,4)]
        and all(p['completed'] is True and p['failed_predict_calls'] == 0 for p in budget['phases']),
        'planning budget phase mismatch')
    require(arm['action_encoder_work'] == dict(calls=84,samples=84,processed_context_tokens=252),
            'action encoder work mismatch')
    schedule = protocol['schedule']
    require(all(schedule[k] == v for k,v in dict(candidates=40,horizon_tokens=2,primitive_steps=10,
        planning_predict_samples=80,replay_predict_samples=4,proposal_dimensions=20,packet_shape=[2,5,2]).items()),
        'shared full20 budget/domain schedule mismatch')


def load_inputs(project, reader):
    """Explicit calibration inputs only, never discover future outcome files."""
    results = Path(project)/'results'
    mp_dir, base_dir = results/MPPI_RUN, results/BASE_RUN
    mp_report, mp_run_protocol = (reader.load(mp_dir/name) for name in ('report.json','protocol.json'))
    mp_prepared_dir = Path(mp_run_protocol['prepared'])
    mp_protocol, mp_prepared = (reader.load(mp_prepared_dir/name) for name in ('protocol.json','prepared.json'))
    reader.bindings(mp_protocol['bindings'])
    reader.seals(mp_prepared['source_seals'])
    _,verified_protocol,verified_prepared,context = mppi.load_prepared(mp_prepared_dir)
    require(verified_protocol == mp_protocol and verified_prepared == mp_prepared, 'MPPI physical prepare changed')
    require(mp_run_protocol['prepared_protocol_sha256'] == mp_report['prepared_protocol_sha256'] ==
        mp_prepared['protocol_sha256'] == mppi.digest(mp_protocol), 'MPPI run/prepare digest mismatch')
    require(mp_report['source_seals'] == mp_prepared['source_seals'] and mp_run_protocol['plans'] == mp_protocol['plans']
        and mp_run_protocol['encode_terminals'] is True, 'MPPI physical source/plan binding mismatch')
    mp_planning_dir = Path(mp_protocol['planning'])
    mp_planning, mp_planning_protocol, mp_planning_prepared = (reader.load(mp_planning_dir/name)
        for name in ('report.json','protocol.json','prepared.json'))
    reader.bindings(mp_planning_protocol['bindings'])
    reader.seals(mp_planning_prepared['source_seals'])
    compilation = reader.load(mp_planning_protocol['compilation'],evidence.COMPILATION_SHA)
    mp_plans = mppi.plans_from_saved(mp_planning,mp_planning_protocol,mp_planning_prepared,compilation)
    require(mp_plans == mp_protocol['plans'], 'MPPI plans differ from saved CUDA witness')

    base_report, base_protocol = (reader.load(base_dir/name) for name in ('report.json','protocol.json'))
    reader.bindings(base_protocol['bindings'])
    reader.seals(base_report['source_seals'])
    planning_dir = Path(base_protocol['planning'])
    base_planning, base_planning_protocol, base_prepared = (reader.load(planning_dir/name)
        for name in ('report.json','protocol.json','prepared.json'))
    reader.bindings(base_planning_protocol['bindings'])
    reader.seals(base_prepared['source_seals'])
    require(base_planning['source_seals'] == base_prepared['source_seals'] and
        mppi.digest(base_planning_protocol) == base_prepared['protocol_sha256'], 'CEM/random prepared source/protocol mismatch')
    _,verified,previous,base_source = baseline.sampling.load_prepared(planning_dir)
    require(verified == base_planning_protocol and previous == base_prepared
        and base_source == compilation, 'CEM/random compilation identity mismatch')
    base_plans = baseline.plans_from_saved(base_planning,base_planning_protocol,compilation)
    require(index(base_plans,('root_sha256','method')) == index(base_protocol['plans'],('root_sha256','method')),
            'CEM/random physical plans differ from saved CUDA selections')
    require(mp_planning_protocol['normalization'] == base_planning_protocol['normalization']
            and mp_planning_protocol['objective'] == base_planning_protocol['objective'], 'normalization or goal metric mismatch')
    return dict(compilation=compilation, groups=[dict(directory=mp_dir,report=mp_report,
        protocol=mp_protocol,plans=mp_plans,planning=mp_planning,planning_protocol=mp_planning_protocol,
        planning_prepared=mp_planning_prepared,task_id=mppi.sampling.TAG),
        dict(directory=base_dir,report=base_report,protocol=base_protocol,plans=base_plans,
        planning=base_planning,planning_protocol=base_planning_protocol,planning_prepared=base_prepared,
        task_id=baseline.sampling.TAG)])


def validate_saved_branch(branch, plan, encoding, compiled, predicted_cost):
    """Validate exact H2, recompute evaluator arithmetic; no installed env call."""
    require(mppi.digest({k:v for k,v in plan.items() if k != 'selection_sha256'}) == plan['selection_sha256']
        and branch['selection_sha256'] == plan['selection_sha256'], 'selected action binding mismatch')
    identity, root = branch['context_identity'], plan['root']
    require(identity['history_rgb_sha256'] == root['root_rgb_sha256']
        and identity['goal_rgb_sha256'] == root['goal_rgb_sha256'], 'physical root/goal RGB mismatch')
    require(branch['status'] == 'completed' and branch['full_horizon_completed'] is True
        and branch['failure'] is None and not branch['failures'] and branch['repetition'] == 0
        and branch['planned_commands'] == branch['completed_commands'] == branch['attempted_commands'] == 10
        and branch['physical_env_step_attempts'] == branch['physical_env_step_returns'] == 10
        and branch['remaining_commands_not_executed'] == 0 and branch['planned_horizon_tokens'] == 2,
        'incomplete fixed-H2 branch; cannot silently drop/substitute endpoint')
    trace = branch['trace']
    require(len(trace) == 10 and [e['primitive_step'] for e in trace] == list(range(1,11))
        and [e['command'] for e in trace] == plan['commands'] and trace[-1]['state'] == branch['final_state'],
        'exact action/trace/final-state mismatch')
    require(branch['guard']['root_id'] == plan['root_sha256']
        and branch['guard']['primitive_commands_authorized'] == 10, 'guard identity/accounting mismatch')
    require(encoding['status'] == 'completed' and encoding['failure'] is None
        and encoding['attempted_encode_samples'] == 1 and encoding['repetition'] == 0
        and (encoding['root_sha256'],encoding['method']) == (plan['root_sha256'],plan['method']),
        'root-bound complete terminal encoding required')
    state, goal = array(branch['final_state'],(7,)), array(identity['goal'],(7,))
    official = labels(state,goal)
    for key,value in official.items():
        if type(value) is bool:
            require(type(branch['official'][key]) is bool and branch['official'][key] == value,
                    'saved official success mismatch')
        else:
            close(branch['official'][key],value,key)
    xy = float(np.linalg.norm(state[2:4]-goal[2:4]))
    angle = float(abs(np.arctan2(np.sin(state[4]-goal[4]),np.cos(state[4]-goal[4]))))
    for key,value in dict(xy_distance=xy,angle_distance=angle,pose_cost=xy+20*angle).items():
        close(branch['pose'][key],value,key)
    vector = array(encoding['terminal_vector'],(192,))
    return dict(root_sha256=plan['root_sha256'],arm=plan['method'],
        root_binding_sha256=mppi.digest(root),action_domain=deepcopy(plan['action_domain']),
        metric_sha256=compiled['scoring']['metric']['sha256'],goal_vectors_sha256=mppi.digest(compiled['goal_vectors']),
        physical_identity=deepcopy(identity),final_state=state.tolist(),state_absolute_error=abs(state-goal).tolist(),
        object_pose_cost=xy+20*angle,object_xy_distance=xy,object_angle_distance=angle,
        official_state_distance=official['official_state_distance'],official_position_error=official['official_position_error'],
        agent_position_error=official['agent_position_error'],object_position_error=official['object_position_error'],
        official_angle_error=official['angle_error'],official_success=int(official['official_success']),
        object_only_success=int(official['object_only_success']),
        real_encoder_q_mse=metric(vector,compiled),predicted_q_mse=evidence.finite(predicted_cost),
        budget=dict(full_H2_candidates=40,planning_predict_samples=80,replay_predict_samples=4,
            action_encoder_calls=84,processed_context_tokens=252,physical_env_steps=10,terminal_encode_samples=1))


def audit_artifacts(group,reader):
    directory, report = group['directory'], group['report']
    require(report['status'] == 'calibration_recorded' and bool(report['gates'])
        and all(v is True for v in report['gates'].values()) and not report['failures']
        and report['new_predict_calls'] == 0, 'physical report integrity gates failed')
    physical = reader.load(directory/'physical_report.json')
    require(index(physical['rows'],('root_sha256','method')) == index(report['rows'],('root_sha256','method'))
        and not physical['failures'], 'durable physical report differs from final report')
    for branch in report['rows']:
        stem = f'root_{branch["root_index"]}_{branch["method"]}_replay0'
        require(reader.load(directory/(stem+'.physical.json')) == branch, 'persisted branch differs')
        reader.bind(branch['journal_path'],branch['journal_sha256'])
        events = [json.loads(line,object_pairs_hook=evidence.strict_object)
                  for line in Path(branch['journal_path']).read_text().splitlines()]
        require(len(events) == 31 and events[0]['kind'] == 'initial'
            and events[0]['state'] == branch['context_identity']['state'], 'durable initial/step census mismatch')
        for i,observation in enumerate(branch['trace']):
            attempt,returned,observed = events[1+i*3:4+i*3]
            require(attempt == dict(kind='attempt',primitive_step=i+1,command=observation['command'])
                and returned == dict(kind='step_returned',primitive_step=i+1)
                and {k:v for k,v in observed.items() if k not in ('rgb_path','rgb_file_sha256')} == observation,
                'durable attempt/return/observation mismatch')
        for event in events:
            if 'rgb_path' in event: reader.bind(event['rgb_path'],event['rgb_file_sha256'])
        require(events[-1]['rgb_path'] == branch['terminal_rgb_path']
            and events[-1]['rgb_file_sha256'] == branch['terminal_rgb_sha256'], 'terminal RGB linkage mismatch')
        reader.bind(branch['terminal_rgb_path'],branch['terminal_rgb_sha256'])
    encodings = index(report['encodings'],('root_sha256','method'))
    for branch in report['rows']:
        path = directory/f'root_{branch["root_index"]}_{branch["method"]}_replay0.encoding.json'
        require(reader.load(path) == encodings[branch['root_sha256'],branch['method']], 'persisted terminal encoding differs')


def build_rows(bundle):
    compiled = index(bundle['compilation']['rows'],('root_sha256',))
    rows = []
    for group in bundle['groups']:
        report = group['report']
        plans = index(group['plans'],('root_sha256','method'))
        branches = index(report['rows'],('root_sha256','method'))
        encodings = index(report['encodings'],('root_sha256','method'))
        require(plans.keys() == branches.keys() == encodings.keys(), 'physical plan/encoding census mismatch')
        planned = index(group['planning']['rows'],('root_sha256',))
        require(group['planning']['actual_predict_samples'] == len(plans)*84
            and group['planning'].get('actual_predict_calls',len(plans)*84) == len(plans)*84,
            'planning aggregate predict budget mismatch')
        for key,plan in plans.items():
            root,method = key
            arm = index(planned[root,]['rows'],('arm',))[method,]
            validate_budget(arm,root,group['planning_protocol'],group['planning_prepared'],group['task_id'])
            require(plan['root'] == arm['witness']['receipt']['root']
                and plan['commands'] == arm['witness']['receipt']['actions']
                and plan['receipt_sha256'] == arm['witness']['receipt']['receipt_sha256'], 'plan/witness identity mismatch')
            require(abs(metric(arm['witness']['receipt']['terminal_context']['latent_history'][-1],
                compiled[root,]['compiled'])-arm['selected_model_cost']) <= 1e-12, 'predicted goal0/Q score mismatch')
            rows.append(validate_saved_branch(branches[key],plan,encodings[key],compiled[root,]['compiled'],arm['selected_model_cost']))
        require(report['selected_path_env_step_attempts'] == report['selected_path_env_step_returns'] == len(plans)*10
            and report['attempted_terminal_encode_samples'] == len(plans), 'physical total budget mismatch')
    return rows


def describe(values, *, higher_better=False):
    values = [evidence.finite(v) for v in values]
    require(len(values) == 4, 'all four paired roots required; no filtering')
    return dict(n=4,mean=statistics.mean(values),median=statistics.median(values),
        sample_sd=statistics.stdev(values),minimum=min(values),maximum=max(values),
        left_wins=sum(v > 0 if higher_better else v < 0 for v in values),
        ties=sum(v == 0 for v in values),left_losses=sum(v < 0 if higher_better else v > 0 for v in values))


def compare_rows(rows):
    indexed = index(rows,('root_sha256','arm'))
    roots = sorted({r['root_sha256'] for r in rows})
    require(len(roots) == 4 and set(indexed) == {(r,a) for r in roots for a in ARMS}, 'four roots / three arms required')
    paired = []
    for root in roots:
        arms = {arm:indexed[root,arm] for arm in ARMS}
        for field in ('root_binding_sha256','action_domain','metric_sha256','goal_vectors_sha256','physical_identity','budget'):
            require(all(arms[arm][field] == arms[ARMS[0]][field] for arm in ARMS), 'cross-arm '+field+' mismatch')
        paired.append(dict(root_sha256=root,arms=arms,
            mppi_minus={arm:{metric:arms[ARMS[0]][metric]-arms[arm][metric] for metric in METRICS}
                for arm in ('cem','random')},
            mppi_minus_state_absolute_error={arm:(np.asarray(arms[ARMS[0]]['state_absolute_error'])-
                np.asarray(arms[arm]['state_absolute_error'])).tolist() for arm in ('cem','random')}))
    summaries = {arm:{metric:describe([r['mppi_minus'][arm][metric] for r in paired],
        higher_better=metric in ('official_success','object_only_success')) for metric in METRICS} for arm in ('cem','random')}
    return dict(schema=TAG+'.report',status='saved_calibration_comparison_completed',evidence_integrity_valid=True,
        paired_roots=4,branches=12,rows=paired,paired_summaries=summaries,
        paired_state_error_summaries={arm:{coordinate:describe([
            r['mppi_minus_state_absolute_error'][arm][i] for r in paired])
            for i,coordinate in enumerate(STATE_COORDINATES)} for arm in ('cem','random')},
        state_absolute_error_coordinates=list(STATE_COORDINATES),
        official_successes={arm:sum(indexed[root,arm]['official_success'] for root in roots) for arm in ARMS},
        delta_convention='MPPI-inspired minus baseline; lower better except success indicators; exact ties, no thresholds',
        uncertainty_scope='sample SD/range across four already-used calibration roots; no CI, p-value or independent-population inference',
        budget_scope='matched recorded candidate/predict/action-encoder/selected-path/terminal-encode units only; 80 planning predicts = 40 full H2 candidates',
        unmeasured_or_unmatched=['FLOPs','upstream encoding/bank construction','common wall-time cap','proposal distributions'],
        claim_boundary=mppi.sampling.CLAIM,official_MPPI=False,fair_total_budget_comparison=False,
        P1_admission=False,P2_admission=False,physical_efficacy_verified=False,
        new_neural_calls=0,new_physical_steps=0,future_heldout_results_opened=False)


def markdown(result):
    lines = ['# Saved MPPI-inspired / CEM / random paired calibration', '',
        f'Status: {result["status"]}; P2_admission=false.', '']
    if not result['evidence_integrity_valid']:
        return '\n'.join(lines+result['integrity_errors'])+'\n'
    lines += ['Four known calibration roots; no new model or physics calls.',
        '80 planning predict samples = 40 complete H2 candidates, plus 4 replay.',
        'MPPI-inspired heuristic, not official MPPI; total compute not matched.', '',
        '| Root | MPPI pose | CEM pose | random pose | MPPI state distance | CEM state distance | random state distance |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['rows']:
        arms = row['arms']
        values = [arms[a][k] for k in ('object_pose_cost','official_state_distance') for a in ARMS]
        lines.append('| '+row['root_sha256'][:8]+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    lines += ['', 'MPPI-inspired minus baseline (negative favors MPPI for these costs):', '',
        '| Baseline | Metric | Mean delta | Sample SD | W/T/L |', '| --- | --- | ---: | ---: | --- |']
    for arm in ('cem','random'):
        for metric in ('object_pose_cost','official_state_distance','real_encoder_q_mse'):
            s = result['paired_summaries'][arm][metric]
            lines.append(f'| {arm} | {metric} | {s["mean"]:.6f} | {s["sample_sd"]:.6f} | {s["left_wins"]}/{s["ties"]}/{s["left_losses"]} |')
    lines += ['', result['uncertainty_scope'], 'No method selection, scientific stage admission or causal composition claim.']
    return '\n'.join(lines)+'\n'


def run(project,output):
    project,output = Path(project).resolve(),Path(output).resolve()
    output.mkdir(exist_ok=False)
    reader = evidence.EvidenceReader()
    # Explicit small audit source snapshot; historical transitive seals are
    # separately verified and recorded in the input manifest.
    own = [Path(__file__),project/'tests'/f'test_{TAG}.py',project/'docs'/f'{TAG}.md',
        Path(evidence.__file__),Path(mppi.__file__),Path(baseline.__file__),
        Path(labels.__code__.co_filename),Path(metric.__code__.co_filename)]
    try:
        for path in own: reader.bind(path)
        seal = create_seal(files=own,args_json=json.dumps(dict(scope='read-only known-root paired audit')),output_dir=output/'source_seal')
        bundle = load_inputs(project,reader)
        for group in bundle['groups']: audit_artifacts(group,reader)
        result = compare_rows(build_rows(bundle))
        reader.unchanged()
        result['source_seal'] = seal
        result['source_verification'] = verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        result = dict(schema=TAG+'.report',status='invalid_evidence',evidence_integrity_valid=False,
            integrity_errors=[f'{type(exc).__name__}: {exc}'],P2_admission=False,
            new_neural_calls=0,new_physical_steps=0)
    result['input_and_source_sha256'] = reader.hashes
    result['hash_scope'] = 'audit-time observed files and historical source seals; hashes do not prove actual execution or pre-outcome independence'
    evidence.write_new(output/'report.json',result)
    with (output/'report.md').open('x') as stream: stream.write(markdown(result))
    print(json.dumps(dict(status=result['status'],P2_admission=False,bound_files=len(reader.hashes),output=str(output))))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    return 0 if run(args.project,args.output)['evidence_integrity_valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
