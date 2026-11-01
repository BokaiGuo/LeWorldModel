"""Model-free, fail-closed consumer of the sealed full20 calibration contract.

There is NO executable SRA arm in this version. Existing spectral compilation
is prefix/suffix composition, not a frozen joint-H2 spectral shooting protocol.
Prepare records that specific gap instead of inventing a renamed sampler.
Exit 2 means a verified scientific refusal, never experiment success.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from experiments.spectral_reachability import p2_composition_admission_20261027_v1 as audit
from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p2_no_composition_sra_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = PROJECT/'results/p2_full20_sampling_20261027_v2_cuda_v1'
DEFAULT_ADMISSION = PROJECT/'results/p2_composition_admission_20261027_v1_audit02/report.json'
SEEDS = (20261027, 20261028, 20261029, 20261030)
DEFINITION = dict(
    operator='J_H2 = d terminal(root, U) / d U; task response uses the frozen Q projection',
    parameterization='one joint 20-coordinate H2 action trajectory in right-singular action directions of Q@J_H2',
    progress='bounded damped goal-progress solve in that same joint spectral chart',
    no_composition='one complete H2 program; no bank-pair enumeration, prefix branching, suffix recompilation or intermediate reachable-value selection',
    witness='one edge containing all ten raw commands; two recurrence packets are not two composed programs',
    unresolved='raw versus pre-tanh derivatives, action scale, nominal, rank, signs/degeneracy, damping/radii and candidate slots require an independently frozen method contract')
GAPS = dict(
    joint_h2_spectral_compiler='Existing spectral_bank accepts 2 or 10 columns, not 20. compile_arm branches over prefixes and recomputes each suffix Jacobian; it is not a no-composition H2 compiler. chart_matrix(horizon=2) supplies coordinates only.',
    frozen_candidate_policy='No inherited contract specifies forty joint-H2 spectral candidates, nominal trajectories, rank/degenerate-subspace rule, progress and signed-mode slots, proposal support, or stable selection. Cannot invent these from calibration outcomes or copy CEM/random.',
    spectral_resource_contract='80+4 is the reference prediction-only budget. Joint-H2 derivative forwards, Jacobian invocations, VJP directions, SVDs, bounded solves and selection are additional work without a frozen allocation or shared full-resource cap.',
    composition_ablation_support='Existing composition searches a 16-command bank; full20 searches a raw box. Equal action bounds and H2 do not establish equal proposal support or isolate composition.',
    independent_scientific_evidence='Four roots are already-used calibration roots. Independent P1, prospective cohort, all-slot physical evidence and original composition-increment gates remain required.')
RESOURCES = ('planning_predict_calls', 'planning_predict_samples', 'replay_predict_calls',
    'replay_predict_samples', 'compilation_predict_calls', 'compilation_predict_samples',
    'jacobian_calls', 'scalar_output_vjp_directions', 'backward_calls', 'svd_calls',
    'bounded_solve_calls', 'encoder_calls', 'optimizer_steps', 'candidate_evaluations',
    'attempted_expansions', 'selection_evaluations', 'flops', 'upstream_encoding_and_bank_construction',
    'physical_env_steps', 'wall_time_limit_s')


def resource_contract():
    return dict(reference_prediction_only=dict(candidates=40, horizon_tokens=2,
        planning_predict_calls=80, planning_predict_samples=80,
        replay_predict_calls=4, replay_predict_samples=4),
        proposed_sra_limits={key: None for key in RESOURCES},
        unknown_is_not_zero=True, compilation_included_in_total=True,
        failed_attempts_charged=True, fair_total_budget_comparison=False)


def source_files():
    directory = PROJECT/'experiments/spectral_reachability'
    return [Path(__file__), PROJECT/'tests'/f'test_{TAG}.py', PROJECT/'innerdocs'/f'{TAG}.md',
        Path(audit.__file__), directory/'d2p_composition_witness_v2.py',
        directory/'d2p_prerun_source_seal.py', directory/'d2p_p2_comparison_contract.py',
        directory/'p2_full20_sampling_20261027_v2.py', directory/'p2_full20_mppi_20261027_v1.py',
        directory/'p2_matched_composition_20261027.py', directory/'p2_matched_sampling_20261027_v1.py',
        directory/'pusht_full_packet_compiler_20261027.py', directory/'pusht_full_packet_chart_20261027.py',
        PROJECT.parent/'docs/D2P_P2_COMPARISON_CONTRACT_20261026.md']


def reference_contract(reference, admission, reader):
    """Explicit known inputs only; no model imports or physical outcome joins."""
    reference = Path(reference).resolve()
    protocol = reader.load(reference/'protocol.json')
    prepared = reader.load(reference/'prepared.json')
    report = reader.load(reference/'report.json')
    audit.require(protocol['schema'] == 'p2_full20_sampling_20261027_v2.protocol', 'full20 schema required')
    audit.require(digest(protocol) == prepared['protocol_sha256'], 'full20 protocol digest mismatch')
    audit.require(report['source_seals'] == prepared['source_seals'], 'full20 report seal mismatch')
    reader.seals(prepared['source_seals'])
    reader.bindings(protocol['bindings'])
    compilation = reader.load(protocol['compilation'], audit.COMPILATION_SHA)
    # Saved candidate/witness validation only. Never run a new selection/model.
    audit.validate_planning(report, protocol, compilation, sampling=True)
    schedule = protocol['schedule']
    for key, value in dict(horizon_tokens=2, primitive_steps=10, candidates=40,
            planning_predict_calls=80, planning_predict_samples=80, replay_predict_calls=4,
            replay_predict_samples=4, batch_size=1, proposal_dimensions=20, packet_shape=[2,5,2]).items():
        audit.require(schedule[key] == value, 'reference budget/domain changed: '+key)
    audit.require(protocol['objective'] == 'mean square Q@(terminal-goal_vectors[0])', 'objective changed')
    audit.require(protocol['arms'] == ['cem','random'], 'reference arms changed')
    ready = deepcopy(protocol['prepared'])
    audit.require(len(ready) == 4 and {r['root_index'] for r in ready} == set(range(4)), 'four indexed roots required')
    source = {r['root_sha256']: r for r in compilation['rows']}
    cases = []
    for row in sorted(ready, key=lambda r:r['root_index']):
        domain = row['action_domain']
        audit.require(type(row['root_index']) is int and row['seed'] == SEEDS[row['root_index']], 'root seed changed')
        audit.require(domain['name'] == 'raw' and domain['action_block'] == 5 and
            domain['low'] == [-1.,-1.] and domain['high'] == [1.,1.] and
            domain['normalization_sha256'] == digest(protocol['normalization']), 'raw normalization/domain changed')
        compiled = source[row['root_sha256']]['compiled']
        audit.require(row['metric_sha256'] == compiled['scoring']['metric']['sha256'], 'metric identity changed')
        cases.append(dict(row, goal_vector_sha256=digest(compiled['goal_vectors'][0]),
            projection_sha256=digest(compiled['scoring']['metric']['projection'])))
    decision = reader.load(admission)
    audit.require(decision['schema'] == audit.TAG+'.report' and decision['evidence_integrity_valid'] is True
        and decision['P2_admission'] is False, 'historical rejection required')
    audit.require(decision['executable_prerequisites'] == audit.PREREQUISITES, 'admission prerequisites changed')
    return dict(reference=str(reference), admission=str(Path(admission).resolve()), cases=cases,
        normalization=protocol['normalization'], objective=protocol['objective'],
        compilation=protocol['compilation'], policy=protocol['policy'],
        original_source_seals=prepared['source_seals'])


def refusal():
    return dict(schema=TAG+'.report', status='rejected_missing_native_sra_contract',
        executable_baseline=False, selected_actions=None, candidate_receipts=[],
        definition=deepcopy(DEFINITION), gaps=deepcopy(GAPS), resources=resource_contract(),
        new_work={key:0 for key in RESOURCES if key != 'wall_time_limit_s'},
        new_work_scope='no planner/model/physics work; file hashing, JSON validation and prepare wall time are not zero-cost claims',
        new_neural_calls=0, new_physical_steps=0, P1_admission=False, P2_admission=False,
        prediction_budget_matched=False, fair_total_budget_comparison=False,
        physical_efficacy_verified=False, future_heldout_results_opened=False,
        known_calibration_roots=4, independent_outcome_roots=0)


def prepare(reference, output, admission=DEFAULT_ADMISSION):
    output = Path(output).absolute()
    audit.require(TAG in output.name, 'dedicated tagged output required')
    output.mkdir(exist_ok=False)
    reader = audit.EvidenceReader()
    try:
        context = reference_contract(reference, admission, reader)
        files = source_files()
        for path in files:
            reader.bind(path)
        protocol = dict(schema=TAG+'.protocol', context=context, definition=deepcopy(DEFINITION),
            gaps=deepcopy(GAPS), resources=resource_contract(), bindings=deepcopy(reader.hashes),
            calibration_only=True, executable_baseline=False, P2_admission=False)
        audit.write_new(output/'protocol.json', protocol)
        seal = create_seal(files=files, args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
                           output_dir=output/'source_seal_0')
        result = dict(refusal(), evidence_integrity_valid=True, protocol_sha256=digest(protocol))
        audit.write_new(output/'report.json', result)
        audit.write_new(output/'prepared.json', dict(schema=TAG+'.prepared',
            status='prepared_rejection_only', protocol_sha256=digest(protocol),
            report_sha256=audit.sha(output/'report.json'), source_seals=[seal],
            new_neural_calls=0, executable_baseline=False, P2_admission=False))
        reader.unchanged()
        load_prepared(output)
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        # Never overwrite a partially written artifact or bless a partial seal.
        result = dict(refusal(), status='invalid_evidence', evidence_integrity_valid=False,
            integrity_errors=[f'{type(exc).__name__}: {exc}'], observed_bindings=reader.hashes)
        audit.write_new(output/'invalid.json', result)
    return result


def load_prepared(output):
    output = Path(output).absolute()
    audit.require(not (output/'invalid.json').exists(), 'invalid/incomplete prepare')
    protocol = audit.read_json(output/'protocol.json')
    prepared = audit.read_json(output/'prepared.json')
    result = audit.read_json(output/'report.json')
    audit.require(protocol['schema'] == TAG+'.protocol' and prepared['schema'] == TAG+'.prepared'
        and digest(protocol) == prepared['protocol_sha256'] == result['protocol_sha256'], 'protocol binding changed')
    audit.require(audit.sha(output/'report.json') == prepared['report_sha256'], 'report binding changed')
    expected = dict(refusal(), evidence_integrity_valid=True, protocol_sha256=digest(protocol))
    audit.require(result == expected, 'refusal report changed')
    audit.require(protocol['definition'] == DEFINITION and protocol['gaps'] == GAPS
        and protocol['resources'] == resource_contract() and protocol['executable_baseline'] is False
        and protocol['P2_admission'] is False and protocol['calibration_only'] is True, 'frozen refusal changed')
    audit.require(prepared['status'] == 'prepared_rejection_only' and prepared['new_neural_calls'] == 0
        and prepared['executable_baseline'] is False and prepared['P2_admission'] is False
        and len(prepared['source_seals']) == 1, 'prepared refusal changed')
    for seal in prepared['source_seals']:
        verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
        args = audit.read_json(Path(seal['output_dir'])/'args.json')
        audit.require(args == dict(protocol_sha256=digest(protocol)), 'source/protocol binding changed')
    reader = audit.EvidenceReader()
    reader.bindings(protocol['bindings'])
    context = protocol['context']
    actual = reference_contract(context['reference'], context['admission'], reader)
    audit.require(actual == context, 'reference identities changed')
    reader.unchanged()
    return protocol, prepared, result


def selected_actions(*args, **kwargs):
    raise ValueError('rejected_missing_native_sra_contract: no executable SRA or deployable actions')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE)
    prep.add_argument('--admission', type=Path, default=DEFAULT_ADMISSION)
    prep.add_argument('--output', type=Path, required=True)
    for name in ('verify', 'cuda'):
        sub.add_parser(name).add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args(argv))
    phase = args.pop('phase')
    try:
        if phase == 'prepare':
            result = prepare(**args)
        else:
            _, _, result = load_prepared(args['output'])
            if phase == 'cuda':
                result = dict(result, execution_refused_before_allocation=True)
        print(json.dumps(dict(status=result['status'], output=str(args['output']),
            executable_baseline=False, P2_admission=False, new_neural_calls=0,
            execution_refused_before_allocation=phase == 'cuda')))
        return 2 if result['evidence_integrity_valid'] else 1
    except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps(dict(status='invalid_evidence', error=f'{type(exc).__name__}: {exc}', P2_admission=False)))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
