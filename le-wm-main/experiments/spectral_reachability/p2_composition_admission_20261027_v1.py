"""Read-only, fail-closed P2 evidence audit of the existing calibration cohort.

Stdlib plus the pure action-witness validator. Never imports a model, discovers
held-out outcomes, chooses a method, executes physics, or changes a threshold.
This version supports historical calibration schemas only: no admission path
exists until independently audited prospective P1/P2 evidence is supported.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from experiments.spectral_reachability import d2p_composition_witness_v2 as witness

TAG = 'p2_composition_admission_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
COMPILATION = 'pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json'
COMPILATION_SHA = 'd5f544689a9d2b5c6807b2454c1a98b9db6d88bb374badc457bdcc6defa87c8a'
CHECKPOINT_SHA = '48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607'
FILES = {
    'composition': 'pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json',
    'composition_protocol': 'pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_protocol.json',
    'composition_sources': 'pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_sources.json',
    'composition_physical': 'pusht_p2_matched_physical4_v1_20261027_cuda/report.json',
    'sampling': 'p2_full20_sampling_20261027_v2_cuda_v1/report.json',
    'sampling_protocol': 'p2_full20_sampling_20261027_v2_cuda_v1/protocol.json',
    'sampling_prepared': 'p2_full20_sampling_20261027_v2_cuda_v1/prepared.json',
    'sampling_physical': 'p2_full20_physical_20261027_v2_cuda_v1/report.json',
    'physical_protocol': 'p2_full20_physical_20261027_v2_cuda_v1/protocol.json',
}
# Necessary evidence, not new acceptance thresholds. No self-declared boolean
# or successful sampling/physical run can satisfy these scientific gates.
PREREQUISITES = {
    'independent_p1': 'Supply the independently audited original P1 calibration/outcome evidence, all original implementation gates and strictly favorable predeclared paired intervals; do not reopen or tune on held-out outcomes.',
    'prospective_cohort': 'Externally seal new root/goal/context identities, method, seeds, candidate slots, metric, stopping rule and analysis before outcome access; retain every failed or aborted branch. Current four roots are already used calibration roots.',
    'strong_no_composition_sra': 'Implement and freeze a faithful no-composition spectral SRA control; flat ordered-pair bank search is not this control. Bind code/config and every candidate receipt.',
    'common_action_search_domain': 'Freeze a composition ablation and strong baselines under the same raw action domain and declared proposal/support policy; current 16-command bank versus full20 box confounds composition with search support.',
    'complete_matched_budget': 'Record and enforce the existing strict contract budget for every arm/root: planning+replay, all candidates/failed attempts, encoding, backward/optimizer, FLOPs, expansions, physical steps and common wall-time cap, including upstream construction. Unknown costs remain unknown, never zero.',
    'all_candidate_physical_evidence': 'Provide exact action/receipt/metric-bound physical outcomes for all frozen paired candidate slots as required by the strict P2 contract, not just eight selected paths; retain guard endpoints without substituting them for H2.',
    'independent_composition_increment': 'On the externally sealed independent cohort, supply non-greedy composition mechanism witnesses and predeclared paired uncertainty/multiplicity analysis against no-composition SRA and strong baselines using the original scientific gates. Predicted Q or sampling completion is not physical benefit.',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def read_json(path):
    def reject(value):
        raise ValueError('nonfinite JSON constant: '+value)
    value = json.loads(Path(path).read_text(), object_pairs_hook=strict_object,
                       parse_constant=reject)
    witness._canonical(value)  # Also rejects overflow such as 1e999.
    return value


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


class EvidenceReader:
    """Explicit references only; no recursive result discovery or code execution."""
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).absolute()
        require(path.is_file() and not path.is_symlink(), 'regular nonsymlink evidence required: '+str(path))
        value = sha(path)
        if expected is not None:
            witness._sha(expected)
            require(value == expected, 'source/input SHA mismatch: '+str(path))
        require(str(path) not in self.hashes or self.hashes[str(path)] == value,
                'evidence changed while reading: '+str(path))
        self.hashes[str(path)] = value
        return path

    def load(self, path, expected=None):
        return read_json(self.bind(path, expected))

    def bindings(self, values):
        require(isinstance(values, dict) and bool(values), 'nonempty input bindings required')
        for path, expected in values.items():
            require(Path(path).is_absolute(), 'absolute bound input required')
            self.bind(path, expected)

    def seals(self, seals):
        require(isinstance(seals, list) and bool(seals), 'nonempty source seals required')
        for seal in seals:
            directory = Path(seal['output_dir'])
            manifest = self.load(directory/'manifest.json', seal['manifest_sha256'])
            require(manifest['schema'] == 'd2p_prerun_source_seal.v1', 'source schema mismatch')
            self.bind(directory/'args.json', manifest['args_sha256'])
            entries = manifest['files']
            require(bool(entries) and len(entries) == seal['source_count'], 'source census mismatch')
            require(len({e['source'] for e in entries}) == len(entries), 'duplicate source entry')
            for i, entry in enumerate(entries):
                require(entry['archive'] == f'sources/{i:03d}.source', 'source archive path mismatch')
                require(Path(entry['source']).is_absolute(), 'absolute source required')
                for path in (Path(entry['source']), directory/entry['archive']):
                    self.bind(path, entry['sha256'])
                    require(path.stat().st_size == entry['size'], 'source size mismatch')

    def unchanged(self):
        for path, expected in list(self.hashes.items()):
            self.bind(path, expected)


def index(rows, keys):
    require(isinstance(rows, list) and bool(rows), 'nonempty identity census required')
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(all(isinstance(k, str) and k for k in key), 'explicit stable identities required')
        require(key not in result, 'duplicate stable identity')
        result[key] = row
    return result


def finite(value):
    require(type(value) in (float, int) and math.isfinite(value), 'finite numeric score required')
    return value


def entries_match(arm):
    ledger = arm['ledger']
    require(len(ledger) == 84, 'complete 80+4 packet ledger required')
    require([e['phase'] for e in ledger] == ['planning']*80+['replay']*4, 'phase accounting mismatch')
    expected = []
    for event in ledger:
        require(event['completed'] is True and all(type(event[k]) is int and event[k] == 1
                for k in ('predict_calls', 'predict_samples', 'action_encoder_calls')), 'failed/nonunit packet retained: incomplete calibration')
        for kind in ('action_encoder', 'predict'):
            expected.append((kind, 1, True, arm['arm'], event['phase'], event['path']))
    actual = []
    for event in arm['actual_model_entries']:
        scope = event['scope']
        require(type(event['samples']) is int and event['completed'] is True, 'invalid actual entry')
        actual.append((event['kind'], event['samples'], event['completed'], scope['arm'], scope['phase'], scope['path']))
    require(actual == expected, 'actual entries differ from dispatch ledger')


def validate_planning(report, protocol, compilation, *, sampling):
    require(report['protocol_sha256'] == witness.digest(protocol), 'planning protocol digest mismatch')
    require(report['status'] == 'calibration_completed' and bool(report['gates'])
            and all(v is True for v in report['gates'].values()), 'planning integrity gates failed')
    roots = index(compilation['rows'], ('root_sha256',))
    rows = index(report['rows'], ('root_sha256',))
    ready = index(protocol['prepared'], ('root_sha256',)) if sampling else index(
        [dict(r, root_sha256=r['root']['root_id']) for r in protocol['prepared']], ('root_sha256',))
    require(len(roots) == 4 and roots.keys() == rows.keys(), 'all four stable roots required')
    require(ready.keys() == roots.keys(), 'prepared root census differs')
    selected = {}
    for key, row in rows.items():
        compiled = roots[key]['compiled']
        root = row['root'] if sampling else row['declaration']['root']
        require(root == ready[key]['root'], 'prepared root mismatch')
        if not sampling:
            require(witness.digest(row['declaration']) == row['declaration_sha256'], 'declaration digest mismatch')
            require(row['fixture_only'] is False and row['prediction_budget_matched'] is True, 'fixture or incomplete pair')
        require(root['root_id'] == key[0], 'root identity mismatch')
        initial = compiled['input_context']
        require(root['context'] == dict(layout='states_with_between_state_actions_v1', context_length=3,
            latent_dim=192, action_dim=10, latent_history=initial['emb'][0], action_history=initial['actions'][0][1:]), 'full predictor root context mismatch')
        arms = index(row['rows'], ('arm',))
        require(set(arms) == ({('cem',), ('random',)} if sampling else {('composition',), ('flat_search',)}), 'arm census mismatch')
        for (arm_name,), arm in arms.items():
            require(arm['planning_predict_samples'] == 80 and arm['replay_predict_samples'] == 4
                    and arm['replay_error'] is None, 'incomplete selected replay')
            entries_match(arm)
            saved = arm['witness']
            require(saved['independent_full_replay_exact'] is True, 'saved replay missing')
            receipt, edges = saved['receipt'], saved['edges']
            require(receipt['root'] == root, 'receipt root mismatch')
            provenance = dict(checkpoint_sha256=CHECKPOINT_SHA, encoder_sha256=CHECKPOINT_SHA,
                source_sha256=report['source_seals'][0]['manifest_sha256'] if sampling else protocol['source_manifest_sha256'],
                protocol_sha256=witness.digest(protocol), metric_sha256=ready[key]['metric_sha256'],
                candidate_bank_sha256=witness.digest(dict(schedule=protocol['schedule'], seed=ready[key]['seed']))
                    if sampling else witness.digest(row['declaration']['bank']))
            require(receipt['provenance'] == provenance, 'selected provenance mismatch')
            require(witness.digest({k:v for k,v in receipt.items() if k != 'receipt_sha256'}) == receipt['receipt_sha256'], 'receipt digest mismatch')
            outputs = iter(deepcopy(e['output_context']) for e in edges)
            rebuilt = witness.compose(root=root, provenance=receipt['provenance'],
                action_domain=receipt['action_domain'], edges=edges,
                replay=lambda c,a: next(outputs), max_edges=receipt['max_edges'], max_actions=receipt['max_actions'])
            require(rebuilt == receipt and len(receipt['actions']) == 10, 'exact H2 witness mismatch')
            domain = row['action_domain'] if sampling else row['declaration']['action_domain']
            require(receipt['action_domain'] == domain, 'raw action domain mismatch')
            require(len(edges) == (2 if arm_name == 'composition' else 1), 'composition/flat witness structure mismatch')
            if sampling:
                require(arm['fixture_only'] is False and arm['prediction_budget_matched'] is True, 'sampling fixture/incomplete')
                attempts = arm['attempts']
                require(len(attempts) == 40 and [a['candidate_id'] for a in attempts] == [f'{arm_name}_{i:02d}' for i in range(40)], 'all forty frozen candidates required')
                for attempt in attempts:
                    require(attempt['completed'] is True and attempt['proposal_dimensions'] == 20
                            and attempt['packet_shape'] == [2,5,2], 'full20 candidate incomplete')
                    require([step for packet in attempt['commands'] for step in packet] == attempt['actions'], 'full20 mapping mismatch')
                    finite(attempt['cost'])
                best = min(attempts, key=lambda a:a['cost'])
                require(best['candidate_id'] == arm['selected_candidate_id'] and best['actions'] == receipt['actions']
                        and best['cost'] == arm['selected_model_cost'], 'saved incumbent mismatch')
            else:
                path, bank = arm['selected_path'], row['declaration']['bank']
                require(len(path) == 2 and all(type(i) is int and 0 <= i < 16 for i in path), 'bank path mismatch')
                require([step for i in path for step in bank[i]] == receipt['actions'], 'bank selected actions mismatch')
                expected_bank = [[r['program_live_commands'][0]]*5 for r in compiled['receipts']]
                require(bank == expected_bank, 'compiled bank mismatch')
                attempts = arm['attempts']
                require(len(attempts) == (80 if arm_name == 'composition' else 40), 'complete bank candidate ledger required')
                require(len({tuple(a['path']) for a in attempts}) == len(attempts)
                        and all(a['completed'] is True for a in attempts), 'duplicate or failed bank attempt')
                for attempt in attempts:
                    finite(attempt['cost'])
                terminals = [a for a in attempts if len(a['path']) == 2]
                require(len(terminals) == (64 if arm_name == 'composition' else 40), 'terminal search census mismatch')
                best = min(terminals, key=lambda a:a['cost'])
                require(best['path'] == path and best['cost'] == arm['selected_model_cost'], 'bank incumbent mismatch')
            selected[key[0], arm_name] = dict(root=root, receipt=receipt,
                predicted_q=finite(arm['selected_model_cost']), planning_samples=80, replay_samples=4)
    return selected


def join_physical(selected, physical, *, sampling, protocol=None):
    """Root/arm joins only; retain incomplete endpoints, never zip array orders."""
    branches = physical['rows'] if sampling else [dict(b, root_sha256=r['root_sha256'])
        for r in physical['rows'] for b in r['branches']]
    if not sampling:
        require(len(index(physical['rows'], ('root_sha256',))) == 4, 'physical root census mismatch')
    branches = index(branches, ('root_sha256', 'method' if sampling else 'arm'))
    require(branches.keys() == selected.keys(), 'physical stable root/arm census mismatch')
    plans = index(protocol['plans'], ('root_sha256', 'method')) if sampling else None
    if plans is not None:
        require(plans.keys() == selected.keys(), 'physical plan census mismatch')
    result = []
    for key in sorted(selected):
        plan, branch = selected[key], branches[key]
        actions = plan['receipt']['actions']
        if sampling:
            frozen = plans[key]
            require(witness.digest({k:v for k,v in frozen.items() if k != 'selection_sha256'}) == frozen['selection_sha256'], 'physical selection digest mismatch')
            require(frozen['root'] == plan['root'] and frozen['commands'] == actions
                    and frozen['receipt_sha256'] == plan['receipt']['receipt_sha256']
                    and branch['selection_sha256'] == frozen['selection_sha256'], 'physical selected receipt mismatch')
            identity = branch['context_identity']
            require(identity['history_rgb_sha256'] == plan['root']['root_rgb_sha256']
                    and identity['goal_rgb_sha256'] == plan['root']['goal_rgb_sha256'], 'physical root/goal RGB mismatch')
        else:
            require(physical['protocol']['selections'][key[0]+':'+key[1]] == actions, 'physical sealed actions mismatch')
        count, trace = branch['completed_commands'], branch['trace']
        require(type(count) is int and 0 <= count <= 10 and len(trace) == count, 'physical trace count mismatch')
        require(branch['planned_commands'] == 10 and branch['remaining_commands_not_executed'] == 10-count, 'physical horizon accounting mismatch')
        require([t['primitive_step'] for t in trace] == list(range(1,count+1))
                and [t['command'] for t in trace] == actions[:count], 'physical executed action prefix mismatch')
        require(not trace or trace[-1]['state'] == branch['final_state'], 'physical terminal state mismatch')
        official = branch['official']
        require(type(official['official_success']) is bool, 'explicit official success required')
        complete = branch['status'] == 'completed' and count == 10
        result.append(dict(root_sha256=key[0], arm=key[1], root_binding_sha256=witness.digest(plan['root']),
            status=branch['status'], completed_commands=count, fixed_horizon_complete=complete,
            predicted_q=plan['predicted_q'], object_pose_cost=finite(branch['pose']['pose_cost']),
            official_state_distance=finite(official['official_state_distance']), official_success=official['official_success'],
            endpoint_scope='fixed_H2' if complete else 'guard_or_unavailable_endpoint_not_H2'))
    return result


def decide(rows, *, integrity_errors=()):
    require(len(index(rows, ('root_sha256', 'arm'))) == 16, 'all sixteen selected branches required')
    roots = {}
    for row in rows:
        roots.setdefault(row['root_sha256'], []).append(row)
    require(len(roots) == 4 and all(len(v) == 4 for v in roots.values()), 'four common roots required')
    for group in roots.values():
        require({r['arm'] for r in group} == {'composition','flat_search','cem','random'}, 'four-arm census mismatch')
        require(len({r['root_binding_sha256'] for r in group}) == 1, 'cross-report full root/goal/context mismatch')
    comparisons = []
    for root_id, group in sorted(roots.items()):
        arms = {r['arm']:r for r in group}
        composition = arms['composition']
        for name in ('flat_search','cem','random'):
            baseline = arms[name]
            comparisons.append(dict(root_sha256=root_id, baseline=name,
                same_fixed_H2_endpoint=composition['fixed_horizon_complete'] and baseline['fixed_horizon_complete'],
                composition_minus_baseline={key:composition[key]-baseline[key] for key in
                    ('predicted_q','object_pose_cost','official_state_distance')},
                full_budget_causal_comparison_admissible=False,
                scope='known-root selected endpoints only; domain/total-budget/mechanism confounds remain'))
    return dict(schema=TAG+'.report', status='invalid_evidence' if integrity_errors else 'rejected_missing_scientific_evidence',
        evidence_integrity_valid=not integrity_errors, integrity_errors=list(integrity_errors),
        P1_admission=False, P2_admission=False, physical_efficacy_verified=False,
        sampling_completion_is_P2_success=False, fair_total_budget_comparison=False,
        prediction_samples_per_root_arm=dict(planning=80, independent_replay=4),
        known_calibration_roots=4, independent_outcome_roots=0, selected_branches=16,
        fixed_horizon_branches=sum(r['fixed_horizon_complete'] for r in rows),
        fixed_horizon_composition_flat_pairs=sum(all(r['fixed_horizon_complete'] for r in group
            if r['arm'] in ('composition','flat_search')) for group in roots.values()),
        official_successes=sum(r['official_success'] for r in rows), rows=rows,
        descriptive_comparisons=comparisons,
        unmeasured_or_unmatched_resources=['upstream encoding/bank construction', 'total FLOPs',
            'common wall-time cap', 'all-candidate physical budget', 'candidate evaluation/expansion equivalence'],
        necessary_scientific_gates={k:False for k in PREREQUISITES},
        executable_prerequisites=deepcopy(PREREQUISITES),
        unsupported_positive_admission=True,
        scope='historical selected-path calibration only; no prospective P1/P2 adapter, no causal composition effect or population CI',
        new_neural_calls=0, new_physical_steps=0, future_heldout_results_opened=False)


def audit(project, reader):
    project = Path(project)
    data = {key: reader.load(project/'results'/rel) for key,rel in FILES.items()}
    compilation = reader.load(project/'results'/COMPILATION, COMPILATION_SHA)
    for key in ('composition_protocol','sampling_protocol','physical_protocol'):
        reader.bindings(data[key]['bindings'])
    reader.bindings(data['composition_physical']['protocol']['bindings'])
    require(witness.digest(data['composition_sources']) == data['composition_protocol']['source_manifest_sha256'], 'composition source manifest digest mismatch')
    sources = data['composition_sources']
    require(bool(sources) and len({s['path'] for s in sources}) == len(sources), 'composition source census mismatch')
    for entry in sources:
        require(hashlib.sha256(entry['source'].encode()).hexdigest() == entry['sha256'], 'source mirror digest mismatch')
        reader.bind(entry['path'], entry['sha256'])
    for key in ('sampling','sampling_physical','composition_physical'):
        reader.seals(data[key]['source_seals'])
    require(data['sampling']['source_seals'] == data['sampling_prepared']['source_seals']
            and witness.digest(data['sampling_protocol']) == data['sampling_prepared']['protocol_sha256'], 'prepared/report source binding mismatch')
    first = validate_planning(data['composition'], data['composition_protocol'], compilation, sampling=False)
    second = validate_planning(data['sampling'], data['sampling_protocol'], compilation, sampling=True)
    rows = join_physical(first, data['composition_physical'], sampling=False)
    rows += join_physical(second, data['sampling_physical'], sampling=True, protocol=data['physical_protocol'])
    for key in ('composition_physical','sampling_physical'):
        physical = data[key]
        require(bool(physical['gates']) and all(v is True for v in physical['gates'].values())
                and not physical.get('failures'), 'physical integrity gates failed')
        branches = physical['rows'] if key == 'sampling_physical' else [b for r in physical['rows'] for b in r['branches']]
        for branch in branches:
            reader.bind(branch['terminal_rgb_path'], branch['terminal_rgb_sha256'])
            if 'journal_path' in branch:
                reader.bind(branch['journal_path'], branch['journal_sha256'])
    return decide(rows)


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def run(project, output):
    project, output = Path(project).resolve(), Path(output).absolute()
    output.mkdir(exist_ok=False)  # Never overwrite even an empty old output.
    reader = EvidenceReader()
    try:
        for path in (Path(__file__), project/'tests'/f'test_{TAG}.py', project/'docs'/f'{TAG}.md',
                     Path(witness.__file__), project/'experiments/spectral_reachability/d2p_p2_comparison_contract.py',
                     project.parent/'docs/D2P_PARALLEL_CUDA_PROGRESS_20261027.md',
                     project.parent/'docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md',
                     project.parent/'docs/p2_full20_sampling_20261027_v2.md',
                     project.parent/'docs/p2_full20_physical_20261027_v2.md'):
            reader.bind(path)
        result = audit(project, reader)
        reader.unchanged()
    except (ValueError, OSError, KeyError, TypeError, IndexError, StopIteration) as exc:
        result = dict(schema=TAG+'.report', status='invalid_evidence', evidence_integrity_valid=False,
            integrity_errors=[f'{type(exc).__name__}: {exc}'], P1_admission=False, P2_admission=False,
            executable_prerequisites=deepcopy(PREREQUISITES), new_neural_calls=0, new_physical_steps=0,
            future_heldout_results_opened=False)
    result['input_and_source_sha256'] = reader.hashes
    result['hash_scope'] = 'observed audit-time bytes plus verified historical seals; not independent pre-outcome freezing or execution proof'
    write_new(output/'report.json', result)
    summary = [f'# {TAG}', '', f'Status: {result["status"]}; P2_admission=false.', '',
               'No new neural/physics execution. Sampling completion is not P2 success.', '']
    summary += ['Integrity error: '+e for e in result.get('integrity_errors', [])]
    summary += ['', '## Required evidence (not executed)', '']
    summary += [f'- {key}: {value}' for key,value in PREREQUISITES.items()]
    with (output/'report.md').open('x') as stream:
        stream.write('\n'.join(summary)+'\n')
    print(json.dumps(dict(status=result['status'], P2_admission=False,
                         output=str(output), bound_files=len(reader.hashes))))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.project, args.output)
    return 2 if result['evidence_integrity_valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
