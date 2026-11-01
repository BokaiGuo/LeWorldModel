"""Standard-library-only saved-evidence audit. Never imports neural/simulator code."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

NAME = 'p3_quality_cost_audit_20261027_v2'
MAIN = 'results/P3_ADAPTIVE_RUNTIME_ACCOUNTING_AUDIT_20261027_main_v2/report.json'
BINDING = 'results/P3_ACCOUNTING_BINDING_AUDIT_20261027/report.json'
PHYSICAL = 'results/P3_MATCHED_PHYSICAL_20261027_v1_cuda/report.json'
METHODS = ('p3_adaptive_runtime_fixed', 'p3_adaptive_runtime_adaptive')
BACKWARD = ('action_backward_calls', 'action_vjp_rows', 'input_backward_calls', 'input_vjp_rows')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def decode(raw):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    canonical(value)  # Reject overflowed numeric literals too.
    return value


def index(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        require(key not in result, 'ambiguous duplicate join: '+str(key))
        result[key] = row
    return result


def count(value):
    require(type(value) is int and value >= 0, 'nonnegative integer count required')
    return value


class Evidence:
    """Read explicit bounded files, bind all consumed bytes, check again at end."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def raw(self, path, expected=None):
        path = Path(path)
        require('..' not in path.parts, 'path traversal')
        require(path.is_absolute(), 'absolute evidence path required')
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink evidence')
        require(path.resolve().is_relative_to(self.root.parent), 'evidence outside repository')
        require(path.is_file() and path.stat().st_size <= 32_000_000, 'bounded regular evidence required')
        raw = path.read_bytes()
        require(len(raw) <= 32_000_000, 'evidence too large')
        digest = sha(raw)
        require(expected is None or digest == expected, 'SHA256 mismatch: '+str(path))
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest, 'evidence changed')
        self.hashes[str(path)] = digest
        return raw

    def json(self, path, expected=None):
        return decode(self.raw(path, expected))

    def seal(self, receipt):
        directory = Path(receipt['output_dir'])
        manifest = self.json(directory/'manifest.json', receipt['manifest_sha256'])
        require(manifest['schema'] == 'd2p_prerun_source_seal.v1', 'unknown seal schema')
        args = self.json(directory/'args.json', manifest['args_sha256'])
        if 'protocol_path' in args:
            self.raw(Path(args['protocol_path']), args['protocol_sha256'])
        require(bool(manifest['files']), 'empty source seal')
        index(manifest['files'], ('source',))
        for entry in manifest['files']:
            archive = Path(entry['archive'])
            require(not archive.is_absolute() and '..' not in archive.parts, 'unsafe archive path')
            raw = self.raw(directory/archive, entry['sha256'])
            require(len(raw) == entry['size'], 'archive size differs')
            self.raw(Path(entry['source']), entry['sha256'])
        return dict(manifest_sha256=receipt['manifest_sha256'], status='live_and_archive_verified',
                    source_count=len(manifest['files']))

    def recheck(self):
        for path, digest in list(self.hashes.items()):
            self.raw(Path(path), digest)


def audit(main, binding, physical, original, runtime_protocol, physical_protocol):
    """Join by full SHA root/method/repetition, never row positions or outcomes."""
    require(main['status'] == 'saved_accounting_corrected', 'wrong corrected status')
    require(binding['status'] == 'saved_binding_audit_pass' and binding['all_checks_pass'] is True,
            'binding audit not passed')
    require(physical['status'] == 'calibration_recorded', 'physical calibration missing')
    for report in (main, binding, physical):
        require(report['P1_admission'] is False and report['P3_admission'] is False, 'admission scope changed')
    protocol_id = sha(canonical(runtime_protocol['protocol']).encode())
    require(protocol_id == runtime_protocol['protocol_id'] == original['protocol_id'] == binding['protocol_id'],
            'runtime protocol identity differs')
    roots = runtime_protocol['protocol']['roots']
    require(len(roots) == len(set(roots)) == 4 and all(isinstance(r, str) and len(r) == 64 and
            all(c in '0123456789abcdef' for c in r) for r in roots), 'four frozen SHA roots required')
    expected = {(root, method) for root in roots for method in METHODS}
    fields = ('root_sha256', 'method')
    corrected = index(main['corrections'], fields)
    bound = index(binding['rows'], fields)
    saved = index(original['rows'], fields)
    plans = index(physical_protocol['plans'], fields)
    require(binding['paired'] == main['paired'], 'corrected paired accounting differs')
    paired = {p['binding']['root_id']: p for p in main['paired']['pairs']}
    require(len(paired) == len(main['paired']['pairs']) == len(roots) and set(paired) == set(roots),
            'corrected pair root census differs')
    for mapping in (corrected, bound, saved, plans):
        require(set(mapping) == expected, 'missing/extra root-arm join')
    require(physical_protocol['physical_repetitions'] == 2, 'frozen two-repeat census required')
    branches = index(physical['rows'], fields+('repetition',))
    encodings = index(physical['encodings'], fields+('repetition',))
    expanded = {key+(rep,) for key in expected for rep in (0, 1)}
    require(set(branches) == set(encodings) == expanded and all(type(k[2]) is int for k in branches),
            'missing/extra physical or encoding repeat join')
    rows, groups = [], []
    for root in roots:
        for method in METHODS:
            key = root, method
            c, b, s, plan = corrected[key], bound[key], saved[key], plans[key]
            selection = {k:v for k,v in plan.items() if k != 'selection_sha256'}
            require(sha(canonical(selection).encode()) == plan['selection_sha256'], 'plan SHA mismatch')
            best = s['search']['best_plan']
            require(c['best_path'] == b['selected_path'] == plan['primitive_indices'] == best['primitive_indices'],
                    'selected path join mismatch')
            require(best['actions'] == plan['commands'] and
                    len(plan['commands']) == plan['planned_commands'] == 5*best['depth'], 'selected commands mismatch')
            require(best['depth'] == b['selected_depth'] == plan['planned_horizon_tokens'] == len(c['best_path']),
                    'selected horizon mismatch')
            require(c['actual_horizon'] == b['actual_horizon'] == plan['corrected_search_horizon'] ==
                    b['corrected_resources']['max_planning_horizon_tokens'], 'search horizon mismatch')
            receipt = s['prediction_receipt']
            require(plan['original_prediction_receipt'] == receipt and receipt['method'] == method and
                    receipt['binding']['root_id'] == root and receipt['binding']['protocol_id'] == protocol_id,
                    'predict receipt identity mismatch')
            phases = index(receipt['phases'], ('phase',))
            require(set(phases) == {('planning',), ('evaluation',)}, 'planning/verification phases missing')
            planning, verification = phases['planning',], phases['evaluation',]
            for phase in (planning, verification):
                require(phase['method'] == method and phase['completed'] is True and
                        phase['failed_predict_calls'] == 0, 'incomplete prediction evidence')
                count(phase['predict_samples']); count(phase['predict_calls'])
            require(planning['predict_samples'] == b['planning_events'] == s['search']['attempted_expansions'] and
                    verification['predict_samples'] == best['depth'], 'phase/event sample mismatch')
            resources = b['corrected_resources']
            arm = paired[root]['baseline' if method == METHODS[0] else 'candidate']
            require(arm['method'] == method and arm['resources'] == resources and
                    receipt['binding'] == paired[root]['binding'], 'authoritative paired resources differ')
            require(resources['predict_samples'] == c['predict_samples'] == receipt['attempted_predict_samples'] ==
                    planning['predict_samples']+verification['predict_samples'], 'prediction total mismatch')
            require(resources['predict_calls'] == planning['predict_calls']+verification['predict_calls'], 'call total mismatch')
            require(c['search_status'] == s['search']['status'], 'search stop differs')
            for name in BACKWARD:
                count(resources[name])
                require(resources[name] == s['resources'][name], 'backward counter mismatch')
            repeats = []
            for rep in (0, 1):
                r, enc = branches[key+(rep,)], encodings[key+(rep,)]
                require(r['selection_sha256'] == plan['selection_sha256'] and
                        r['planned_horizon_tokens'] == best['depth'] and r['planned_commands'] == plan['planned_commands'] and
                        r['corrected_search_horizon'] == c['actual_horizon'], 'physical selection join differs')
                attempted, returned = count(r['physical_env_step_attempts']), count(r['physical_env_step_returns'])
                require(returned <= attempted <= r['planned_commands'] and
                        returned == r['completed_commands'] == r['actual_endpoint_primitive_steps'] == len(r['trace']) and
                        attempted == r['attempted_commands'] and
                        r['remaining_commands_not_executed'] == r['planned_commands']-returned, 'physical count inconsistency')
                require(r['guard']['root_id'] == root, 'guard root differs')
                require(type(r['guard']['abstained']) is bool and type(r['full_horizon_completed']) is bool,
                        'boolean guard/completion required')
                require(r['full_horizon_completed'] == (r['status'] == 'completed' and returned == r['planned_commands']),
                        'completion flag inconsistent')
                require(not r['guard']['abstained'] or not r['full_horizon_completed'], 'aborted guard marked complete')
                for metric in ('pose', 'official'):
                    require(isinstance(r.get(metric), dict), 'missing actual endpoint metrics')
                    for v in r[metric].values():
                        require(type(v) is bool or (type(v) in (int,float) and math.isfinite(v) and v >= 0),
                                'invalid endpoint metric')
                require(enc['status'] == 'completed' and enc['attempted_encode_samples'] == 1, 'encoding receipt incomplete')
                row = dict(root_sha256=root, method=method, repetition=rep,
                    dependence_group=root+'/'+method, selection_sha256=plan['selection_sha256'],
                    search_status=c['search_status'], actual_search_horizon=c['actual_horizon'],
                    selected_horizon_tokens=best['depth'], actual_executed_primitive_steps=returned,
                    actual_executed_token_equivalent=returned/5, planned_commands=r['planned_commands'],
                    physical_env_step_attempts=attempted, physical_env_step_returns=returned,
                    remaining_commands_not_executed=r['remaining_commands_not_executed'], status=r['status'],
                    full_horizon_completed=r['full_horizon_completed'], guard=r['guard'],
                    failure=r.get('failure'), failures=r['failures'],
                    planning_predict_samples=planning['predict_samples'], verification_predict_samples=verification['predict_samples'],
                    total_predict_samples=resources['predict_samples'], backward_and_vjp_counts={k:resources[k] for k in BACKWARD},
                    jacobian_materialization_count=None,
                    jacobian_count_scope='not separately instrumented; VJP rows are not dense Jacobian counts',
                    saved_action_encoder_samples=s['action_encoder_samples'], terminal_encode_attempts=enc['attempted_encode_samples'],
                    saved_search_resources=resources, pose=r['pose'], officialstate=r['official'],
                    endpoint_scope='full_selected_path' if r['full_horizon_completed'] else 'partial_guard_or_failure_endpoint')
                rows.append(row); repeats.append(row)
            compare_keys = ('status','actual_executed_primitive_steps','guard','pose','officialstate','failure','failures')
            groups.append(dict(root_sha256=root, method=method, repetitions=[0,1], planning_charged_once=True,
                repeated_endpoints_equal=all(repeats[0][k] == repeats[1][k] for k in compare_keys),
                independent_sample_count_claim=False))
    ri = index(rows, fields+('repetition',))
    comparisons = []
    for root in roots:
        f, a = bound[root,METHODS[0]], bound[root,METHODS[1]]
        fr, ar = f['corrected_resources'], a['corrected_resources']
        for rep in (0,1):
            left, right = ri[root,METHODS[0],rep], ri[root,METHODS[1],rep]
            matches = dict(search_horizon=left['actual_search_horizon']==right['actual_search_horizon'],
                selected_horizon=left['selected_horizon_tokens']==right['selected_horizon_tokens'],
                predict_samples=fr['predict_samples']==ar['predict_samples'],
                planning_samples=left['planning_predict_samples']==right['planning_predict_samples'],
                verification_samples=left['verification_predict_samples']==right['verification_predict_samples'],
                saved_resource_vector=fr==ar,
                physical_attempts=left['physical_env_step_attempts']==right['physical_env_step_attempts'],
                actual_endpoint_steps=left['actual_executed_primitive_steps']==right['actual_executed_primitive_steps'])
            full = left['full_horizon_completed'] and right['full_horizon_completed']
            all_matched = all(matches.values())
            comparisons.append(dict(root_sha256=root,repetition=rep,matches=matches,
                unmatched_dimensions=[k for k,v in matches.items() if not v], both_full_horizon=full,
                classification=('matched_full_path_saved_resources' if full else 'matched_partial_guard_endpoint')
                    if all_matched else 'unmatched',
                identical_selected_path=f['selected_path']==a['selected_path'],
                identical_initial_context=branches[root,METHODS[0],rep]['context_identity']==branches[root,METHODS[1],rep]['context_identity'],
                adaptive_minus_fixed_pose_cost=right['pose']['pose_cost']-left['pose']['pose_cost'],
                adaptive_minus_fixed_official_state_distance=right['officialstate']['official_state_distance']-left['officialstate']['official_state_distance'],
                quality_delta_scope='descriptive paired endpoint arithmetic, not efficacy', total_flops_equal=False))
            require(comparisons[-1]['identical_initial_context'], 'initial root context join mismatch')
    total_planning = sum(r['planning_events'] for r in bound.values())
    total_verification = sum(r['selected_depth'] for r in bound.values())
    attempts = sum(r['physical_env_step_attempts'] for r in rows)
    require(total_planning == binding['planning_events'] and total_planning+total_verification ==
            binding['actual_predict_samples'] == main['actual_predict_samples'], 'cohort predict totals differ')
    require(attempts == physical['selected_path_env_step_attempts'] and
            sum(r['physical_env_step_returns'] for r in rows) == physical['selected_path_env_step_returns'], 'physical totals differ')
    return dict(schema=NAME+'.v1',status='offline_quality_cost_audit_verified',
        root_count=len(roots),root_arm_count=len(expected),physical_execution_count=len(rows),
        planning_predict_samples_once_per_root_arm=total_planning,
        verification_predict_samples_once_per_root_arm=total_verification,
        total_saved_predict_samples_once_per_root_arm=total_planning+total_verification,
        selected_path_physical_step_attempts_all_repeats=attempts,
        guard_aborted_executions=sum(r['guard']['abstained'] for r in rows),
        classifications_per_execution_pair=dict(Counter(c['classification'] for c in comparisons)),
        rows=rows,comparisons=comparisons,dependence_groups=groups,
        scope='P1 unpassed; later-stage engineering calibration only; no new neural or simulator work',
        resource_scope='search+verification counters; terminal encodes separate; compiler/reset/warmup/upstream costs excluded',
        backward_scope='saved runtime ledger only, not upstream compilation/Jacobian cost',
        repeat_scope='two dependent replays per selected root-arm; shared planning not double charged; no confidence intervals',
        P1_admission=False,P3_admission=False,adaptive_superiority=False,efficacy_claim=False,
        total_flops_equality=False,confidence_intervals_over_repeats=False,new_predict_samples=0,new_physical_steps=0)


def markdown(result):
    lines = ['# P3 saved resource / physical quality audit', '', result['scope'], '',
        'Search H is the executed search depth, selected H is the saved path length. Actual steps are simulator returns; fractional token equivalents do not assert complete tokens.', '',
        '| root SHA prefix | arm | repeat | search H | selected H | actual steps | plan / verify samples | status | pose cost | officialstate distance |',
        '|---|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in result['rows']:
        lines.append(f"| {r['root_sha256'][:12]} | {r['method'].removeprefix('p3_adaptive_runtime_')} | {r['repetition']} | {r['actual_search_horizon']} | {r['selected_horizon_tokens']} | {r['actual_executed_primitive_steps']} | {r['planning_predict_samples']} / {r['verification_predict_samples']} | {r['status']} | {r['pose']['pose_cost']:.9f} | {r['officialstate']['official_state_distance']:.9f} |")
    lines += ['', '| root SHA prefix | repeat | classification | unmatched dimensions |', '|---|---:|---|---|']
    for c in result['comparisons']:
        lines.append(f"| {c['root_sha256'][:12]} | {c['repetition']} | {c['classification']} | {', '.join(c['unmatched_dimensions']) or 'none'} |")
    lines += ['', 'Full SHA identities, all pose/officialstate fields, failures, guards and per-counter evidence are in report.json.', '',
        f"Saved predictor samples: planning {result['planning_predict_samples_once_per_root_arm']}, verification {result['verification_predict_samples_once_per_root_arm']}; charged once per root-arm. Physical attempts across repeats: {result['selected_path_physical_step_attempts_all_repeats']}. Guard-aborted records retained: {result['guard_aborted_executions']}.", '',
        'All saved runtime action/input backward calls and VJP row counts are zero. Dense Jacobian materializations are not separately evidenced (null, not assumed zero). This does not cover upstream compilation.', '',
        'Equal saved counter vectors are not equal total FLOPs. Matched partial guard endpoints are not full-horizon successes. Identical selected paths are disclosed. Repeat equality is a reproducibility observation, not independent efficacy evidence.', '',
        result['resource_scope'], '', result['repeat_scope'], '',
        'No adaptive superiority, P3 admission, total-FLOPs equality, or repeat confidence intervals are claimed.', '']
    return '\n'.join(lines)


def run(root):
    root = Path(root).resolve()
    output = root/'results'/NAME
    require(not output.exists(), 'exclusive NEW result directory required')
    evidence = Evidence(root)
    main, binding, physical = [evidence.json(root/p) for p in (MAIN,BINDING,PHYSICAL)]
    for report in (main,binding):
        for path,digest in report['inputs'].items():
            evidence.raw(Path(path),digest)
    require(binding['authoritative_corrected_report'] == str(root/MAIN) and
            binding['authoritative_corrected_report_sha256'] == evidence.hashes[str(root/MAIN)], 'authoritative hash differs')
    original_path = root/'results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_report_20261027.json'
    protocol_path = original_path.with_name('P3_ADAPTIVE_RUNTIME_protocol_20261027.json')
    original, runtime = evidence.json(original_path), evidence.json(protocol_path)
    phys_protocol = evidence.json((root/PHYSICAL).parent/'protocol.json')
    # Deliberately do not load weights or datasets. Model/config digests remain upstream assertions.
    for path,digest in phys_protocol['bindings'].items():
        if Path(path).resolve().is_relative_to(root.parent):
            evidence.raw(Path(path),digest)
    seals = [evidence.seal(r) for r in (main['source_seal'],binding['source_seal'],*physical['source_seals'])]
    result = audit(main,binding,physical,original,runtime,phys_protocol)
    for r in physical['rows']:
        journal = evidence.raw(Path(r['journal_path']),r['journal_sha256'])
        events = [decode(line) for line in journal.splitlines()]
        for event in events:
            if 'rgb_path' in event:
                evidence.raw(Path(event['rgb_path']),event['rgb_file_sha256'])
        require(sum(e['kind']=='attempt' for e in events)==r['physical_env_step_attempts'] and
                sum(e['kind']=='step_returned' for e in events)==r['physical_env_step_returns'], 'journal count mismatch')
        observations = [e for e in events if e['kind']=='observation' and 'command' in e]
        require(len(observations) == len(r['trace']) and all(
            all(e.get(k) == v for k,v in t.items()) for t,e in zip(r['trace'],observations,strict=True)),
            'journal trace mismatch')
        evidence.raw(Path(r['terminal_rgb_path']),r['terminal_rgb_sha256'])
    sources = [Path(__file__).resolve(), root/'tests'/('test_'+NAME+'.py'),root.parent/'docs'/(NAME+'.md')]
    source_bytes = {str(p):evidence.raw(p) for p in sources}
    evidence.recheck()
    result.update(inputs_sha256=dict(sorted(evidence.hashes.items())), upstream_source_verifications=seals,
        provenance_scope='all consumed repository bytes hashed and rechecked; source live/archive checked; weights/datasets not loaded; hashes are integrity not execution proof',
        external_model_bindings_not_rehashed={p:h for p,h in phys_protocol['bindings'].items()
            if not Path(p).resolve().is_relative_to(root.parent)})
    output.mkdir(exist_ok=False)
    archive = output/'sources'; archive.mkdir()
    for p,raw in source_bytes.items():
        with (archive/Path(p).name).open('xb') as stream:
            stream.write(raw)
    with (output/'report.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with (output/'report.md').open('x') as stream:
        stream.write(markdown(result))
    evidence.recheck()
    artifacts = {str(p.relative_to(output)):sha(p.read_bytes()) for p in
        [output/'report.json',output/'report.md',*(archive/Path(p).name for p in source_bytes)]}
    with (output/'manifest.json').open('x') as stream:
        stream.write(json.dumps(dict(artifacts_sha256=artifacts,inputs_sha256=evidence.hashes),indent=2,sort_keys=True)+'\n')
    print(canonical({k:result[k] for k in ('status','root_count','physical_execution_count','guard_aborted_executions','classifications_per_execution_pair')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    run(parser.parse_args().project_root)
