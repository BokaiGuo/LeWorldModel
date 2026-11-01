"""Fixed-cohort saved-event binding audit; no neural or physical replay.

Main-v2 correction is authoritative, not the older agent corrective artifact.
Only new output directories are allowed. No weights/datasets are loaded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p3_adaptive_runtime_accounting_audit_20261027 as correction
from experiments.spectral_reachability import d2p_p3_paired_accounting as paired
from experiments.spectral_reachability import d2p_planning_compute_ledger as ledger
from experiments.spectral_reachability import d2p_prerun_source_seal as seals
from experiments.spectral_reachability.d2p_remaining_stage_evidence import strict_json

NAME = 'p3_accounting_binding_audit_20261027'
OUTPUT = 'P3_ACCOUNTING_BINDING_AUDIT_20261027'
MAIN = 'results/P3_ADAPTIVE_RUNTIME_ACCOUNTING_AUDIT_20261027_main_v2/report.json'
MAIN_SHA = '2f82dcb6a0b9ff91f2c7ab86813b6660d37be2cd869850413a3c6e5ff088d840'
ORIGINAL = 'results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/'
REPORT = ORIGINAL + 'P3_ADAPTIVE_RUNTIME_report_20261027.json'
PROTOCOL = ORIGINAL + 'P3_ADAPTIVE_RUNTIME_protocol_20261027.json'
COMPILATION = 'results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= 32_000_000, 'bounded regular input required')
    with path.open('rb') as stream:
        raw = stream.read(32_000_001)
    require(len(raw) <= 32_000_000, 'input size limit')
    return raw


def protocol_id(protocol):
    # EXACT original runner serialization: default separators/ensure_ascii.
    return sha(json.dumps(protocol, sort_keys=True, allow_nan=False).encode())


def same(left, right):
    # Reject bool/int coercion and non-finite objects, unlike Python equality.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def path_indices(value, length):
    require(isinstance(value, list) and len(value) == length and
            all(type(i) is int and 0 <= i < 16 for i in value), 'invalid parent/primitive path indices')
    return tuple(value)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'finite nonnegative metric required')
    return value


def validate_protocol(report, envelope, compilation):
    p = envelope['protocol']
    identity = protocol_id(p)
    require(identity == envelope['protocol_id'] == report['protocol_id'] ==
            envelope['paired_plan']['source_seal'], 'protocol content SHA256 mismatch')
    sources = p['sources']
    require(isinstance(sources, dict) and len(sources) == 52, 'exact nonempty 52 sources required')
    require(all(isinstance(k, str) and Path(k).is_absolute() and isinstance(v, str) and
                len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                for k, v in sources.items()), 'source path/SHA binding invalid')
    roots = p['roots']
    require(isinstance(roots, list) and len(roots) == len(set(roots)) == 4, 'four unique protocol roots required')
    require(all(isinstance(r, str) and len(r) == 64 and all(c in '0123456789abcdef' for c in r)
                for r in roots), 'root SHA identity required')
    plan = envelope['paired_plan']
    require([r['contract']['binding']['root_id'] for r in plan['roots']] == roots,
            'protocol/paired-plan root order mismatch')
    require(compilation['protocol']['roots'] == roots and
            [r['root_sha256'] for r in compilation['rows']] == roots,
            'compilation cohort/order mismatch')
    methods = [plan['fixed_method'], plan['adaptive_method']]
    require(methods == p['methods'] == ['p3_adaptive_runtime_fixed', 'p3_adaptive_runtime_adaptive'],
            'protocol methods mismatch')
    require(p['predict_cap'] == 147 and p['planning_edge_cap'] == 144 and p['verification_reserve'] == 3,
            'unexpected frozen runtime caps')
    require(same(p['schedule'], dict(source_seal='protocol_id', max_depth=3, base_beam=4,
            expansion_factor=2, max_expansions=144, chi_threshold=.5, gain_threshold=.05,
            svd_relative_cutoff=1e-6)), 'unexpected frozen schedule')
    require(len(report['rows']) == 8 and len({(r['root_sha256'], r['method']) for r in report['rows']}) == 8 and
            {(r['root_sha256'], r['method']) for r in report['rows']} == {(r, m) for r in roots for m in methods},
            'complete unique root/method census required')
    for root in plan['roots']:
        binding = root['contract']['binding']
        require(binding['protocol_id'] == identity and binding['task_id'] == 'pusht_cached_latent_runtime' and
                binding['frozen_model_id'] ==
                '48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607:'
                '2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09',
                'paired-plan protocol/task/model binding mismatch')
        require(root['branch_ids'] == ['search_and_verification'], 'branch census mismatch')
    return identity


def bank_of(compiled):
    receipts = compiled['receipts']
    require(len(receipts) == 16 and [r['candidate_id'] for r in receipts] == [f'prefix_{i}' for i in range(16)],
            'ordered 16-candidate source bank required')
    bank = []
    for receipt in receipts:
        command = np.asarray(receipt['program_live_commands'][0])
        require(command.shape == (2,) and command.dtype.kind in 'fi' and np.isfinite(command).all() and
                np.all(np.abs(command) <= 1), 'finite in-box raw bank command required')
        bank.append(np.repeat(command[None], 5, axis=0))
    return bank


def validate_events(row, compiled, schedule):
    bank = bank_of(compiled)
    search = row['search']
    mode = row['method'].removeprefix('p3_adaptive_runtime_')
    require(mode in ('fixed', 'adaptive') and search['mode'] == mode, 'search/method mismatch')
    frontier, nodes, total, kernels_seen = [()], [], 0, 0
    width, terminal_status = schedule['base_beam'], 'completed'
    layers = search['layers']
    require(isinstance(layers, list) and 1 <= len(layers) <= schedule['max_depth'], 'nonempty bounded layers required')
    for depth, layer in enumerate(layers, 1):
        require(type(layer['depth']) is int and layer['depth'] == depth, 'layer depth order mismatch')
        require(layer['expected_expansions'] == len(frontier)*16, 'frontier expansion declaration mismatch')
        children, used, expected_parents, signals = [], 0, [], layer['signals']
        for parent in frontier:
            if total + used == schedule['max_expansions']:
                terminal_status = 'budget_exhausted'
                break
            expected_parents.append(parent)
            index = len(expected_parents)-1
            require(index < len(layer['kernels']), 'missing ordered parent kernel')
            kernel = layer['kernels'][index]
            require(path_indices(kernel['parent_indices'], depth-1) == parent, 'parent frontier/order mismatch')
            receipt = kernel['receipt']
            count = min(16, schedule['max_expansions'] - total - used)
            events = receipt['attempts']
            require(len(events) == count and receipt['attempted_expansions'] == count and
                    receipt['failed_attempts'] == 0 and receipt['max_depth'] == 1,
                    'kernel count/failure/depth mismatch')
            require(receipt['status'] == ('completed' if count == 16 else 'budget_exhausted'),
                    'kernel terminal status mismatch')
            for primitive, event in enumerate(events):
                require(path_indices(event['primitive_indices'], 1) == (primitive,) and
                        type(event['depth']) is int and event['depth'] == 1 and
                        event['replay_completed'] is True and event['succeeded'] is True and event['stage'] == 'metric',
                        'candidate event order/status mismatch')
                path = parent + (primitive,)
                node = (number(event['cost']), depth, path)
                children.append(node)
                nodes.append(node)
            used += count
            kernels_seen += 1
            if count < 16:
                terminal_status = 'budget_exhausted'
                break
        require(len(layer['kernels']) == len(expected_parents), 'extra/duplicate parent kernel')
        complete = used == layer['expected_expansions'] and terminal_status == 'completed'
        require(type(layer['layer_complete']) is bool and layer['layer_complete'] == complete and
                layer['attempted_expansions'] == used, 'layer completion/event mismatch')
        if complete:
            require(len(signals) == len(expected_parents) and bool(signals), 'complete frontier signals required')
            for signal in signals:
                require(signal['children'] == 16 and all(type(signal[k]) in (int,float) and
                    math.isfinite(signal[k]) and 0 <= signal[k] <= 1 for k in
                    ('unreachable_fraction', 'horizon_gain')), 'invalid saved scheduler signals')
            horizon = 3 if max(s['horizon_gain'] for s in signals) >= schedule['gain_threshold'] else 1
            proposed = schedule['base_beam'] * (schedule['expansion_factor'] if
                max(s['unreachable_fraction'] for s in signals) >= schedule['chi_threshold'] else 1)
            require(same(layer['decision'], dict(horizon_tokens=horizon, expanded_candidates=proposed,
                    single_pass_predict_samples=horizon*proposed)), 'saved signal/decision mismatch')
            if mode == 'adaptive':
                width = proposed
        else:
            require(layer['decision'] is None, 'incomplete layer cannot authorize decision')
        require(type(layer['beam_width']) is int and layer['beam_width'] == width, 'beam width mismatch')
        children.sort()
        frontier = [n[2] for n in children[:width]]
        require(layer['retained'] == len(frontier), 'retained frontier mismatch')
        total += used
        if not complete or not frontier or (mode == 'adaptive' and horizon == 1):
            if complete and frontier and mode == 'adaptive' and horizon == 1:
                terminal_status = 'dynamics_horizon_stop'
            require(depth == len(layers), 'layers continued after causal stop')
            break
    require(search['status'] == terminal_status and
            (terminal_status != 'completed' or len(layers) == schedule['max_depth']), 'search stop/depth mismatch')
    require(total == search['attempted_expansions'] == search['metric_calls'], 'global event count mismatch')
    best = search['best_plan']
    winner = min(nodes)
    require(path_indices(best['primitive_indices'], best['depth']) == winner[2] and
            best['depth'] == winner[1] and best['cost'] == winner[0], 'selected global minimum/tiebreak mismatch')
    actions = np.asarray(best['actions'])
    require(actions.dtype.kind in 'fi' and np.isfinite(actions).all() and np.all(np.abs(actions) <= 1),
            'selected actions out of box/nonfinite')
    require(np.array_equal(actions, np.concatenate([bank[i] for i in winner[2]])), 'selected raw bank actions mismatch')
    resources = correction.corrected_resources(row)
    require(row['action_encoder_forwards'] == row['action_encoder_samples'] == resources.predict_samples,
            'batch-one action encoder counts mismatch')
    return dict(root_sha256=row['root_sha256'], method=row['method'], planning_events=total,
                kernels=kernels_seen, selected_path=list(winner[2]), selected_depth=winner[1],
                actual_horizon=resources.max_planning_horizon_tokens,
                corrected_resources=asdict(resources), independent_neural_replay=False)


def audit(report, protocol, compilation, main):
    identity = validate_protocol(report, protocol, compilation)
    source = {row['root_sha256']: row['compiled'] for row in compilation['rows']}
    rows = [validate_events(row, source[row['root_sha256']], protocol['protocol']['schedule']) for row in report['rows']]
    rebuilt = correction.audit(report, protocol)
    require(all(k in main and same(v, main[k]) for k, v in rebuilt.items()), 'main_v2 correction differs from recomputation')
    return dict(schema=NAME+'.v1', status='saved_binding_audit_pass', protocol_id=identity,
        sources_count=52, root_count=4, root_method_rows=len(rows), rows=rows,
        planning_events=sum(r['planning_events'] for r in rows), kernels=sum(r['kernels'] for r in rows),
        corrected_rows=rebuilt['corrected_rows'], actual_predict_samples=rebuilt['actual_predict_samples'],
        paired=rebuilt['paired'], all_checks_pass=True,
        scope='saved successful events, causal frontier/order and raw action bindings; not independent transition/SVD/encoding or CUDA trace verification',
        original_count_reconstruction_reused=True, independent_neural_replay=False,
        new_predict_samples=0, new_physical_steps=0, P1_admission=False, P3_admission=False,
        efficacy_claim=False, compute_advantage=False)


def run(project_root, output=None):
    root = Path(project_root).resolve()
    output = Path(output) if output is not None else root/'results'/OUTPUT
    require(not output.exists(), 'new exclusive output required')
    loaded, bindings = {}, {}
    for key, relative in (('main',MAIN), ('report',REPORT), ('protocol',PROTOCOL), ('compilation',COMPILATION)):
        path = root/relative
        raw = read(path)
        bindings[str(path)] = sha(raw)
        loaded[key] = strict_json(raw)
    require(bindings[str(root/MAIN)] == MAIN_SHA, 'authoritative main_v2 report SHA mismatch')
    main, protocol = loaded['main'], loaded['protocol']
    require(main['inputs'] == {str(root/p):bindings[str(root/p)] for p in (REPORT,PROTOCOL)},
            'main_v2 original input hash bindings differ')
    require(protocol['protocol']['bindings'].get(str(root/COMPILATION)) == bindings[str(root/COMPILATION)],
            'source compilation hash binding differs')
    result = audit(loaded['report'], protocol, loaded['compilation'], main)
    # The sealed source map also contains two repository-level docs, outside
    # the Python subdirectory. Never follow weight/dataset bindings.
    for path, expected in protocol['protocol']['sources'].items():
        require(Path(path).resolve().is_relative_to(root.parent), 'source path outside repository')
        require(sha(read(path)) == expected, 'original current source SHA mismatch: '+path)
        bindings[path] = expected
    old = main['source_seal']
    require(Path(old['output_dir']).resolve() == (root/MAIN).parent/'source_seal', 'wrong main_v2 seal directory')
    verified = seals.verify_seal(output_dir=old['output_dir'], expected_manifest_sha256=old['manifest_sha256'])
    files = [Path(__file__), root/'tests'/('test_'+NAME+'.py'), root.parent/'docs'/(OUTPUT+'.md'),
             Path(correction.__file__), Path(paired.__file__), Path(ledger.__file__), Path(seals.__file__),
             root/'experiments/spectral_reachability/d2p_remaining_stage_evidence.py']
    output.mkdir(exist_ok=False)
    seal = seals.create_seal(files=files, args_json=json.dumps(bindings), output_dir=output/'source_seal')
    require(all(sha(read(path)) == expected for path, expected in bindings.items()), 'inputs/source changed during audit')
    result.update(inputs=bindings, authoritative_corrected_report=str(root/MAIN),
                  authoritative_corrected_report_sha256=MAIN_SHA, main_v2_seal_verification=verified,
                  source_seal=seal, source_verification=seals.verify_seal(
                      output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256']))
    with (output/'report.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','sources_count','root_method_rows','planning_events',
                                          'kernels','corrected_rows','actual_predict_samples')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.project_root)
