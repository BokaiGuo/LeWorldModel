"""Prospective fixed-five-step execution contract, never a physics dispatcher.

Reuses sealed root search and native replay via callable execute_panel. No
model loader, network, training, CPU neural fallback, or old report reanalysis.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_common_budget_20261028_v1 as common

native, signal, core = common.native, common.signal, common.core
ROOT = common.ROOT
NAME = 'agent_p3_fixed_execution_contract_20261028_v1'
FLAGS = dict(common.FLAGS)
require, canonical, digest, sha = common.require, common.canonical, common.digest, common.sha


def contract():
    return dict(schema=NAME+'.contract', requested_execution_steps=5, execution_packets=1,
        planning_horizons=[1, 3], planning_depth_semantics='joint root lookahead packets, not closed-loop child expansions',
        executed_extension_depth=1, child_replanning=False, no_root_signal_for_child=True,
        equal_requested_execution_horizon=True, equal_actual_physical_execution_horizon=None,
        actual_physics_status='unknown_not_run',
        selection='unchanged full planning-horizon terminal modelQ; NOT common physical terminal outcome',
        action_selection='only exact first raw packet of full selected plan; all later packets NOT executed',
        shorter_lookahead_extension='none: never repeat, zero-fill, or extend H1 to H3',
        native_stop='do not pad/retry/replace: keep early stop/error and full denominator',
        signal_predict_per_task_method=8, search_predict_per_task_method=24,
        selected_prefix_predict_per_task_method=1, total_predict_per_task_method=33,
        candidates_by_horizon={'1': 12, '3': 4},
        same_predict_API_and_samples=True, same_FLOPs_or_wall=False,
        fixed_H1_cost_minimal_baseline=False, controlled_dual_signal_pipeline=True,
        request_is_not_physical_release=True, complete_C3_tree=False,
        future_receding_horizon='requires actual new observation/history and newly paid same-child signals; absent here',
        common_rule=common.rule(), **FLAGS)


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def profile(base):
    result = deepcopy(base); caps = result['resource_caps']
    # Four real winner-prefix validations, not wasted budget completion calls.
    for resource, shape in [('predict', [1, 3, 192]), ('action_encoder', [1, 3, 10]),
                            ('predictor', [1, 3, 192]), ('pred_proj', [3, 192]), ('replay', None), ('metric', None)]:
        change = core.delta(resource, shape)
        for axis in core.AXES: caps[axis] += 4*change[axis]
    return result


def queue(base):
    require(len(base['branches']) == 4, 'full task/method denominator')
    common.equal_execution_horizon(base['branches'])  # checks exact pairing, not equality of planning H.
    result = []
    for b in base['branches']:
        require(type(b['horizon']) is int and b['horizon'] in (1, 3) and type(b['candidates']) is int and
            b['candidates'] >= 4 and b['candidates']*2*b['horizon'] == 24 and
            b['original_decision']['program_type'] == 'progress', 'useful root-only quota with no padding/reconfiguration substitute')
        task = core.validate_task(base['tasks'][b['slot']])
        require(len(b['requests']) == b['candidates'] and all(req['candidate_index'] == k and
            req['budget_candidates'] == b['candidates'] and req['horizon'] == b['horizon'] and
            req['parent_context'] == task['root_context'] and req['parent_context_sha256'] == digest(task['root_context']) and
            req['binding_sha256'] == digest(task['binding']) and req['prefix_actions'] == req['prefix_edges'] == []
            for k, req in enumerate(b['requests'])), 'all exact real root requests; no child prefix or stale task')
        cost = b['signal_resources']
        require(cost['predict_API'] == cost['predict_samples'] == 8 and cost['jacobian_API'] == 2 and cost['VJP_directions'] == 384,
            'full independent paid H1/H3 signals required, no zero-cost derivative')
        result.append(dict(slot=b['slot'], mode=b['mode'], task_binding_sha256=digest(base['tasks'][b['slot']]['binding']),
            root_context_sha256=digest(base['tasks'][b['slot']]['root_context']),
            planning_horizon_packets=b['horizon'], optimized_raw_coordinates=10*b['horizon'],
            actual_child_expansion_depth=0, requested_tree_depth=b['original_decision']['program_depth_limit'],
            requested_execution_packets=1, requested_execution_steps=5,
            candidates=b['candidates'], search_requests=deepcopy(b['requests']),
            selected_prefix_validation=dict(operation='independent_selected_root_prefix_replay', packets=1,
                action_source='selected_plan.actions[:5]', model_calls=1, truncate_after_selection=True),
            resources=dict(signal=deepcopy(b['signal_resources']), search_predict_API=24, search_predict_samples=24,
                prefix_predict_API=1, prefix_predict_samples=1, total_predict_API=33, total_predict_samples=33),
            commands=None, model_selected_prefix=None, physical_result=None,
            status='pending_search_and_prefix_replay', physical_release=False))
    return result


def material(common_prepared, expected):
    p = common.verify(common_prepared, expected)
    require(p['rule'] == common.rule(), 'unmodified sealed base rule')
    files = set(map(Path, p['sources'])) | set(own_files()) | set(common.own_files()) | {Path(common_prepared).resolve()}
    return dict(schema=NAME+'.prepare', common_prepared=str(Path(common_prepared).resolve()), common_prepare_sha256=expected,
        contract=contract(), base=p, queue=queue(p), profile=profile(p['profile']),
        sources={str(f): sha(f) for f in sorted(files)},
        status='prepared_pending_MAIN_model_runner_and_new_physical_adapter',
        actual_physics_status='unknown_not_run', outcomes_used=False,
        missing_dependencies=['MAIN-reviewed runner must call execute_panel with real native hooks and durable ledger',
            'saved native closure strict consumer for this new combined search/prefix run',
            'new fixed-five-step physical recipe/root/goal binding, metered consumer and MAIN scheduling',
            'independent P1/applicable P2 admission and certified fresh cohort',
            'new observation/child-local signal adapters before any subsequent closed-loop action'], **FLAGS)


def exclusive(path):
    p = Path(path).absolute()
    require(p.parent == ROOT/'results' and p.name.startswith(NAME) and not p.exists() and not p.is_symlink(), 'exclusive new own result')
    return p


def write_new(path, value):
    p = exclusive(path); raw = canonical(value); require(len(raw) <= 16*1024**2, 'bounded metadata')
    with p.open('xb') as f:
        require(f.write(raw) == len(raw), 'complete metadata write'); f.flush(); os.fsync(f.fileno())
    signal.fsync_dir(p.parent); return dict(path=str(p), sha256=sha(p))


def verify(path, expected):
    core.hash_value(expected); require(sha(path) == expected, 'external actual prepare SHA')
    p = signal.read(path)
    require(canonical(p) == canonical(material(p['common_prepared'], p['common_prepare_sha256'])) and sha(path) == expected,
        'complete fixed-length/source/input/budget rebuild')
    return p


def execution_request(task, winner, independent_prefix_path):
    """Return exact real candidate prefix; never fabricate/extend short actions."""
    task = core.validate_task(task); h = winner['horizon']
    require(type(winner['slot']) is int and winner['slot'] in (0, 1) and winner['mode'] in common.MODES, 'fixed task/method census')
    require(winner['task_binding_sha256'] == digest(task['binding']) and
        winner['root_context_sha256'] == digest(task['root_context']), 'exact selected task/model/root identity, not nearby or identical-latent task')
    require(type(h) is int and h in (1, 3), 'H1/H3 joint planning only')
    raw = core.finite(winner['actions'], (5*h, 2), True)
    require(np.all(np.abs(raw) <= 1), 'original raw domain')
    require(type(independent_prefix_path) is list and len(independent_prefix_path) == 1, 'one actually replayed five-step prefix')
    prefix = deepcopy(winner['actions'][:5])
    end = core.checked_path(task, task['root_context'], prefix, independent_prefix_path)
    require(digest(end) == winner['prefix_context_sha256'], 'independent prefix equals actual full-plan first boundary')
    return dict(schema=NAME+'.execution_request', slot=winner['slot'], mode=winner['mode'],
        binding=deepcopy(task['binding']), root_context_sha256=digest(task['root_context']),
        planning_horizon_packets=h, planned_action_steps=5*h, requested_execution_steps=5,
        commands=prefix, commands_sha256=digest(prefix), full_plan_actions_sha256=digest(winner['actions']),
        nonexecuted_suffix_steps=5*h-5, predicted_execution_endpoint_sha256=digest(end),
        predicted_execution_modelQ=core.model_cost(task, end),
        physical_root_binding=None, physical_goal_binding=None, physical_result=None,
        status='prefix_model_verified_physical_blocked', **FLAGS)


def execute_panel(p, factory, ledger, store, rows, *, permit_ram=False):
    """Callable real search+prefix integration, NOT a model loader/dispatcher.

    MAIN supplies NativeProgram factory plus real Store/Ledger and handles source
    and model pre/post closure. Tests explicitly opt into RAM_FIXTURE only.
    """
    require(p['contract'] == contract() and p['profile'] == profile(p['base']['profile']) and p['queue'] == queue(p['base']) and
        ledger.caps == p['profile']['resource_caps'] and
        len(rows) == len(p['base']['jobs']) and rows == native.census(p['base']['jobs']), 'sealed contract/caps/full unstarted denominator')
    backends = {}
    for slot in (0, 1):
        backend = factory(slot); kind = getattr(backend, 'kind', None)
        require(kind == 'HOST_CUDA' or (permit_ram is True and kind == 'RAM_FIXTURE'), 'MAIN metered HOST CUDA adapter, or explicit RAM test only')
        backends[slot] = backend
    result = dict(status='pending', rows=rows, winners=[], prefixes=[dict(slot=s, mode=mode, status='not_started', request=None, error=None)
        for s in (0, 1) for mode in common.MODES], resource_receipt=None,
        equal_requested_execution_horizon=True, equal_actual_physical_execution_horizon=None,
        actual_physics_status='unknown_not_run', physical_result=None, error=None, **FLAGS)
    store.event(dict(kind='fixed_execution_denominator', queue=deepcopy(p['queue'])))
    try:
        result['winners'] = common.panel(p['base'], lambda s: backends[s], ledger, store, rows)
        for winner in result['winners']:
            t = p['base']['tasks'][winner['slot']]
            winner.update(task_binding_sha256=digest(t['binding']), root_context_sha256=digest(t['root_context']))
        for prefix in result['prefixes']:
            slot, mode = prefix['slot'], prefix['mode']; w = next(w for w in result['winners'] if (w['slot'], w['mode']) == (slot, mode))
            selected = next(row for row in rows if (row['slot'], row['mode'], row['index']) == (slot, mode, w['index']))
            prefix['status'] = 'attempted'
            ledger.scope = dict(slot=slot, mode=mode, index=w['index'], horizon=1, operation='selected_prefix_verification')
            store.event(dict(kind='selected_prefix_attempt', slot=slot, mode=mode, commands_sha256=digest(w['actions'][:5])))
            path = ledger.call('replay', lambda: backends[slot].replay(p['base']['tasks'][slot]['root_context'], w['actions'][:5], False, ledger))
            witness = dict(w, prefix_context_sha256=digest(selected['whole'][0]))
            request = execution_request(p['base']['tasks'][slot], witness, path)
            # Count the actual fixed-execution-boundary diagnostic separately.
            cost = ledger.call('metric', lambda: core.model_cost(p['base']['tasks'][slot], path[0]))
            require(cost == request['predicted_execution_modelQ'], 'same fixed-boundary modelQ diagnostic')
            prefix.update(status='model_verified_physical_blocked', request=request)
            store.event(dict(kind='selected_prefix_return', prefix=deepcopy(prefix), path=path))
        require(ledger.attempted == ledger.completed == p['profile']['resource_caps'], 'complete actual native search+prefix resource vector')
        result['status'] = 'fixed_execution_model_ready_physical_blocked'
    except BaseException as exc:
        result['status'] = 'failed_or_incomplete'; result['error'] = type(exc).__name__+': '+str(exc)
        for row in rows:
            if row['status'] == 'not_started': row['status'] = 'blocked_not_dispatched'
            elif row['status'] == 'attempted': row['status'] = 'failed'
        for prefix in result['prefixes']:
            if prefix['status'] == 'not_started': prefix['status'] = 'blocked_not_dispatched'
            elif prefix['status'] == 'attempted': prefix.update(status='failed', error=result['error'])
    result['resource_receipt'] = ledger.receipt(); return result


def audit_budget(p, receipt):
    core.audit_ledger(receipt)
    require(receipt['caps'] == receipt['attempted'] == receipt['completed'] == p['profile']['resource_caps'], 'no partial/padded/estimated receipt')
    out = []
    for b in p['base']['branches']:
        ops = [e for e in receipt['events'] if (e['scope']['slot'], e['scope']['mode']) == (b['slot'], b['mode'])]
        actual = {axis: sum(e['delta'][axis] for e in ops) for axis in core.AXES}
        prefix = [e for e in ops if e['scope'].get('operation') == 'selected_prefix_verification']
        require(sum(e['delta']['predict_API'] for e in prefix) == sum(e['delta']['predict_samples'] for e in prefix) == 1 and
            actual['predict_API'] == actual['predict_samples'] == 25, 'one useful selected prefix verification, no padding')
        total = {a: actual[a]+b['signal_resources'][a] for a in core.AXES}
        require(total['predict_API'] == total['predict_samples'] == 33 and b['signal_resources']['jacobian_API'] == 2 and
            b['signal_resources']['VJP_directions'] == 384, 'full independent paid J and exact total API/sample budget')
        out.append(dict(slot=b['slot'], mode=b['mode'], planning_horizon_packets=b['horizon'], requested_execution_steps=5,
            signal_resources=b['signal_resources'], new_model_resources=actual, total_resources=total,
            actual_physics_steps=None, actual_physics_status='unknown_not_run', backward_engine_calls=None, FLOPs=None, CUDA_ns=None))
    return out


def audit_physical_receipt(request, receipt):
    """Engineering-only future consumer boundary; cannot establish P1/release.

    Caller must first authenticate saved physical journal/recipe/root/goal via
    a MAIN-reviewed consumer. No artifact authenticity or physics run here.
    """
    require(request['schema'] == NAME+'.execution_request' and request['requested_execution_steps'] == 5 and
        request['execution_authorized'] is False and request['physical_release'] is False, 'fixed blocked request only')
    core.keys(receipt, ('request_sha256', 'commands', 'states', 'returned_steps', 'stop_reason', 'native_success', 'error'))
    require(receipt['request_sha256'] == digest(request), 'exact full execution request binding')
    n = receipt['returned_steps']; require(type(n) is int and 0 <= n <= 5, 'no execution extension/padding')
    require(receipt['commands'] == request['commands'][:n], 'only actual returned command prefix, no replacement')
    require(type(receipt['states']) is list and len(receipt['states']) == n+1, 'actual initial plus returned native states')
    for state in receipt['states']: core.finite(state, (7,))
    require(type(receipt['native_success']) is bool or receipt['native_success'] is None, 'native result unknown or unchanged boolean, never synthesized zero')
    stop = receipt['stop_reason']
    require(stop in ('completed', 'terminated', 'truncated', 'engineering_failure'), 'native disposition required')
    require((stop == 'completed' and n == 5 and receipt['error'] is None) or
        (stop in ('terminated', 'truncated') and receipt['error'] is None) or
        (stop == 'engineering_failure' and type(receipt['error']) is str and bool(receipt['error'])), 'no fabricated full-length completion or missing failure cause')
    return dict(requested_steps=5, actual_returned_steps=n, full_length=n == 5,
        eligible_for_complete_length_comparison=n == 5 and stop != 'engineering_failure',
        status='engineering_failure' if stop == 'engineering_failure' else ('full_length' if n == 5 else 'native_early_stop'),
        native_success=receipt['native_success'], stop_reason=stop, artifact_authenticity_verified=False,
        scientific_admission=False, physical_release=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan')
    a = sub.add_parser('prepare'); a.add_argument('--common-prepared', required=True); a.add_argument('--common-sha256', required=True); a.add_argument('--output', required=True)
    for name in ('verify', 'queue'):
        a = sub.add_parser(name); a.add_argument('--prepared', required=True); a.add_argument('--sha256', required=True)
    args = parser.parse_args(argv)
    if args.cmd == 'plan': result = dict(contract=contract(), model_runner_present=False, physical_consumer_present=False)
    elif args.cmd == 'prepare': result = write_new(args.output, material(args.common_prepared, args.common_sha256))
    else:
        p = verify(args.prepared, args.sha256); result = dict(status=p['status'], contract=p['contract'], queue=p['queue'], **FLAGS)
    print(canonical(result).decode())


if __name__ == '__main__': main()
