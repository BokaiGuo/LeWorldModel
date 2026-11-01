"""Saved-only P3 scheduling audit. No upstream imports or neural execution.

Three distinct mechanisms stay distinct: joint pre-search allocation, fixed-H3
full-bank breadth allocation, and causal layer runtime. No counterfactual
unvisited latent, positive-cost stopping certificate, or efficacy is invented.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

NAME = 'p3_adaptive_budget_saved_audit_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
CAP = 2 * 1024**2
FLOOR = 200 * 1024**2
DEFAULTS = {
    'joint': ('p3_joint_horizon_goal_metric_scheduler_20261027_v1_cuda_main01/report.json',
              '23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950'),
    'runtime': ('P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_report_20261027.json',
                '0b581b19caa5ccbd83bef13692a06b611550b229049f769c1105eb91b4c3bd96'),
    'fixed_h3': ('p3_action_jacobian_budget_schedule_20261027_v3_cuda_main_v1/report.json',
                 'c13dab20228fd590c3f9375e29b6be905a8db083942a547c862d97c90cf53921'),
}
ARMS = tuple(q+'__'+s for q in ('old_Q', 'goal_patch_Q') for s in ('uniform', 'dynamic'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


class Evidence:
    def __init__(self):
        self.bindings = {}

    def load(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, 'input hash: '+str(path))
        self.bindings[str(path)] = actual
        return read(path)

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        require(expected is None or actual == expected, 'source hash: '+str(path))
        self.bindings[str(path)] = actual


def check(checks, name, condition):
    checks.append(dict(name=name, ok=bool(condition)))


def close(a, b):
    return bool(np.allclose(a, b, rtol=1e-9, atol=1e-11))


def chi(j, nominal, goal, q):
    j, nominal, goal, q = map(lambda x: np.asarray(x, dtype=float), (j, nominal, goal, q))
    require(j.ndim == 2 and nominal.shape == goal.shape == (j.shape[0],) and
            q.shape == (j.shape[0], j.shape[0]) and all(np.isfinite(x).all() for x in (j, nominal, goal, q)),
            'finite aligned response arrays')
    r = q @ (goal-nominal)
    u, s, _ = np.linalg.svd(q @ j, full_matrices=False)
    rank = int(np.sum(s > s[0]*1e-6)) if len(s) and s[0] > 0 else 0
    norm = float(r @ r)
    return 0. if norm == 0 else float(np.clip(np.sum((r-u[:, :rank]@(u[:, :rank].T@r))**2)/norm, 0, 1))


def allocate(mode, c2, c3):
    require(mode in ('uniform', 'dynamic') and all(math.isfinite(x) and 0 <= x <= 1 for x in (c2, c3)), 'allocation inputs')
    gain = max(c2-c3, 0)/max(c2, 1e-12)
    b3 = 5 if mode == 'uniform' else math.floor(10*(.25+.5*gain)+.5)
    return {'2': 3*(10-b3), '3': 2*b3}


def bank(seed_index, h, count, proposal):
    rng = np.random.Generator(np.random.PCG64(20261027+100*seed_index+h))
    logits = [np.zeros(10*h)]
    for _ in range(count-1):
        direction = rng.normal(size=10*h)
        logits.append(direction/np.linalg.norm(direction)*rng.random()**(1/(10*h))*np.sqrt(5*h)*.75)
    actions = [np.tanh(u).reshape(5*h, 2) for u in logits]
    actions[1] = np.tanh(proposal).reshape(5*h, 2)
    return actions


def operation_ledger(events, kinds):
    """Start/return pairs are ONE attempt; retries are never deduplicated.

    Missing terminal or failed terminal remains charged. Identity-less repeated
    starts are preserved as new attempts and marked ambiguous, never zero work.
    """
    operations, pending, anomalies = [], {}, []
    for position, e in enumerate(events):
        if e.get('kind') not in kinds:
            continue
        identity = tuple(e.get(k) for k in ('kind', 'index', 'role', 'metric', 'horizon'))
        terminal = e.get('completed') is True or 'error' in e
        if terminal and identity in pending:
            op = operations[pending.pop(identity)]
            op.update(terminal_position=position, completed=e.get('completed') is True,
                      error=e.get('error'), terminal=e)
        elif terminal:
            anomalies.append(dict(position=position, problem='terminal_without_start'))
        else:
            if identity in pending:
                anomalies.append(dict(position=position, problem='repeated_pending_start'))
            pending[identity] = len(operations)
            operations.append(dict(start_position=position, terminal_position=None, kind=e['kind'],
                                   role=e.get('role'), completed=False, error=None, start=e))
    counts = {k: dict(attempted=sum(o['kind'] == k for o in operations),
                      completed=sum(o['kind'] == k and o['completed'] for o in operations),
                      unreturned=sum(o['kind'] == k and o['terminal_position'] is None for o in operations)) for k in kinds}
    return operations, counts, anomalies


def stop_certificate(candidates):
    """Only a zero lower-bound certificate; no outcome-chosen tolerance.

    Future costs are unknown but nonnegative. An exact zero can certify the
    stable winner ONLY when no remaining (H,index) could beat its tie key.
    This is diagnostic, not an implemented or realized early exit.
    """
    best, certificates = None, []
    for i, c in enumerate(candidates):
        if not c.get('completed'):
            continue
        key = (c['cost'], c['horizon'], c['index'])
        best = key if best is None else min(best, key)
        later = candidates[i+1:]
        if later and best[0] == 0 and all((d['horizon'], d['index']) >= best[1:] for d in later):
            certificates.append(dict(after_candidate=i, avoidable_search_predicts=sum(d['horizon'] for d in later)))
    return dict(rule='exact_zero_nonnegative_cost_and_stable_tie_bound',
                first_certificate=certificates[0] if certificates else None,
                realized_saved_predicts=0, positive_cost_certificate=False,
                note='No unseen lower bounds saved; small positive cost/stagnation is not a certificate.')


def audit_joint(report, protocol, journals):
    checks, rows, pairings = [], [], []
    records = {x['packet']['root_sha256']: x for x in protocol['records']}
    check(checks, 'unique_root_census', len(records) == len(protocol['records']) == len(report['roots']) == 4 and
          {r['root_sha256'] for r in report['roots']} == set(records))
    totals = Counter()
    for root in report['roots']:
        rid = root['root_sha256']; rec = records[rid]; events = journals[rid]
        ops, ledger, anomalies = operation_ledger(events, ('predict', 'action_encoder', 'jacobian', 'svd'))
        for kind, value in ledger.items():
            totals[kind+'_attempted'] += value['attempted']; totals[kind+'_completed'] += value['completed']
        check(checks, rid+'/journal_pairs', not anomalies and all(o['terminal_position'] is not None for o in ops))
        neural = [o['terminal'] for o in ops if o['kind'] in ('predict', 'action_encoder') and 'terminal' in o]
        # Nested module entries return in a different order than their starts.
        check(checks, rid+'/receipt_identity', sorted(neural, key=lambda e: e['index']) == root.get('neural_receipt', {}).get('events', []))
        check(checks, rid+'/all_arms', set(root['arms']) == set(ARMS))
        if not all(label in root.get('compilation', {}).get('metrics', {}) and
                   all(str(h) in root['compilation']['metrics'][label] for h in (2, 3))
                   for label in ('old_Q', 'goal_patch_Q')):
            check(checks, rid+'/complete_compilation', False)
            rows.append(dict(root_sha256=rid, status=root.get('status'), arms={k:v.get('status') for k,v in root['arms'].items()},
                journal_ledger=ledger, known_work_lower_bound=True, unknown_unsaved_work=True))
            continue
        first_search = next((i for i, e in enumerate(events) if e['kind'] == 'candidate'), len(events))
        signal_positions = [i for i, e in enumerate(events) if e['kind'] == 'response_arrays' and 'signal' in e.get('arrays', {})]
        check(checks, rid+'/signals_before_search', len(signal_positions) >= 4 and max(signal_positions) < first_search)
        check(checks, rid+'/fresh_shared_cost', all(sum(o['kind'] == k and o['role'] == 'shared' for o in ops) == 10
                                                 for k in ('predict', 'action_encoder')))
        check(checks, rid+'/jacobian_svd_census', ledger['jacobian']['attempted'] == ledger['jacobian']['completed'] == 2 and
              ledger['svd']['attempted'] == ledger['svd']['completed'] == 8)
        for arm, a in root['arms'].items():
            label, mode = arm.split('__'); q = np.asarray(rec['metrics'][label]['projection']); goal = np.asarray(rec['packet']['goal'])
            check(checks, rid+'/'+arm+'/own_source_metric', a.get('active_metric_id') == rec['metrics'][label]['active_metric_id'] and
                  a.get('source_input_sha256') == rec['packet']['input_sha256'])
            signals = {}
            for h in (2, 3):
                raw = root['compilation']['raw'][str(h)]
                signals[str(h)] = chi(raw['jacobian'], raw['nominal'], goal, q)
                check(checks, rid+'/'+arm+'/chi'+str(h), close(signals[str(h)], root['compilation']['metrics'][label][str(h)]['signal']['chi']))
            expected = allocate(mode, signals['2'], signals['3']); candidates = a.get('candidates', [])
            check(checks, rid+'/'+arm+'/allocation', expected == a.get('allocation', {}).get('candidates'))
            expected_keys = [(h, k) for h in (2, 3) for k in range(expected[str(h)])]
            keys = [(c['horizon'], c['index']) for c in candidates]
            check(checks, rid+'/'+arm+'/full_allocated_pool', keys == expected_keys and all(c.get('completed') for c in candidates))
            generated = {(h, k): actions for h in (2, 3) for k, actions in enumerate(bank(rec['seed_index'], h,
                expected[str(h)], root['compilation']['metrics'][label][str(h)]['proposal_logits']))}
            check(checks, rid+'/'+arm+'/actions_from_own_compiler', all((c['horizon'], c['index']) in generated and
                np.array_equal(c['actions'], generated[c['horizon'], c['index']]) for c in candidates))
            costs, prefix = [], []
            for c in candidates:
                if c.get('completed'):
                    z = np.asarray(c['context']['latent_history'])[-1]
                    cost = float(np.mean((q @ (z-goal))**2)); costs.append(cost)
                    check(checks, rid+'/'+arm+'/cost'+str(len(costs)), close(cost, c['cost']))
                    prefix.append(min(costs))
            search = sum(c['horizon'] for c in candidates)
            winner = a.get('winner'); verification = 2*winner['horizon'] if winner else None
            if winner and len(costs) == len(candidates):
                k = min(range(len(candidates)), key=lambda i: (costs[i], *keys[i]))
                check(checks, rid+'/'+arm+'/winner', candidates[k] == winner)
            actual = {kind: dict(attempted=sum(o['kind'] == kind and o['role'] == arm for o in ops),
                                 completed=sum(o['kind'] == kind and o['role'] == arm and o['completed'] for o in ops))
                      for kind in ('predict', 'action_encoder')}
            check(checks, rid+'/'+arm+'/budget', search == 60 and verification == a.get('verification_predicts') and
                  verification is not None and all(v == dict(attempted=search+verification, completed=search+verification) for v in actual.values()))
            rows.append(dict(root_sha256=rid, arm=arm, status=a.get('status'), chi=signals,
                allocation=expected, candidates=len(candidates), candidate_key_order=keys, prefix_best_cost=prefix,
                failed_or_partial_candidates=[dict(horizon=c['horizon'], index=c['index'], error=c.get('error'))
                    for c in candidates if not c.get('completed')],
                duplicate_actions=len(candidates)-len({hashlib.sha256(canonical(c['actions'])).hexdigest() for c in candidates}),
                search_predicts=search, verification_predicts=verification, actual=actual,
                winner_horizon=winner['horizon'] if winner else None, stopping=stop_certificate(candidates),
                causal_inputs='own Q/goal + both zero nominal/J, before candidate search; no physical input',
                feedback_rescheduling=False, early_exit_implemented=False))
        for q in ('old_Q', 'goal_patch_Q'):
            u, d = (root['arms'][q+'__'+m] for m in ('uniform', 'dynamic'))
            if not u.get('winner') or not d.get('winner'):
                pairings.append(dict(root_sha256=rid, metric=q, stratum='unmatched_or_partial'))
                continue
            pairings.append(dict(root_sha256=rid, metric=q,
                stratum='sameH' if u['winner']['horizon'] == d['winner']['horizon'] else 'mixedH',
                uniform_horizon=u['winner']['horizon'], dynamic_horizon=d['winner']['horizon'],
                same_metric_id=u['active_metric_id'] == d['active_metric_id']))
    for kind in ('predict', 'action_encoder'):
        check(checks, kind+'/reported_total', totals[kind+'_attempted'] == report['resources']['attempted'][kind])
    return dict(mechanism='root_signal_preallocation_then_exhaust_assigned_pool', checks=checks, rows=rows, pairings=pairings,
        totals=dict(totals), source_validation_svds=report['resources'].get('completed_source_validation_svds'),
        inherited_once=report['resources'].get('inherited'), backward_engine_api_calls=None,
        mathematical_vjp_directions_completed=192*totals['jacobian_completed'],
        source_causality='saved signal arrays precede first search; allocation itself has no timestamped journal event',
        failures=report.get('errors', []), budget_saving_evidence=False, partials_discarded=False,
        audit_only_numpy_svds=sum('chi' in row for row in rows)*2,
        action_context_tokens=3*totals['action_encoder_attempted'], AD_forward_predicts_already_in_shared=True)


def audit_runtime(report, wrapper):
    policy = wrapper['protocol']['schedule']; checks, rows = [], []
    check(checks, 'protocol_id', report['protocol_id'] == wrapper['protocol_id'])
    expected = {(r, m) for r in wrapper['protocol']['roots'] for m in wrapper['protocol']['methods']}
    check(checks, 'paired_root_method_census', len(report['rows']) == len(expected) and
          {(r['root_sha256'], r['method']) for r in report['rows']} == expected)
    for row in report['rows']:
        s = row.get('search'); detail = []; edges = svds = 0
        if s is None:
            rows.append(dict(root_sha256=row['root_sha256'], method=row['method'], failure=row.get('failure'), search=None)); continue
        adaptive = s['mode'] == 'adaptive'; width = policy['base_beam']; previous_retained = 1
        prior_stop = False; frontier = [[]]; seen_nodes = []
        for l in s['layers']:
            attempts = [a for k in l['kernels'] for a in k['receipt']['attempts']]
            n = len(attempts); edges += n; signals = l['signals']; svds += len(signals)
            check(checks, row['method']+'/'+str(l['depth'])+'/causal_layer', not prior_stop and
                  l['expected_expansions'] == previous_retained*16 and n == l['attempted_expansions'])
            check(checks, row['method']+'/'+str(l['depth'])+'/parent_prefix_order',
                  [k['parent_indices'] for k in l['kernels']] == frontier[:len(l['kernels'])])
            nodes = []
            for k in l['kernels']:
                attempts_here = k['receipt']['attempts']
                check(checks, row['method']+'/'+str(l['depth'])+'/primitive_order',
                      [a['primitive_indices'] for a in attempts_here] == [[i] for i in range(len(attempts_here))])
                for a in attempts_here:
                    if a.get('succeeded'):
                        nodes.append((a['cost'], l['depth'], k['parent_indices']+a['primitive_indices']))
            seen_nodes.extend(nodes)
            decision = None
            if l['layer_complete']:
                require(signals, 'complete layer without saved signals')
                gain = max(x['horizon_gain'] for x in signals); unreachable = max(x['unreachable_fraction'] for x in signals)
                decision = dict(horizon_tokens=policy['max_depth'] if gain >= policy['gain_threshold'] else 1,
                    expanded_candidates=policy['base_beam']*(policy['expansion_factor'] if unreachable >= policy['chi_threshold'] else 1))
                decision['single_pass_predict_samples'] = decision['horizon_tokens']*decision['expanded_candidates']
                check(checks, row['method']+'/'+str(l['depth'])+'/decision', decision == l['decision'] and
                      n == l['expected_expansions'] and len(signals) == len(l['kernels']))
                if adaptive:
                    width = decision['expanded_candidates']
            else:
                check(checks, row['method']+'/'+str(l['depth'])+'/no_partial_decision', l['decision'] is None)
            check(checks, row['method']+'/'+str(l['depth'])+'/beam', width == l['beam_width'])
            frontier = [x[2] for x in sorted(nodes)[:width]]
            check(checks, row['method']+'/'+str(l['depth'])+'/retained_count', len(frontier) == l['retained'])
            triggered = adaptive and decision is not None and decision['horizon_tokens'] == 1
            remaining_depths = max(policy['max_depth']-l['depth'], 0)
            detail.append(dict(depth=l['depth'], attempted_edges=n, expected_edges=l['expected_expansions'],
                complete=l['layer_complete'], retained=l['retained'], beam_width=l['beam_width'], decision=decision,
                signal_stop_triggered=triggered, remaining_depths=remaining_depths,
                effective_early_signal_stop=triggered and remaining_depths > 0,
                failures=[a for a in attempts if not a.get('succeeded')],
                signal_arrays_recomputable=False))
            prior_stop = triggered or not l['layer_complete']; previous_retained = l['retained']
        best = s.get('best_plan'); verify = best['depth'] if best else 0
        phases = row['prediction_receipt']['phases']; total = row['prediction_receipt']['attempted_predict_samples']
        check(checks, row['method']+'/phase_costs', len(phases) == 2 and
              [p['predict_samples'] for p in phases] == [edges, verify] and total == edges+verify and
              total == row['action_encoder_samples'] and row['processed_action_context_tokens'] == 3*total)
        actual_depth = max([verify]+[l['depth'] for l in s['layers'] if l['attempted_expansions']])
        if best and seen_nodes:
            b = min(seen_nodes)
            check(checks, row['method']+'/stable_best_over_visited', close(best['cost'], b[0]) and
                  best['depth'] == b[1] and best['primitive_indices'] == b[2])
        rows.append(dict(root_sha256=row['root_sha256'], method=row['method'], status=s['status'], failure=row.get('failure'),
            layers=detail, search_predicts=edges, verification_predicts=verify, predict_attempted=total,
            action_encoder_attempted=row['action_encoder_samples'], signal_svd_completed=svds,
            declared_max_depth=row['resources']['max_planning_horizon_tokens'], actual_attempted_depth=actual_depth,
            effective_early_signal_stop=any(l['effective_early_signal_stop'] for l in detail),
            note='A zero-attempt tail layer is not an executed depth; terminal-layer stop saves no next layer.'))
    return dict(mechanism='causal_layer_feedback_connected',
        checks=checks, rows=rows, predict_attempted=sum(x.get('predict_attempted', 0) for x in rows),
        action_encoder_attempted=sum(x.get('action_encoder_attempted', 0) for x in rows),
        signal_svd_completed=sum(x.get('signal_svd_completed', 0) for x in rows),
        signal_svd_validation_per_arm=1, signal_svd_validation_completed=sum(x.get('search') is not None for x in report['rows']),
        signal_svd_validation_note='root-vs-root response_signal call before search; not a layer decision',
        signal_recompute_gap='Only scalar/spectrum summaries and kernel best context saved, not every child latent; cannot independently reconstruct response SVD/chi/gain.',
        signal_semantics='uncentered child-minus-parent finite responses include autonomous drift; not action Jacobian',
        inherited_cost='prior compilation/input encoding unknown attributed cost, not zero',
        backward_AD_calls=0, visual_encoder_calls=0, budget_saving_evidence=False, partials_discarded=False)


def audit_fixed_h3(report):
    checks, rows = [], []
    for row in report['rows']:
        s = row['search']; events = s['events']; alloc = s['allocation']; mode = row['mode']
        if s.get('signal') is not None:
            c = s['signal']['chi']; k2 = 4 if mode == 'uniform' else 2+math.floor(4*c+.5)
            check(checks, row['root_sha256']+'/'+mode+'/allocation', alloc == [k2, 8-k2])
        else:
            check(checks, row['root_sha256']+'/'+mode+'/signal_available', False)
        layers = []
        for l in s['layers']:
            es = [e for e in events if e['depth'] == l['depth']]
            expected = [(p, i) for p in l['parents'] for i in range(16)]
            check(checks, row['root_sha256']+'/'+mode+'/bank'+str(l['depth']),
                  [(e['parent_path'], e['primitive_index']) for e in es] == expected and len(es) == l['attempted_edges'])
            layers.append(dict(depth=l['depth'], parents=len(l['parents']), attempted_edges=len(es),
                               failures=sum(not e['succeeded'] for e in es), full_bank_per_parent=len(es) == 16*len(l['parents'])))
        pred = row['prediction_receipt']; n = pred['attempted_predict_samples']
        check(checks, row['root_sha256']+'/'+mode+'/budget', sum(l['attempted_edges'] for l in layers) == 144 and
              [p['predict_samples'] for p in pred['phases']] == [144, 3] and n == row['action_encoder_samples'] == 147)
        rows.append(dict(root_sha256=row['root_sha256'], mode=mode, status=row['status'], failure=row.get('failure'),
            allocation=alloc, layers=layers, predict_attempted=n, verification_predicts=3,
            early_exit_implemented=False, full_horizon=3, signal_svd_calls=s['svd_calls']))
    check(checks, 'unique_pairs', len(rows) == len({(r['root_sha256'], r['mode']) for r in rows}) == 8)
    return dict(mechanism='signal_changes_retained_parent_counts_not_bank_or_horizon_or_total_budget', checks=checks,
        rows=rows, predict_attempted=sum(r['predict_attempted'] for r in rows), inherited_once=report['inherited_cost'],
        new_jacobian_materializations=report['new_jacobian_materializations'], budget_saving_evidence=False,
        action_encoder_attempted=sum(r['action_encoder_samples'] for r in report['rows']),
        signal_svd_attempted=sum(r['signal_svd_calls'] for r in rows), new_backward_AD_calls=0, visual_encoder_calls=0,
        note='Full 16-bank at each visited parent is not exhaustive 16^3 tree enumeration; breadth adapts, work stays 144+3.')


def source_inventory(evidence):
    names = {
        'p3_joint_horizon_goal_metric_scheduler_20261027_v1': ('search', 'compile_shared', 'NeuralCap'),
        'p3_horizon_residual_budget_20261027_v4': ('allocate', 'search', 'pool'),
        'p3_action_jacobian_budget_schedule_20261027_v3': ('run_search', 'inherited_cost'),
        'p3_fixed_horizon_budget_schedule_20261027_v2': ('allocate', 'run_search'),
        'p3_adaptive_runtime_20261027': ('response_signal', 'run_search'),
        'p3_adaptive_runtime_cuda_20261027': ('run_arm',),
    }
    result = []
    for name, symbols in names.items():
        path = ROOT/'experiments/spectral_reachability'/(name+'.py'); evidence.bind(path)
        tree = ast.parse(path.read_text())
        result.append(dict(path=str(path), sha256=sha(path), symbols={n.name:dict(line=n.lineno, end=n.end_lineno)
            for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in symbols}))
    return result


def run(output):
    ev = Evidence(); loaded = {}
    for name, (path, expected) in DEFAULTS.items():
        loaded[name] = ev.load(ROOT/'results'/path, expected)
    joint_path = ROOT/'results'/DEFAULTS['joint'][0]
    execution = ev.load(joint_path.parent/'execution_protocol.json')
    prepared = Path(execution['prepared']); pr = ev.load(prepared/'prepare.json', execution['prepare_sha256'])
    protocol = ev.load(prepared/'protocol.json', pr['protocol_sha256'])
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == pr['protocol_id'] == loaded['joint']['protocol_id'], 'joint protocol content identity')
    for path, expected in protocol['sources_sha256'].items():
        ev.bind(path, expected)
    # Binding all original model/input files is explicit; no model is imported.
    for path, expected in {**protocol['bindings'], **protocol['model_bindings']}.items():
        ev.bind(path, expected)
    journals = {}
    for root in loaded['joint']['roots']:
        path = Path(root['journal_path']); ev.bind(path, root['journal_sha256'])
        journals[root['root_sha256']] = [json.loads(line) for line in path.read_text().splitlines()]
    runtime_path = ROOT/'results'/DEFAULTS['runtime'][0]
    runtime_protocol = ev.load(runtime_path.parent/'P3_ADAPTIVE_RUNTIME_protocol_20261027.json')
    require(hashlib.sha256(json.dumps(runtime_protocol['protocol'], sort_keys=True).encode()).hexdigest() ==
            runtime_protocol['protocol_id'] == loaded['runtime']['protocol_id'], 'runtime protocol content identity')
    for path, expected in runtime_protocol['protocol']['sources'].items():
        ev.bind(path, expected)
    for row in loaded['fixed_h3']['rows']:
        ev.bind(row['journal_path'], row['journal_sha256'])
    v3dir = ROOT/'results/p3_action_jacobian_budget_schedule_20261027_v3_prepared'
    v3execution = ev.load((ROOT/'results'/DEFAULTS['fixed_h3'][0]).parent/'execution_protocol.json')
    v3prepare = ev.load(v3dir/'prepare.json', v3execution['prepare_sha256'])
    v3 = ev.load(v3dir/'protocol.json', v3prepare['protocol_sha256'])
    require(hashlib.sha256(json.dumps(v3['protocol'], sort_keys=True, allow_nan=False).encode()).hexdigest() ==
            v3['protocol_id'] == loaded['fixed_h3']['protocol_id'], 'fixed H3 protocol content identity')
    inventory = source_inventory(ev)
    result = dict(schema=NAME+'.report', joint=audit_joint(loaded['joint'], protocol, journals),
        runtime=audit_runtime(loaded['runtime'], runtime_protocol), fixed_h3=audit_fixed_h3(loaded['fixed_h3']),
        source_inventory=inventory, source_report_statuses={k:v['status'] for k,v in loaded.items()},
        scientific_gaps=[
            'Joint/v4: root-signal preallocation then fixed pools; no candidate-feedback reschedule or early-exit path.',
            'FixedH3 v3: causal breadth allocation exists, but all 16 children per parent and fixed144+3 work; inherited J cost unknown.',
            'Runtime: causal breadth feedback exists; this saved run has no preterminal signal stop; budget exhaustion is not signal benefit.',
            'Runtime raw all-child response arrays absent; threshold decision reproducible, response geometry not independently reconstructed.',
            'No counterfactual unsampled response, held-out root benefit, physical quality gain, or equal total FLOPs is established.',
        ], new_neural_calls=0, new_physical_steps=0, P3_admission=False, efficacy_claim=False, compute_advantage=False)
    result['all_checks_passed'] = all(c['ok'] for k in ('joint', 'runtime', 'fixed_h3') for c in result[k]['checks'])
    result['status'] = 'saved_audit_completed' if result['all_checks_passed'] else 'saved_audit_has_discrepancies'
    for path in (Path(__file__), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')):
        ev.bind(path)
    result['bindings_sha256'] = ev.bindings
    # Close read-only TOCTOU window before emitting exclusive small report.
    for path, expected in ev.bindings.items():
        require(sha(path) == expected, 'input changed during audit: '+path)
    raw = canonical(result); require(len(raw) <= CAP, '2MiB output cap')
    output = Path(output).absolute()
    require(not output.exists(), 'exclusive new output')
    require(shutil.disk_usage(output.parent).free >= FLOOR+CAP, '200MiB floor plus full2MiB reserve')
    output.mkdir()
    with (output/'report.json').open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return dict(status=result['status'], all_checks_passed=result['all_checks_passed'], report_sha256=sha(output/'report.json'),
                output_bytes=len(raw), report=str(output/'report.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), sort_keys=True))
