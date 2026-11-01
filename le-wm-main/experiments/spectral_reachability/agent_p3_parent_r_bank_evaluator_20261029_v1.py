"""Injected RAM-only R8 evaluator. No model loader or production dispatch."""
from __future__ import annotations

from copy import deepcopy
import argparse
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as original

NAME = 'agent_p3_parent_r_bank_evaluator_20261029_v1'
SOLVER_SHA256 = 'f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a'
require, digest = core.require, core.digest
AXES = core.AXES + ('AD_primal_calls', 'J_elements')
FLAGS = dict(production_capable=False, physical_release=False, P1_admission=False,
             full_policy_implemented=False, full_C3_demonstrated=False,
             bank_certified_by_kernel=False)


def vector(value):
    core.keys(value, AXES)
    for v in value.values():
        core.integer(v)
    return deepcopy(value)


def zeros():
    return dict.fromkeys(AXES, 0)


def add(*values):
    return {k: sum(v[k] for v in values) for k in AXES}


def tariff(operation, h=1):
    """Normative RAM callback accounting, NOT measured neural kernel counts."""
    require(operation in ('edge', 'signal', 'suffix', 'solve'), 'known operation')
    require(type(h) is int and h in (1, 3), 'H1 or H3')
    v = zeros()
    if operation == 'solve':
        v.update(array_SVD_calls=1, solve_calls=1)
        return v
    n = 2 if operation == 'signal' else h if operation == 'edge' else 1
    for k in ('predict_API', 'predict_samples', 'action_encoder_calls',
              'action_encoder_samples', 'predictor_calls', 'predictor_samples', 'pred_proj_calls'):
        v[k] = n
    for k in ('predict_context_tokens', 'action_encoder_context_tokens',
              'predictor_context_tokens', 'pred_proj_rows'):
        v[k] = 3*n
    if operation == 'signal':
        v.update(signal_compile_calls=1, jacobian_API=1, VJP_directions=192,
                 grad_API=192, raw_backward_callbacks=192, AD_primal_calls=1, J_elements=1920)
    else:
        v['replay_calls'] = 1
        if operation == 'edge':
            v.update(program_compile_calls=1, metric_calls=1)
        else:
            v['metric_calls'] = 1
    return v


def minimum(h):
    """Nine full evaluations: eight proposals AND a separately paid zero."""
    per = add(tariff('edge', h), tariff('signal'), tariff('solve'), tariff('suffix'))
    return {k: 9*per[k] for k in AXES}


def contract():
    return dict(schema=NAME+'.contract', candidates=8, controls=1, suffix_horizon=1,
        edge_horizons=[1, 3], suffix_is_executable_edge=False,
        projection='identity192', score='original bounded_ridge.own_reach_value',
        damping=original.DAMPING, radius=original.CAP,
        solver_source_sha256=SOLVER_SHA256,
        suffix_chart='10 independent logits; raw=tanh(params).reshape(5,2), cast once float32',
        jacobian_chart='d terminal192/d raw10 at zero; tanh derivative at zero is identity',
        eligibility='immediate > zero.immediate AND reach < zero.reach; no epsilon',
        selection='minimum (reach,index) among eligible; no fallback; no copied winner',
        stop_on_first_callback_or_validation_error=True, refunds=False,
        budget_semantics='RAM normative attempted/validated-returned, not measured native census',
        unknown_resources=dict(FLOPs=None, cuda_ns=None, backward_engine_calls=None), **FLAGS)


def validate_inputs(task, parent, prefix_actions, prefix_path, bank, zero, h, authority):
    require(core.sha(original.__file__) == SOLVER_SHA256, 'original solver source unchanged')
    task = core.validate_task(deepcopy(task))
    require(task['root_context']['latent_dim'] == 192 and
            np.array_equal(np.asarray(task['projection']), np.eye(192)), 'identity192 task Q')
    parent = core.context(deepcopy(parent))
    core.integer(h, 1)
    require(h in (1, 3), 'H1 or H3 exact edge')
    if prefix_actions:
        actual = core.checked_path(task, task['root_context'], prefix_actions, prefix_path)
    else:
        require(prefix_path == [], 'empty prefix path')
        actual = task['root_context']
    require(core.canonical(actual) == core.canonical(parent), 'whole root-to-parent identity')
    core.finite(bank, (8, 5*h, 2), True)
    z = core.finite(zero, (5*h, 2), True)
    require(np.all(np.abs(bank) <= np.float32(1)) and np.all(z == 0), 'legal raw bank and explicit zero')
    core.keys(authority, ('source_sha256', 'bank_sha256', 'zero_sha256', 'root_task_sha256',
        'parent_context_sha256', 'goal_sha256', 'Q_sha256', 'normalizer_sha256',
        'action_domain_sha256', 'bank_authenticated', 'authentication_sha256'))
    for k, v in authority.items():
        if k != 'bank_authenticated':
            if k == 'authentication_sha256' and v is None:
                continue
            core.hash_value(v)
    require(type(authority['bank_authenticated']) is bool, 'literal bank authentication status')
    require(not authority['bank_authenticated'] or authority['authentication_sha256'] is not None,
            'claimed authentication needs explicit authority pointer')
    checks = dict(bank_sha256=digest(bank), zero_sha256=digest(zero), root_task_sha256=digest(task),
        parent_context_sha256=digest(parent), goal_sha256=digest(task['goal']),
        Q_sha256=digest(task['projection']), normalizer_sha256=digest([task['action_mean'], task['action_std']]),
        action_domain_sha256=task['binding']['action_domain_sha256'])
    require(all(authority[k] == v for k, v in checks.items()), 'explicit bank/task/Q/normalizer bindings')
    return task, parent


def evaluate(*, task, parent, prefix_actions, prefix_path, bank, zero, horizon,
             authority, callback, caps, used, verification_reserve, sink):
    """callback(request)->{request_sha256,resources,payload}; sink(event).

    Each callback receives a deep copy, full task/root/parent identity and its
    exact input. Only kind=RAM_FIXTURE is accepted. Production requires a future
    separate bank authority, native meter and orchestrator; no flag opens it.
    Validation errors before dispatch raise. Dispatch/response failures preserve
    attempted charges and partial rows in the returned object.
    """
    require(getattr(callback, 'kind', None) == 'RAM_FIXTURE', 'RAM fixtures only; production forbidden')
    require(callable(callback) and callable(sink), 'explicit callback and event sink')
    task, parent = validate_inputs(task, parent, prefix_actions, prefix_path, bank, zero, horizon, authority)
    caps, used, reserve = map(vector, (caps, used, verification_reserve))
    require(all(used[k] <= caps[k] for k in AXES), 'existing usage within caps')
    need = minimum(horizon)
    r = dict(schema=NAME+'.evaluation', rule=contract(), task=deepcopy(task), parent=deepcopy(parent),
        prefix_actions=deepcopy(prefix_actions), prefix_path=deepcopy(prefix_path),
        bank=deepcopy(bank), zero=deepcopy(zero), authority=deepcopy(authority), horizon=horizon,
        denominator=dict(candidates=8, controls=1), rows=[dict(index=i, status='not_requested', eligible=None)
        for i in range(8)], control=dict(index='zero', status='not_requested'), events=[],
        caps=caps, used_before=used, verification_reserve=reserve, minimum=need,
        attempted=zeros(), completed=zeros(), status='not_started', error=None,
        used_after=deepcopy(used), wall_ns=None, eligible_indices=[], selected_index=None, **FLAGS)
    start = time.monotonic_ns()

    def event(value):
        value = dict(value, seq=len(r['events']), elapsed_ns=time.monotonic_ns()-start)
        r['events'].append(deepcopy(value))
        sink(deepcopy(value))

    if any(used[k]+need[k]+reserve[k] > caps[k] for k in AXES):
        r['status'] = 'resource_cap_before_bank'
        event(dict(kind='atomic_refusal', minimum=need, verification_reserve=reserve))
        r['wall_ns'] = time.monotonic_ns()-start
        return r
    event(dict(kind='atomic_admitted', minimum=need, verification_reserve=reserve))

    def call(operation, index, context, inputs, validate):
        cost = tariff(operation, horizon)
        req = dict(schema=NAME+'.request', operation=operation, index=index,
            root_task_sha256=digest(task), parent_context_sha256=digest(parent),
            input_context_sha256=digest(context), context=deepcopy(context), task=deepcopy(task),
            authority=deepcopy(authority), inputs=deepcopy(inputs), resources=cost)
        r['attempted'] = add(r['attempted'], cost)
        event(dict(kind='dispatch', request=req))
        try:
            if operation == 'solve':
                response = dict(request_sha256=digest(req), resources=cost,
                    payload=original.bounded_ridge(inputs['jacobian'], inputs['residual']))
            else:
                response = deepcopy(callback(deepcopy(req)))
            event(dict(kind='returned', request_sha256=digest(req), response=deepcopy(response)))
            core.keys(response, ('request_sha256', 'resources', 'payload'))
            require(response['request_sha256'] == digest(req) and vector(response['resources']) == cost,
                    'exact callback identity and full tariff; no hidden/refunded resources')
            value = validate(response['payload'])
            r['completed'] = add(r['completed'], cost)
            event(dict(kind='validated', request_sha256=digest(req)))
            return value
        except Exception as exc:
            event(dict(kind='failed', request_sha256=digest(req), error=type(exc).__name__+': '+str(exc)))
            raise

    def rolled(context, actions, value):
        core.keys(value, ('path',))
        end = core.checked_path(task, context, actions, value['path'])
        return dict(path=deepcopy(value['path']), context=end)

    current = None
    try:
        # Zero is a real, separately charged evaluation, never a fabricated baseline.
        for index, actions in [('zero', zero)] + list(enumerate(bank)):
            current = r['control'] if index == 'zero' else r['rows'][index]
            current['status'] = 'running'
            current['edge_actions'] = deepcopy(actions)
            edge = call('edge', index, parent, dict(actions=actions),
                lambda v: rolled(parent, actions, v))
            current['edge'] = edge
            post = edge['context']
            immediate = float(np.mean((np.asarray(task['goal'])-np.asarray(post['latent_history'][-1]))**2))
            require(np.isfinite(immediate), 'finite immediate cost')
            current['immediate'] = immediate

            def checked_signal(v):
                core.keys(v, ('nominal_path', 'jacobian', 'input_context_sha256', 'jacobian_chart'))
                require(v['input_context_sha256'] == digest(post) and
                        v['jacobian_chart'] == contract()['jacobian_chart'], 'postcontext J identity/chart')
                nominal = core.checked_path(task, post, [[0., 0.]]*5, v['nominal_path'])
                core.finite(v['jacobian'], (192, 10), True)
                return dict(v, nominal=deepcopy(nominal['latent_history'][-1]))

            signal = call('signal', index, post, dict(horizon=1, raw_zero=[[0., 0.]]*5), checked_signal)
            current['signal'] = signal
            residual = (np.asarray(task['goal'])-np.asarray(signal['nominal'])).tolist()
            solved = call('solve', index, post, dict(jacobian=signal['jacobian'], residual=residual), lambda v: v)
            require(np.isfinite(solved['own_reach_value']), 'finite bounded reach value')
            current['solve'] = solved
            suffix = np.tanh(np.asarray(solved['params'], np.float64)).astype(np.float32).reshape(5, 2).tolist()
            current['suffix_actions'] = suffix
            current['suffix'] = call('suffix', index, post, dict(actions=suffix, lookahead_only=True),
                lambda v: rolled(post, suffix, v))
            current['reach'] = solved['own_reach_value']
            current['bounded_suffix_cost'] = float(np.mean((np.asarray(task['goal'])-
                np.asarray(current['suffix']['context']['latent_history'][-1]))**2))
            require(np.isfinite(current['bounded_suffix_cost']), 'finite bounded suffix cost')
            current['status'] = 'complete'
        baseline = r['control']
        for row in r['rows']:
            row['eligible'] = row['immediate'] > baseline['immediate'] and row['reach'] < baseline['reach']
        r['eligible_indices'] = [v['index'] for v in r['rows'] if v['eligible']]
        if r['eligible_indices']:
            r['selected_index'] = min(r['eligible_indices'], key=lambda i: (r['rows'][i]['reach'], i))
            r['status'] = 'witness_found'
        else:
            r['status'] = 'no_witness'
        require(r['attempted'] == need == r['completed'], 'complete nonrefundable bank census')
    except Exception as exc:
        if current is not None:
            current['status'] = 'failed'
        r.update(status='partial_failure', error=type(exc).__name__+': '+str(exc))
    r['used_after'] = add(used, r['attempted'])
    r['wall_ns'] = time.monotonic_ns()-start
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', action='store_true', help='print model-free ABI; no production entry')
    args = parser.parse_args()
    if args.contract:
        print(core.canonical(contract()).decode())


if __name__ == '__main__':
    main()
