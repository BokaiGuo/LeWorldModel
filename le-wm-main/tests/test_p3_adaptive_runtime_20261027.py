"""Non-neural finite-response fixtures; no CUDA execution/efficacy evidence."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from experiments.spectral_reachability import p3_adaptive_runtime_20261027 as rt
from experiments.spectral_reachability.p3_adaptive_runtime_cuda_20261027 import ReplayAccounting
from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudgetExceeded


def context(x=0., y=0.):
    return dict(layout='states_with_between_state_actions_v1', context_length=2,
        latent_dim=2, action_dim=2, latent_history=[[0., 0.], [x, y]], action_history=[[0., 0.]])


def bank():
    return [np.tile([v, 0.], (5, 1)) for v in (-1., 1.)]


def replay(c, a):
    child = deepcopy(c)
    child['latent_history'] = [c['latent_history'][-1],
        (np.asarray(c['latent_history'][-1]) + a[0]).tolist()]
    child['action_history'] = [a[-1].tolist()]
    return child


def search(**updates):
    args = dict(primitive_bank=bank(), root_context=context(), replay=replay,
        goal=[4., 0.], projection=np.eye(2),
        policy=rt.RuntimePolicy('sealed', base_beam=1, max_expansions=10))
    return rt.run_search(**(args | updates))


def test_known_response_geometry_no_fitting():
    p = rt.RuntimePolicy('seal')
    s, details = rt.response_signal(context(), [context(-1), context(1)], [4, 0], np.eye(2), p)
    assert s.unreachable_fraction == 0
    assert s.horizon_gain == .25
    assert details['response_rank'] == 1
    s, _ = rt.response_signal(context(), [context(-1), context(1)], [0, 4], np.eye(2), p)
    assert s.unreachable_fraction == 1 and s.horizon_gain == 0


def test_dynamics_changes_depth_at_identical_root_goal_and_bank():
    moving = search()
    stationary = search(replay=lambda c, a: deepcopy(c))
    assert len(moving['layers']) == 3
    assert len(stationary['layers']) == 1
    assert stationary['status'] == 'dynamics_horizon_stop'
    assert moving['best_plan']['depth'] == 3


def test_off_axis_residual_changes_breadth_and_search_work():
    p = rt.RuntimePolicy('seal', base_beam=1, max_expansions=10)
    result = search(goal=[4., 8.], policy=p)
    assert result['layers'][0]['beam_width'] == 2
    fixed = search(goal=[4., 8.], policy=p, mode='fixed')
    assert fixed['layers'][0]['beam_width'] == 1
    assert result['attempted_expansions'] == 10
    assert fixed['attempted_expansions'] == 6


def test_fixed_runs_full_depth_with_zero_response_and_pays_signals():
    result = search(mode='fixed', replay=lambda c, a: deepcopy(c))
    assert len(result['layers']) == 3
    assert result['attempted_expansions'] == result['metric_calls'] == 6
    assert all(l['signals'] for l in result['layers'])


def test_signal_not_static_score_early_stop():
    # +1 and -1 have the SAME squared distance to an orthogonal goal. A
    # nonzero goal-parallel response hidden in the other coordinate permits
    # continuation, even though each child's cost is worse than its parent.
    def curved(c, a):
        child = replay(c, a)
        child['latent_history'][-1][1] += .25
        return child
    result = search(goal=[0., 1.], replay=curved)
    first = result['layers'][0]
    assert all(e['cost'] > .5 for e in first['kernels'][0]['receipt']['attempts'])
    assert first['decision']['horizon_tokens'] == 3
    assert len(result['layers']) > 1


@pytest.mark.parametrize('cap', [1, 2, 3, 4, 5])
def test_expansion_cap_and_no_partial_signal(cap):
    result = search(mode='fixed', policy=rt.RuntimePolicy('seal', base_beam=1, max_expansions=cap))
    assert result['attempted_expansions'] == cap
    assert result['status'] == 'budget_exhausted'
    assert result['layers'][-1]['decision'] is None


def test_failures_preserved_no_schedule_from_success_subset():
    def broken(c, a):
        if a[0, 0] < 0:
            raise ValueError('attempted edge failure')
        return replay(c, a)
    result = search(replay=broken)
    assert result['attempted_expansions'] == 2
    assert result['status'] == 'failed'
    assert result['layers'][0]['decision'] is None
    assert result['layers'][0]['kernels'][0]['receipt']['failed_attempts'] == 1
    assert result['best_plan'] is not None


def test_prediction_cap_failure_not_refunded_or_rescheduled():
    def blocked(c, a):
        raise PredictionBudgetExceeded('blocked')
    result = search(replay=blocked)
    assert result['status'] == 'budget_exhausted'
    assert result['attempted_expansions'] == 1
    assert result['best_plan'] is None


def test_exact_best_actions_and_context_no_padding():
    result = search()
    best = result['best_plan']
    c = context()
    for a in np.asarray(best['actions']).reshape(-1, 5, 2):
        c = replay(c, a)
    assert c == best['context']
    assert len(best['actions']) == best['depth'] * 5


def test_determinism_and_input_nonmutation():
    c, b, g, q = context(), bank(), [4., 8.], np.eye(2)
    saved = deepcopy((c, b, g, q))
    a = search(root_context=c, primitive_bank=b, goal=g, projection=q)
    z = search(root_context=c, primitive_bank=b, goal=g, projection=q)
    a.pop('signal_seconds'); z.pop('signal_seconds')
    assert a == z
    assert c == saved[0] and g == saved[2]
    np.testing.assert_array_equal(b, saved[1]); np.testing.assert_array_equal(q, saved[3])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, -1., 0., 1.])
def test_invalid_relative_cutoff(bad):
    with pytest.raises(ValueError):
        rt.RuntimePolicy('seal', svd_relative_cutoff=bad)


@pytest.mark.parametrize('field,value', [('max_depth', 1), ('max_depth', True),
    ('base_beam', 0), ('expansion_factor', False), ('max_expansions', 1.5),
    ('chi_threshold', float('nan')), ('gain_threshold', 2.)])
def test_strict_policy(field, value):
    with pytest.raises(ValueError):
        replace(rt.RuntimePolicy('seal'), **{field: value})


def test_no_extra_outcomes_in_context_or_signal_api():
    with pytest.raises(ValueError):
        search(root_context=context() | {'success': True})
    with pytest.raises(TypeError):
        rt.response_signal(context(), [context()], [1, 0], np.eye(2),
                           rt.RuntimePolicy('seal'), outcome=True)


@pytest.mark.parametrize('goal,q', [([float('nan'), 0], np.eye(2)),
    ([1], np.eye(2)), ([1, 0], np.eye(3))])
def test_invalid_metric_rejected_before_replay(goal, q):
    entered = []
    with pytest.raises(ValueError):
        search(goal=goal, projection=q, replay=lambda c, a: entered.append(1))
    assert entered == []


def test_zero_residual_and_zero_response_safe():
    result = search(goal=[0, 0], replay=lambda c, a: deepcopy(c))
    assert len(result['layers']) == 1
    assert result['layers'][0]['signals'][0]['horizon_gain'] == 0


def test_multi_token_rejected_not_misaccounted():
    with pytest.raises(ValueError):
        search(primitive_bank=[np.zeros((10, 2))])


def test_resource_observer_includes_failed_rollout_and_actual_hook_entries():
    def fail(c, a):
        raise RuntimeError('failed before predict')
    work = ReplayAccounting(fail)
    with pytest.raises(RuntimeError):
        work(context(), bank()[0])
    tensor = type('Metadata', (), {'shape': (2, 3, 10)})()
    work.action_entry(None, (tensor,))
    work.encoder_entry(None, (tensor,))
    r = work.resources(dict(phases=[dict(predict_calls=1)], attempted_predict_samples=2))
    assert r.rollout_samples == 1
    assert r.imagined_action_tokens == 2
    assert work.action_encoder_samples == 2
    assert work.processed_action_context_tokens == 6
    assert r.predict_samples == r.encoder_samples == 2
    assert r.action_backward_calls == r.physical_env_steps == 0


def test_stage_flags_always_false():
    r = search()
    assert r['p1_dependency'] == 'not_passed'
    assert not r['p3_admitted'] and not r['compute_advantage'] and not r['efficacy_claim']
