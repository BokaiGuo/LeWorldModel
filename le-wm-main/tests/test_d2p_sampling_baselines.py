import numpy as np
import pytest

from experiments.spectral_reachability.d2p_sampling_baselines import (
    mppi, random_shooting, softmax_weights,
)


def quadratic(x):
    return np.sum((x - 0.6) ** 2, axis=tuple(range(1, x.ndim)))


@pytest.mark.parametrize('solver', [mppi, random_shooting])
def test_quadratic_bounds_seed_and_complete_accounting(solver):
    seen = []

    def cost(x):
        seen.append(x.copy())
        assert np.all(x >= -1) and np.all(x <= 1)
        return quadratic(x)

    kwargs = dict(initial=np.zeros((2, 1)), lower=-1, upper=1, seed=42,
                  samples=256)
    if solver is mppi:
        kwargs.update(temperature=0.05, noise_scale=0.5, iterations=5)
    action, result = solver(cost, **kwargs)
    replay, _ = solver(quadratic, **kwargs)
    np.testing.assert_array_equal(action, replay)
    assert result['best_cost'] < 0.02
    assert result['best_cost'] == pytest.approx(quadratic(action[None])[0])
    assert result['cost_calls'] == len(seen)
    assert result['cost_evaluations'] == sum(len(x) for x in seen)
    assert result['cost_evaluations'] == (1 + 5 * 257 if solver is mppi else 257)
    assert np.all(np.diff(result['best_cost_history']) <= 0)
    other, _ = solver(quadratic, **{**kwargs, 'seed': 43})
    assert not np.array_equal(action, other)


def test_softmax_temperature_shift_and_extreme_values():
    costs = np.array([1., 2., 3.])
    cold = softmax_weights(costs, temperature=0.01)
    hot = softmax_weights(costs, temperature=100.)
    assert cold[0] > hot[0]
    np.testing.assert_allclose(cold.sum(), 1)
    np.testing.assert_allclose(softmax_weights(costs + 1000, temperature=2),
                               softmax_weights(costs, temperature=2))
    np.testing.assert_array_equal(
        softmax_weights(np.array([-1e308, 1e308]), temperature=1e-300), [1., 0.])


@pytest.mark.parametrize('solver', [mppi, random_shooting])
def test_best_ever_includes_initial_and_constant_shift(solver):
    kwargs = dict(initial=np.full((1,), .6), lower=-1, upper=1, seed=1,
                  samples=20)
    if solver is mppi:
        kwargs.update(temperature=.3, noise_scale=.8, iterations=3)
    a, ar = solver(quadratic, **kwargs)
    b, br = solver(lambda x: quadratic(x) + 100, **kwargs)
    np.testing.assert_array_equal(a, kwargs['initial'])
    np.testing.assert_allclose(a, b)
    assert br['best_cost'] == ar['best_cost'] + 100


@pytest.mark.parametrize('solver', [mppi, random_shooting])
@pytest.mark.parametrize('change', [
    {'lower': 1, 'upper': -1}, {'lower': -np.inf},
    {'initial': np.array([2.])}, {'initial': np.array([np.nan])},
    {'initial': np.array([])}, {'samples': 0}, {'samples': True},
    {'samples': 2.5}, {'seed': -1}, {'seed': True},
])
def test_invalid_inputs_fail_before_cost(solver, change):
    kwargs = dict(initial=np.zeros(1), lower=-1, upper=1, seed=1,
                  samples=10)
    if solver is mppi:
        kwargs.update(temperature=1., noise_scale=.5, iterations=2)
    kwargs.update(change)

    def forbidden(x):
        pytest.fail('invalid input reached callable')

    with pytest.raises(ValueError):
        solver(forbidden, **kwargs)


@pytest.mark.parametrize('bad', [np.nan, np.inf, np.zeros((1, 1)), 0.])
@pytest.mark.parametrize('solver', [mppi, random_shooting])
def test_invalid_cost_fails_closed(solver, bad):
    kwargs = dict(initial=np.zeros(1), lower=-1, upper=1, seed=1,
                  samples=2)
    if solver is mppi:
        kwargs.update(temperature=1., noise_scale=.5, iterations=1)
    with pytest.raises(ValueError):
        solver(lambda x: bad, **kwargs)


@pytest.mark.parametrize('temperature', [0, -1, np.nan, np.inf, True])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        softmax_weights(np.array([1., 2.]), temperature=temperature)


def test_vector_bounds_and_clipped_mppi():
    action, _ = mppi(lambda x: -x.sum(axis=1), initial=np.zeros(2),
                  lower=np.array([-1., -2.]), upper=np.array([.1, .2]),
                  seed=0, samples=64, iterations=2, temperature=.01, noise_scale=10.)
    np.testing.assert_allclose(action, [.1, .2])


@pytest.mark.parametrize('noise_std', [0, -1, np.nan, np.inf, [1, 2, 3]])
def test_invalid_noise(noise_std):
    with pytest.raises(ValueError):
        mppi(quadratic, initial=np.zeros(2), lower=-1, upper=1, seed=0,
             samples=4, iterations=1, temperature=1., noise_scale=noise_std)


def test_mppi_shift_invariance_for_nonoptimal_initial():
    batches = [[], []]

    def cost(which, shift):
        def evaluate(x):
            batches[which].append(x.copy())
            return quadratic(x) + shift
        return evaluate

    kwargs = dict(initial=np.zeros((2, 1)), lower=-1, upper=1, seed=7,
                  samples=30, iterations=4, temperature=.2, noise_scale=.4)
    a, _ = mppi(cost(0, 0), **kwargs)
    b, _ = mppi(cost(1, 64), **kwargs)
    np.testing.assert_allclose(a, b, atol=1e-12)
    for x, y in zip(*batches):
        np.testing.assert_allclose(x, y, atol=1e-12)


def test_mppi_weighted_nominal_matches_declared_contract():
    batches = []

    def cost(x):
        batches.append(x.copy())
        return quadratic(x)

    mppi(cost, initial=np.zeros(2), lower=-1, upper=1, seed=8,
         samples=7, iterations=1, temperature=.4, noise_scale=2.)
    proposals = batches[1]
    expected = softmax_weights(quadratic(proposals), temperature=.4) @ proposals
    np.testing.assert_allclose(batches[2][0], expected)


@pytest.mark.parametrize('solver', [mppi, random_shooting])
def test_later_bad_batch_aborts_instead_of_returning_incumbent(solver):
    calls = 0

    def cost(x):
        nonlocal calls
        calls += 1
        return np.zeros(len(x)) if calls == 1 else np.full(len(x), np.nan)

    kwargs = dict(initial=np.zeros(2), lower=-1, upper=1, samples=8, seed=1)
    if solver is mppi:
        kwargs.update(iterations=2, temperature=1., noise_scale=.3)
    with pytest.raises(ValueError, match='cost_fn'):
        solver(cost, **kwargs)
    assert calls == 2


@pytest.mark.parametrize('change', [
    {'iterations': 0}, {'iterations': True}, {'iterations': 1.5},
    {'temperature': 0}, {'temperature': np.nan}, {'temperature': True},
    {'lower': [0, 0, 0]}, {'upper': -1}, {'noise_scale': np.nan},
])
def test_mppi_invalid_configuration_before_callable(change):
    kwargs = dict(initial=np.zeros(2), lower=-1, upper=1, samples=4,
                  seed=1, iterations=2, temperature=1., noise_scale=.3)
    kwargs.update(change)

    def forbidden(x):
        pytest.fail('invalid configuration reached cost')

    with pytest.raises(ValueError):
        mppi(forbidden, **kwargs)


@pytest.mark.parametrize('solver', [mppi, random_shooting])
@pytest.mark.parametrize('dtype', [complex, object, bool, str])
def test_nonreal_cost_dtype_rejected_with_failed_receipt(solver, dtype):
    kwargs = dict(initial=np.zeros(1), lower=-1, upper=1, samples=3, seed=0)
    if solver is mppi:
        kwargs.update(iterations=1, temperature=1., noise_scale=.5)
    with pytest.raises(ValueError) as error:
        solver(lambda x: np.ones(len(x), dtype=dtype), **kwargs)
    receipt = error.value.receipt
    assert receipt['completed'] is False
    assert receipt['cost_calls_attempted'] == 1
    assert receipt['cost_evaluations_attempted'] == 1
    assert receipt['cost_calls'] == receipt['cost_evaluations'] == 0
    assert receipt['cost_calls_failed'] == receipt['cost_evaluations_failed'] == 1


@pytest.mark.parametrize('field', ['initial', 'lower', 'upper', 'noise_scale', 'temperature'])
@pytest.mark.parametrize('dtype', [complex, object, bool, str])
def test_nonreal_parameter_dtype_rejected_before_callable(field, dtype):
    kwargs = dict(initial=np.zeros(1), lower=-1, upper=1, samples=3, seed=0,
                  iterations=1, temperature=1., noise_scale=.5)
    # Use otherwise feasible values, so rejection proves dtype validation.
    value = np.full(1, 0 if field in ('initial', 'lower') else 1, dtype=dtype)
    kwargs[field] = value.reshape(()) if field == 'temperature' else value

    def forbidden(x):
        pytest.fail('nonreal parameter reached cost')

    with pytest.raises(ValueError):
        mppi(forbidden, **kwargs)


@pytest.mark.parametrize('dtype', [complex, object, bool, str])
def test_softmax_rejects_nonreal_cost_dtype(dtype):
    with pytest.raises(ValueError):
        softmax_weights(np.ones(2, dtype=dtype), temperature=1.)


@pytest.mark.parametrize('solver', [mppi, random_shooting])
def test_receipt_binds_full_broadcast_parameter_domain(solver):
    initial = np.zeros((2, 2))
    kwargs = dict(initial=initial, lower=[-1, -2], upper=3., samples=3, seed=0)
    if solver is mppi:
        kwargs.update(iterations=1, temperature=1., noise_scale=.5)
    _, receipt = solver(quadratic, **kwargs)
    domain = receipt['parameter_domain']
    assert domain['shape'] == [2, 2]
    assert domain['dtype'] == 'float64'
    assert domain['coordinate'] == 'caller_defined_parameters_no_action_transform'
    np.testing.assert_array_equal(domain['initial'], initial)
    np.testing.assert_array_equal(domain['lower'], [[-1, -2], [-1, -2]])
    np.testing.assert_array_equal(domain['upper'], np.full((2, 2), 3.))
    initial[:] = 1  # receipt is not a live view of input
    assert domain['initial'] == [[0., 0.], [0., 0.]]
    assert receipt['completed'] is True
    assert receipt['cost_calls_attempted'] == receipt['cost_calls']
    assert receipt['cost_evaluations_attempted'] == receipt['cost_evaluations']
    assert receipt['cost_calls_failed'] == receipt['cost_evaluations_failed'] == 0


@pytest.mark.parametrize('failure', ['nan', 'shape', 'exception', 'complex'])
def test_later_failure_counts_only_completed_batches_as_success(failure):
    calls = 0

    def cost(x):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.zeros(len(x))
        if failure == 'exception':
            raise RuntimeError('adapter failed')
        if failure == 'shape':
            return np.zeros((len(x), 1))
        return np.full(len(x), np.nan if failure == 'nan' else 1+2j)

    with pytest.raises(ValueError) as error:
        random_shooting(cost, initial=np.zeros(1), lower=-1, upper=1,
                        samples=3, seed=0)
    receipt = error.value.receipt
    assert receipt['cost_calls'] == receipt['cost_evaluations'] == 1
    assert receipt['cost_calls_attempted'] == 2
    assert receipt['cost_evaluations_attempted'] == 4
    assert receipt['cost_calls_failed'] == 1
    assert receipt['cost_evaluations_failed'] == 3
    assert receipt['completed'] is False
    assert 'action' not in receipt


def test_float64_conversion_overflow_rejected_without_warning():
    from experiments.spectral_reachability.d2p_sampling_baselines import _real_array
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        return  # no wider finite floating representation on this platform
    with pytest.raises(ValueError, match='finite'):
        _real_array(np.array([np.finfo(np.longdouble).max]), 'cost_fn')


@pytest.mark.parametrize('solver', [mppi, random_shooting])
def test_later_invalid_dtype_no_deployable_result_and_compatible_exception(solver):
    from experiments.spectral_reachability.d2p_sampling_baselines import CostEvaluationError
    calls = 0

    def cost(x):
        nonlocal calls
        calls += 1
        return np.zeros(len(x)) if calls == 1 else np.full(len(x), 1+2j)

    kwargs = dict(initial=np.zeros(1), lower=-1, upper=1, samples=5, seed=0)
    if solver is mppi:
        kwargs.update(iterations=2, noise_scale=.5, temperature=1.)
    with pytest.raises(CostEvaluationError) as error:
        solver(cost, **kwargs)
    receipt = error.value.receipt
    assert isinstance(error.value, ValueError)
    assert receipt['configuration']['seed'] == 0
    assert receipt['cost_evaluations_attempted'] == 6
    assert receipt['cost_evaluations'] == 1
    assert receipt['cost_evaluations_failed'] == 5
    assert receipt['completed'] is False


def test_real_integer_inputs_and_costs_remain_supported_json_receipt():
    import json
    _, receipt = random_shooting(lambda x: np.zeros(len(x), dtype=np.int64),
                                initial=np.zeros((1, 2), dtype=np.int64),
                                lower=-1, upper=1, samples=2, seed=0)
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt
    assert receipt['cost_calls'] == receipt['cost_calls_attempted'] == 2
    assert receipt['cost_evaluations'] == receipt['cost_evaluations_attempted'] == 3
