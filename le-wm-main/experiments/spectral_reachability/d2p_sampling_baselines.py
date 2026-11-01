"""Pure NumPy sampling kernels; not an official MPPI reproduction.

The callable consumes [batch, *initial.shape] float64 bounded parameters and
returns one finite scalar cost per action. It must be deterministic and include
any desired trajectory/control cost itself. No neural/runtime dependency.

MPPI here is a deliberately explicit heuristic: fixed diagonal Gaussian noise,
clipped proposals, exp(-cost/temperature) weights, weighted effective (clipped)
perturbations. There is NO proposal-density importance correction, NO automatic
control-energy/cross term, and NO claim of exact path-integral control under
clipping. The initial incumbent is scored once; every iteration scores all N
proposals and its updated nominal separately. All visited actions are eligible
for best-ever output, not just the final nominal. Random shooting samples N
uniform actions once and also scores the supplied initial incumbent.

Only real integer/unsigned/floating dtypes are accepted before conversion;
complex, object, boolean and string arrays are rejected. No physical action
transform is applied here: bounds describe the caller's parameter coordinates.
cost_calls/cost_evaluations count validated successful batches/samples. Attempted
and failed counters are separate. CostEvaluationError.receipt exposes failure
accounting without returning a deployable action; its completed flag is false.
"""
from __future__ import annotations

import numpy as np


def _real_array(value, name):
    """Validate the source dtype before any potentially lossy float64 cast."""
    try:
        source = np.asarray(value)
        if source.dtype.kind not in 'iuf':
            raise ValueError(f'{name} must have a real numeric dtype (not bool/object/complex)')
        with np.errstate(over='ignore', invalid='ignore'):
            converted = np.array(source, dtype=np.float64, copy=True)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f'invalid real numeric {name}') from exc
    if not np.isfinite(converted).all():
        raise ValueError(f'{name} must be finite after float64 conversion')
    return converted


class CostEvaluationError(ValueError):
    """Cost callable failure/invalid output, with non-deployable accounting."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def _positive(value, name):
    array = _real_array(value, name)
    if array.ndim != 0:
        raise ValueError(f'{name} must be a finite positive scalar')
    value = float(array)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def softmax_weights(costs, *, temperature):
    """Shift-invariant stable cost softmax; finite extremes may underflow to 0."""
    temperature = _positive(temperature, 'temperature')
    costs = _real_array(costs, 'costs')
    if costs.ndim != 1 or not costs.size or not np.isfinite(costs).all():
        raise ValueError('costs must be a nonempty finite vector')
    # Overflow in a positive cost difference means zero weight, not bad input.
    with np.errstate(over='ignore', under='ignore'):
        weights = np.exp(-(costs - costs.min()) / temperature)
    return weights / weights.sum()


def _inputs(initial, lower, upper, samples, seed):
    initial = _real_array(initial, 'initial')
    if initial.ndim < 1 or not initial.size or not np.isfinite(initial).all():
        raise ValueError('initial must be a nonempty finite action array')
    try:
        lower = np.broadcast_to(_real_array(lower, 'lower'), initial.shape).copy()
        upper = np.broadcast_to(_real_array(upper, 'upper'), initial.shape).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError('bounds must broadcast to initial.shape') from exc
    with np.errstate(over='ignore'):
        span = upper - lower
    if (not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or not np.isfinite(span).all() or np.any(span <= 0)
            or np.any(initial < lower) or np.any(initial > upper)):
        raise ValueError('finite strict bounds and feasible initial required')
    samples = _integer(samples, 'samples')
    seed = _integer(seed, 'seed', 0)
    return initial, lower, upper, samples, seed


class _Evaluator:
    def __init__(self, cost_fn, initial, *, lower, upper, configuration, algorithm, contract):
        if not callable(cost_fn):
            raise ValueError('cost_fn must be callable')
        self.cost_fn = cost_fn
        self.best = initial.copy()
        self.best_cost = None
        self.calls = self.evaluations = 0
        self.attempted_calls = self.attempted_evaluations = 0
        self.failed_calls = self.failed_evaluations = 0
        self.configuration = configuration
        self.domain = {'shape': list(initial.shape), 'dtype': 'float64',
                       'coordinate': 'caller_defined_parameters_no_action_transform',
                       'initial': initial.tolist(), 'lower': lower.tolist(),
                       'upper': upper.tolist()}
        self.history = []
        self.algorithm, self.contract = algorithm, contract

    def score(self, batch):
        if not np.isfinite(batch).all():
            raise ValueError('nonfinite generated actions')
        # Isolate the optimizer from adapter mutation; never pass a live state.
        argument = batch.copy()
        argument.flags.writeable = False
        self.attempted_calls += 1
        self.attempted_evaluations += len(batch)
        try:
            costs = _real_array(self.cost_fn(argument), 'cost_fn output')
            if costs.shape != (len(batch),):
                raise ValueError('cost_fn must return a finite vector of shape (batch,)')
        except Exception as exc:
            self.failed_calls += 1
            self.failed_evaluations += len(batch)
            receipt = self._receipt(completed=False)
            receipt['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise CostEvaluationError(f'cost_fn failed: {exc}', receipt) from exc
        self.calls += 1
        self.evaluations += len(batch)
        index = int(np.argmin(costs))
        if self.best_cost is None or costs[index] < self.best_cost:
            self.best, self.best_cost = batch[index].copy(), float(costs[index])
        return costs

    def _receipt(self, *, completed):
        return {
            'schema': 'd2p_sampling_baseline_receipt.v2', 'completed': completed,
            'algorithm': self.algorithm, 'best_cost': self.best_cost,
            'best_cost_history': self.history.copy(),
            'cost_calls': self.calls, 'cost_evaluations': self.evaluations,
            'cost_calls_attempted': self.attempted_calls,
            'cost_evaluations_attempted': self.attempted_evaluations,
            'cost_calls_failed': self.failed_calls,
            'cost_evaluations_failed': self.failed_evaluations,
            'counting_unit': 'candidate_cost_batches_and_samples_not_model_predict_work',
            'parameter_domain': self.domain,
            'contract': self.contract, 'configuration': self.configuration,
            'official_reproduction': False,
        }

    def result(self):
        return self.best.copy(), self._receipt(completed=True)


def random_shooting(cost_fn, *, initial, lower, upper, samples, seed):
    """Best of initial plus N independent uniform box samples; N+1 costs."""
    initial, lower, upper, samples, seed = _inputs(initial, lower, upper, samples, seed)
    evaluator = _Evaluator(cost_fn, initial, algorithm='uniform_random_shooting',
                           lower=lower, upper=upper, configuration={'samples': samples, 'seed': seed},
                           contract={'proposal': 'uniform_box', 'initial_scored': True,
                                     'selection': 'best_ever', 'control_cost': 'caller_only'})
    evaluator.score(initial[None])
    evaluator.history.append(evaluator.best_cost)
    rng = np.random.default_rng(seed)
    bank = rng.uniform(lower, upper, size=(samples, *initial.shape))
    evaluator.score(bank)
    evaluator.history.append(evaluator.best_cost)
    return evaluator.result()


def mppi(cost_fn, *, initial, lower, upper, samples, iterations,
         temperature, noise_scale, seed):
    """Clipped Gaussian cost-weighted update; see module importance contract."""
    initial, lower, upper, samples, seed = _inputs(initial, lower, upper, samples, seed)
    iterations = _integer(iterations, 'iterations')
    temperature = _positive(temperature, 'temperature')
    try:
        scale = np.broadcast_to(_real_array(noise_scale, 'noise_scale'), initial.shape).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError('noise_scale must broadcast to initial.shape') from exc
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('noise_scale must be finite and positive')
    evaluator = _Evaluator(cost_fn, initial, algorithm='clipped_gaussian_mppi_heuristic',
        lower=lower, upper=upper,
        configuration={'samples': samples, 'iterations': iterations,
                       'temperature': temperature, 'noise_scale': scale.tolist(), 'seed': seed},
        contract={'proposal': 'fixed_diagonal_gaussian_then_clip',
                  'importance_correction': 'none', 'control_cost': 'caller_only',
                  'update': 'cost_softmax_weighted_clipped_actions',
                  'selection': 'best_ever_including_initial_proposals_and_updated_nominals'})
    evaluator.score(initial[None])
    evaluator.history.append(evaluator.best_cost)
    nominal = initial.copy()
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        with np.errstate(over='ignore', invalid='ignore'):
            perturbation = rng.normal(size=(samples, *initial.shape)) * scale
            unbounded = nominal + perturbation
        if not np.isfinite(unbounded).all():
            raise ValueError('Gaussian proposal overflow')
        bank = np.clip(unbounded, lower, upper)
        costs = evaluator.score(bank)
        weights = softmax_weights(costs, temperature=temperature)
        nominal = np.clip(np.tensordot(weights, bank, axes=(0, 0)), lower, upper)
        evaluator.score(nominal[None])
        evaluator.history.append(evaluator.best_cost)
    return evaluator.result()
