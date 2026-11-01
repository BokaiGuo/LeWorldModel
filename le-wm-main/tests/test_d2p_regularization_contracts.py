"""Non-neural NumPy contract evidence, not P1 efficacy or neural fidelity.

H3/H5 solve analytic fixtures, not the live solvers. H7 executes the actual
helper function bodies extracted by AST, with NumPy-only tensor operations and
a recording arithmetic predictor. No torch import, checkpoints or simulator.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability"


@pytest.mark.parametrize("ridge", [.01, .3, 1.])
def test_h3_absolute_and_delta_normal_equations_against_augmented_lstsq(ridge):
    # Deliberately rectangular/non-symmetric; nonzero nominal distinguishes
    # J.T J n from (J.T J + lambda I)n in an absolute-endpoint QP.
    jac = np.array([[1., 2.], [-3., .5], [.2, 1.1]])
    nominal = np.array([.4, -.2])
    residual = np.array([.6, -.1, .8])  # goal - F(nominal)
    gram = jac.T @ jac + ridge * np.eye(2)
    absolute_delta = np.linalg.solve(gram, jac.T @ residual - ridge * nominal)
    delta = np.linalg.solve(gram, jac.T @ residual)
    augmented = np.vstack((jac, np.sqrt(ridge) * np.eye(2)))
    absolute_ref = np.linalg.lstsq(
        augmented, np.r_[residual, -np.sqrt(ridge) * nominal], rcond=None
    )[0]
    delta_ref = np.linalg.lstsq(augmented, np.r_[residual, np.zeros(2)], rcond=None)[0]
    np.testing.assert_allclose(absolute_delta, absolute_ref, atol=1e-14)
    np.testing.assert_allclose(delta, delta_ref, atol=1e-14)
    np.testing.assert_allclose(
        nominal + absolute_delta,
        np.linalg.solve(gram, jac.T @ residual + jac.T @ jac @ nominal),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        nominal + delta, np.linalg.solve(gram, jac.T @ residual + gram @ nominal),
        atol=1e-14,
    )
    assert np.linalg.norm(absolute_delta - delta) > 1e-4
    # Both stationary points solve their own objective, not the other's.
    np.testing.assert_allclose(
        jac.T @ (jac @ absolute_delta - residual) + ridge * (nominal + absolute_delta),
        0., atol=1e-14,
    )
    np.testing.assert_allclose(jac.T @ (jac @ delta - residual) + ridge * delta, 0., atol=1e-14)


def test_h3_scalar_counterexample_and_zero_nominal_boundary():
    # F(p)=p, g=1, n=.5, lambda=1: absolute p=.5, delta p=.75.
    n, residual, ridge = .5, .5, 1.
    absolute = n + (residual - ridge * n) / (1 + ridge)
    delta = n + residual / (1 + ridge)
    assert absolute == .5
    assert delta == .75
    absolute_cost = lambda p: (p - 1)**2 + ridge * p**2
    delta_cost = lambda p: (p - 1)**2 + ridge * (p - n)**2
    assert absolute_cost(absolute) < absolute_cost(delta)
    assert delta_cost(delta) < delta_cost(absolute)
    # At n=0 the two normal equations coincide: a zero-start-only test
    # cannot detect a change of regularization center.
    assert (1. - ridge * 0.) / (1 + ridge) == 1. / (1 + ridge)


def _bounded_polynomial_minimum(curvature, ridge):
    """Exact candidate enumeration for (u+c*u²-1)² + ridge*u², u in [0,1]."""
    from numpy.polynomial import Polynomial

    residual = Polynomial([-1., 1., curvature])
    objective = residual * residual + Polynomial([0., 0., ridge])
    roots = objective.deriv().roots()
    candidates = [0., 1.] + [float(r.real) for r in roots if abs(r.imag) < 1e-12 and 0 <= r.real <= 1]
    action = min(candidates, key=objective)
    return action, float(objective(action))


@pytest.mark.parametrize("ridge", [0., .5])
def test_h5_curvature_gap_exists_without_changing_regularization(ridge):
    # Same F(0)=0 and J(0)=1 for c=0 and c=1. Thus identical local V and
    # local actions, even though their nonlinear endpoint objectives differ.
    linear_action = 1. / (1 + ridge)
    local_value = (linear_action - 1)**2 + ridge * linear_action**2
    affine_action, affine_value = _bounded_polynomial_minimum(0., ridge)
    curved_action, curved_value = _bounded_polynomial_minimum(1., ridge)
    assert affine_action == pytest.approx(linear_action)
    assert affine_value == pytest.approx(local_value)
    actual_at_local = (linear_action + linear_action**2 - 1)**2 + ridge * linear_action**2
    assert actual_at_local > curved_value + 1e-3
    assert abs(curved_action - linear_action) > .05
    if ridge == 0:
        assert curved_action == pytest.approx((np.sqrt(5) - 1) / 2)
        assert curved_value == pytest.approx(0., abs=1e-14)


def test_h5_regularization_gap_exists_without_curvature():
    unregularized, _ = _bounded_polynomial_minimum(0., 0.)
    regularized, _ = _bounded_polynomial_minimum(0., .5)
    assert unregularized == pytest.approx(1.)
    assert regularized == pytest.approx(2. / 3.)
    # An MSE-only search and an absolute-ridge V disagree even for affine F.
    assert (unregularized - 1)**2 < (regularized - 1)**2
    assert (regularized - 1)**2 + .5 * regularized**2 < .5


class _Array(np.ndarray):
    """Only the small tensor surface needed by the real recurrence helpers."""

    @property
    def device(self):
        return "numpy-fixture"

    def expand(self, *shape):
        return np.broadcast_to(self, tuple(old if new == -1 else new for old, new in zip(self.shape, shape))).view(_Array)


def _arr(value):
    return np.asarray(value, dtype=float).view(_Array)


def _helper(filename, name, namespace):
    # Execute exactly the production function body, without importing its
    # module (which would import torch/Gym). Annotations remain unevaluated.
    tree = ast.parse((SOURCE / filename).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE / filename), "exec"), namespace)
    return namespace[name]


class _TracePredictor:
    """Not a neural module: identity action encoder and frame-label +1."""

    def __init__(self):
        self.calls = []

    def action_encoder(self, actions):
        return actions

    def predict(self, frames, actions):
        self.calls.append((np.array(frames), np.array(actions)))
        return _arr(frames + 1.)


def _assert_calls(model):
    assert len(model.calls) == 3
    for i, (frames, actions) in enumerate(model.calls):
        np.testing.assert_array_equal(frames[0, :, 0], np.arange(101 + i, 104 + i))
        expected = np.array([22., 33., 44., 55., 66.])[i:i + 3]
        np.testing.assert_array_equal(actions[0], np.repeat(expected[:, None], 10, axis=1))


def _history():
    frames = _arr([[[101.], [102.], [103.]]])
    actions = _arr(np.repeat(np.array([11., 22., 33.])[None, :, None], 10, axis=2))
    future = _arr(np.repeat(np.array([44., 55., 66.])[None, :, None], 10, axis=2))
    return frames, actions, future


def test_h7_actual_reacher_normalized_helper_nonzero_unique_history():
    namespace = {"torch": SimpleNamespace(cat=lambda xs, dim: _arr(np.concatenate(xs, axis=dim)))}
    terminal = _helper("reacher_d2p_masked_action_chart.py", "terminal_normalized", namespace)
    frames, history, future = _history()
    model = _TracePredictor()
    result = terminal(model, frames, history, future)
    _assert_calls(model)
    np.testing.assert_array_equal(result, [[106.]])


def test_h7_actual_pusht_prefix_then_observed_suffix_nonzero_unique_history():
    # Bypass action normalization deliberately: these are unique packet labels,
    # not legal physical commands. This test isolates history indexing only.
    namespace = {
        "torch": SimpleNamespace(cat=lambda xs, dim: _arr(np.concatenate(xs, axis=dim))),
        "_batched_live_action_tokens": lambda params, *, action_block: params,
    }
    advance = _helper("preflight_pusht_d2p_p1_reconfiguration.py", "_advance", namespace)
    suffix = _helper("preflight_pusht_d2p_p1_reconfiguration.py", "_suffix_terminal", namespace)
    frames, history, future = _history()
    model = _TracePredictor()
    after, actions = advance(model, frames, history, future[:, :1])
    np.testing.assert_array_equal(after[0, :, 0], [102., 103., 104.])
    np.testing.assert_array_equal(actions[0, :, 0], [22., 33., 44.])
    # Observed successor replaces the predicted one; give it the same unique
    # label to check the production nuisance runner's causal boundary splice.
    observed = _arr([[[104.]]])
    observed_frames = _arr(np.concatenate((frames[:, 1:], observed), axis=1))
    observed_actions = _arr(np.concatenate((history[:, 1:], future[:, :1]), axis=1))
    result = suffix(model, observed_frames, observed_actions, future[:, 1:])
    _assert_calls(model)
    np.testing.assert_array_equal(result, [[106.]])
    direct_model = _TracePredictor()
    direct = suffix(direct_model, frames, history, future)
    _assert_calls(direct_model)
    np.testing.assert_array_equal(direct, result)
