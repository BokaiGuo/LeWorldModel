"""Metadata/AST and NumPy arithmetic doubles only; no neural or GPU calls."""
import ast
import contextlib
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import pusht_d2p_metric_program_compiler as caller


def test_cpu_rejected_before_touching_model():
    class Forbidden:
        def modules(self):
            raise AssertionError("model touched")
    context = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    with pytest.raises(ValueError, match="CPU neural fallback forbidden"):
        caller.compile_programs(Forbidden(), context, context, [], [], 1)


DEVICE = SimpleNamespace(type="cuda")  # Metadata label; never allocates CUDA.


class Array(np.ndarray):
    device = DEVICE
    requires_grad = False

    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)
    def clone(self): return self.copy()
    def requires_grad_(self, value): return self
    def new_zeros(self, shape): return array(np.zeros(shape))


def array(value, **kwargs):
    return np.asarray(value, dtype=np.float32).view(Array)


def fixture_caller():
    """Run production body with explicit NumPy substitutes, no torch import.

    Removing imports in this AST-only harness prevents accidentally running
    real neural helpers on CPU. It does NOT test torch autograd or CUDA.
    """
    calls = []

    def advance(model, emb, actions, prefix):
        calls.append(("prefix", np.array(prefix)))
        after = emb.copy()
        after[0, -1] = [0., 1. + float(prefix[0, 0, 0])]
        return after, actions.copy()

    def terminal(model, emb, actions, params):
        calls.append(("suffix", float(emb[0, -1, 1]), np.array(params)))
        x = params[0].ravel()
        # Different own states, plus curvature invisible at nominal zero.
        state = float(emb[0, -1, 1])
        return array([[0., state + x.sum() + 4.*(x*x).sum()]])

    def jacobian(fn, nominal, vectorize):
        # Analytic derivative of the arithmetic terminal above at zero only.
        assert np.all(nominal == 0)
        return array(np.vstack((np.zeros(nominal.size), np.ones(nominal.size))))

    fake = SimpleNamespace(Tensor=Array, float32=np.dtype("float32"),
        as_tensor=array, isfinite=np.isfinite, no_grad=contextlib.nullcontext,
        enable_grad=contextlib.nullcontext, tanh=np.tanh,
        cat=lambda xs, dim: array(np.concatenate(xs, axis=dim)),
        autograd=SimpleNamespace(functional=SimpleNamespace(jacobian=jacobian)))
    tree = ast.parse(inspect.getsource(caller.compile_programs))
    fn = tree.body[0]
    fn.body = [node for node in fn.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {"np": np, "torch": fake, "p1": SimpleNamespace(_advance=advance, _suffix_terminal=terminal),
                 "scoring": caller.scoring, "compile_metric": caller.compile_metric,
                 "_require_real_input": caller._require_real_input}
    exec(compile(ast.fix_missing_locations(tree), "<numpy-only-caller-fixture>", "exec"), namespace)
    return namespace["compile_programs"], calls


def inputs():
    model = SimpleNamespace(modules=lambda: [], parameters=lambda: [], buffers=lambda: [])
    return dict(model=model, emb=array(np.zeros((1, 3, 2))), actions=array(np.zeros((1, 3, 10))),
        goal_vectors=[[0., 0.], [2., 0.]], prefix_params=[[[0., 0.]], [[.5, 0.]]], suffix_horizon=2)


def test_numpy_fixture_own_context_jacobian_full_params_and_metric_receipts():
    compile_programs, calls = fixture_caller()
    result = compile_programs(**inputs(), beta=2.)
    assert len([c for c in calls if c[0] == "prefix"]) == 2
    for index, (receipt, row) in enumerate(zip(result["receipts"], result["scoring"]["rows"], strict=True)):
        assert receipt["prefix_vector"] == [0., 1. + .5*index]
        assert receipt["nominal_terminal"] == receipt["prefix_vector"]
        np.testing.assert_array_equal(receipt["nominal"], np.zeros((2, 2)))
        np.testing.assert_array_equal(receipt["jacobian"], [[0.]*4, [1.]*4])
        np.testing.assert_allclose(receipt["executed_program_params"], row["program_params"], rtol=1e-6)
        assert len(receipt["executed_program_params"]) == 3
        assert receipt["metric_sha256"] == result["scoring"]["metric"]["sha256"]
        assert row["objective"] == pytest.approx(row["stage_cost"] + 2.*row["own_reach_value"])
    assert result["P1_admission"] is False
    assert result["spectral_prefix_generation"] is False
    assert result["physical_labels_used"] is False
    json.dumps(result, allow_nan=False)


def test_nonlinear_diagnostic_cannot_rerank_primary_score(monkeypatch):
    compile_programs, _ = fixture_caller()
    original = caller.scoring.score_terminal_predictions
    def adversarial(**kwargs):
        result = original(**kwargs)
        result["predicted_terminal_costs"] = [1e6, 0.]
        return result
    monkeypatch.setattr(caller.scoring, "score_terminal_predictions", adversarial)
    result = compile_programs(**inputs())
    assert result["selected_candidate_id"] == "prefix_0"
    assert result["receipts"][1]["nonlinear_terminal_cost_diagnostic_only"] == 0.
    assert result["nonlinear_predictions_used_for_selection"] is False


@pytest.mark.parametrize("changes", [
    {"suffix_horizon": 0}, {"suffix_horizon": True}, {"damping": 0.},
    {"beta": -1.}, {"prefix_params": [[[.2, 0.]]]},
    {"goal_vectors": [[0., 0.]]}, {"prefix_params": [[[float("nan"), 0.]]]},
])
def test_invalid_contract_rejected_before_forecasts(changes):
    compile_programs, calls = fixture_caller()
    with pytest.raises(ValueError):
        compile_programs(**(inputs() | changes))
    assert calls == []


def test_physical_labels_not_an_input():
    assert "physical_labels" not in inspect.signature(caller.compile_programs).parameters
    tree = ast.parse(inspect.getsource(caller.compile_programs))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(isinstance(c.func, ast.Attribute) and c.func.attr == "_advance" for c in calls)
    assert any(isinstance(c.func, ast.Attribute) and c.func.attr == "_suffix_terminal" for c in calls)
    assert any(isinstance(c.func, ast.Attribute) and c.func.attr == "compile_candidates" for c in calls)


@pytest.mark.parametrize("key,value", [
    ("goal_vectors", np.array([[0.+1.j, 0.], [2., 0.]])),
    ("prefix_params", np.array([[[0., 0.]], [[.5+1.j, 0.]]])),
    ("prefix_params", [[[False, False]]]),
    ("goal_vectors", [["0", "0"], ["2", "0"]]),
])
def test_nonreal_input_rejected_before_cast_or_model(key, value):
    compile_programs, calls = fixture_caller()
    with pytest.raises(ValueError, match="real numeric"):
        compile_programs(**(inputs() | {key: value}))
    assert calls == []


def test_zero_not_first_and_stable_bank_order_ties():
    compile_programs, _ = fixture_caller()
    args = inputs() | {"prefix_params": [[[.5, 0.]], [[0., 0.]], [[0., 0.]]]}
    result = compile_programs(**args, beta=0.)
    assert result["scoring"]["zero_candidate_id"] == "prefix_1"
    assert result["selected_candidate_id"] == "prefix_1"
    assert len(result["receipts"]) == 3  # no deduplication or filtering


@pytest.mark.parametrize("kind", ["training", "trainable_parameter", "wrong_buffer_device"])
def test_unfrozen_or_wrong_device_model_rejected_without_predicting(kind):
    compile_programs, calls = fixture_caller()
    model = inputs()["model"]
    if kind == "training":
        model.modules = lambda: [SimpleNamespace(training=True)]
    elif kind == "trainable_parameter":
        model.parameters = lambda: [SimpleNamespace(requires_grad=True, device=DEVICE)]
    else:
        model.buffers = lambda: [SimpleNamespace(device="cpu")]
    with pytest.raises(ValueError, match="frozen eval"):
        compile_programs(**(inputs() | {"model": model}))
    assert calls == []
