from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_observed_h2_kernel_20261030_v1 as kernel


def test_two_packets_use_predicted_context_and_preserve_budget(monkeypatch):
    calls = []
    first = {"history": "actual_cp5"}
    monkeypatch.setattr(kernel.observed.base.encode.core, "validate_task", lambda t: None)
    monkeypatch.setattr(kernel.observed, "observed_context", lambda t, z, p: first)
    monkeypatch.setattr(kernel.observed.base.d2p, "validate_advance", lambda *args: None)

    def compile_one(task, context, original, backend, operation, journal):
        calls.append((deepcopy(context), original.copy(), backend, operation))
        return {"status": "observed_suffix_selected_not_admitted",
                "winner": {"commands": np.zeros((5, 2), np.float32).tolist(),
                           "context": {"history": "prediction_" + str(len(calls))}}}

    monkeypatch.setattr(kernel.observed, "compile_suffix", compile_one)
    original = np.concatenate((np.full((5, 2), .25, np.float32),
                               np.full((5, 2), .5, np.float32)))
    result = kernel.compile_two_packets({}, np.zeros(192), np.zeros((5, 2)),
                                        original, lambda i: ("backend", i),
                                        lambda i: ("guard", i), lambda e: None)
    assert calls[0][0] == first
    assert calls[1][0] == {"history": "prediction_1"}
    assert np.array_equal(calls[0][1], original[:5])
    assert np.array_equal(calls[1][1], original[5:])
    assert calls[0][2:] == (("backend", 0), ("guard", 0))
    assert calls[1][2:] == (("backend", 1), ("guard", 1))
    assert result["per_packet_budget"] == dict(predict=5, action_encoder=5, jacobian=1, svd=1)
    assert len(result["raw10"]) == 10 and result["physical_authorized"] is False


def test_wrong_horizon_fails_before_backend(monkeypatch):
    monkeypatch.setattr(kernel.observed.base.encode.core, "validate_task", lambda t: None)
    with pytest.raises(ValueError, match="exactly two"):
        kernel.compile_two_packets({}, None, None, np.zeros((5, 2)), None, None, None)


def test_failed_first_packet_never_dispatches_second(monkeypatch):
    monkeypatch.setattr(kernel.observed.base.encode.core, "validate_task", lambda t: None)
    monkeypatch.setattr(kernel.observed, "observed_context", lambda *args: {})
    seen = []
    monkeypatch.setattr(kernel.observed, "compile_suffix",
                        lambda *args: {"status": "partial"})
    with pytest.raises(ValueError, match="non-admitted"):
        kernel.compile_two_packets({}, None, None, np.zeros((10, 2), np.float32),
                                   lambda i: seen.append(i), lambda i: None, None)
    assert seen == [0]


def test_backtrack_task_schema_is_not_implicitly_accepted():
    # Its actual top-level field names are different from the compiler task.
    backtrack_shape = dict.fromkeys(
        ("action_mean", "action_std", "binding", "goal", "projection", "root_context"))
    with pytest.raises((ValueError, AssertionError, KeyError, TypeError)):
        kernel.compile_two_packets(backtrack_shape, None, None, None, None, None, None)
