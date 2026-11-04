import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p2_common_objective_ledger_20261102_exclusive as ledger
from experiments.spectral_reachability import p2_same_bank_native_objective_20261102_exclusive as native
from experiments.spectral_reachability import p2_same_bank_ledger_consumer_20261102_exclusive as consumer


def transition(state, history, packet):
    next_state = state[:, -1] + packet.sum().expand(192) / 100
    return torch.cat((state[:, 1:], next_state[:, None]), dim=1), history


def run_fixture(path, tamper=False, fail=False, simulated_hooks=False):
    fixed = native.verified_input()
    journal = ledger.Ledger(path)
    try:
        meter = native.CappedMeter(journal)
        def instrumented_transition(state, history, packet):
            meter.call("action_encoder", lambda: None)
            def predict():
                meter.call("predictor", lambda: None)
                meter.call("pred_proj", lambda: None)
                return transition(state, history, packet)
            return meter.call("predict", predict)
        callback = native.make_objective(fixed, instrumented_transition if simulated_hooks else transition,
                                         meter, torch.device("cpu"))
        def emit(event):
            event = dict(event)
            if tamper and event["kind"] == "attempt" and event["iteration"] == 0:
                event["action_sha256"] = "0" * 64
            journal.record("optimizer_" + event["kind"], event)
        if fail:
            def failing(*args):
                raise RuntimeError("failed fixture")
            callback = failing
        return native.run_schedule(fixed, callback, emit)
    finally:
        journal.close()


def test_real_fixed_bank_fixture_consumer_and_native_rejection(tmp_path):
    path = tmp_path / "fixture.jsonl"
    run_fixture(path)
    report = consumer.audit(path, "cpu_fixture")
    assert report["candidate_count_per_arm"] == 16
    assert all(x == {"scores": 144, "gradients": 128} for x in report["callbacks_returned"].values())
    assert report["native_execution_authenticated"] is False
    with pytest.raises(ValueError, match="operation attempts"):
        consumer.audit(path, "native_hooks")


def test_rehashed_wrong_initial_action_rejected(tmp_path):
    path = tmp_path / "wrong.jsonl"
    run_fixture(path, tamper=True)
    with pytest.raises(ValueError, match="initial action"):
        consumer.audit(path, "cpu_fixture")


def test_synthetic_hook_counts_do_not_certify_native_execution(tmp_path):
    path = tmp_path / "synthetic_hooks.jsonl"
    run_fixture(path, simulated_hooks=True)
    report = consumer.audit(path, "native_hooks")
    assert report["native_execution_authenticated"] is False
    assert report["production_execution_authorized"] is False
    assert report["equal_total_compute"] is False


def test_failed_schedule_is_not_complete(tmp_path):
    path = tmp_path / "failed.jsonl"
    with pytest.raises(RuntimeError):
        run_fixture(path, fail=True)
    with pytest.raises(ValueError):
        consumer.audit(path, "cpu_fixture")
