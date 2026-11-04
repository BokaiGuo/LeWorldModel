import json

import pytest
import torch

from experiments.spectral_reachability import codex_p2_program_model_scope_20261030_v1 as hooks
from experiments.spectral_reachability import p2_common_objective_ledger_20261102_exclusive as core
from experiments.spectral_reachability import p2_conditional_hook_meter_20261102_exclusive as routing
from experiments.spectral_reachability import p2_same_bank_conditional_suffix_ablation_20261102_exclusive as bridge


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = torch.nn.Identity()
        self.predictor = torch.nn.Identity()
        self.pred_proj = torch.nn.Identity()

    def predict(self, state, encoded):
        signal = self.pred_proj(self.predictor(encoded))
        return state[:, -1] + torch.cat((signal, signal.new_zeros((1, 182))), dim=1)


def fixture_transition(model):
    def transition(state, history, packet):
        token = packet.reshape(1, 10)
        successor = model.predict(state, model.action_encoder(token))
        return torch.cat((state[:, 1:], successor[:, None]), 1), torch.cat((history[:, 1:], token[:, None]), 1)
    return transition


def test_full_real_candidate_schedule_and_hooks_cpu_fixture(tmp_path):
    journal = core.Ledger(tmp_path / "events.jsonl")
    meter = routing.RoutedModelMeter(journal)
    model = Model()
    initial_predict = model.predict.__func__
    try:
        with hooks.model_scope(model, meter):
            result = bridge.run_bridge(bridge.native.verified_input(), fixture_transition(model),
                                       meter.ledger_view, torch.device("cpu"))
        summary = meter.finish()
    finally:
        journal.close()
    assert model.predict.__func__ is initial_predict
    assert len(result["rows"]) == 16
    assert summary["attempted"] == routing.CAPS == summary["returned"]
    assert meter.counts == routing.CAPS == meter.returned_counts
    verified = core.verify_ledger(tmp_path / "events.jsonl")
    assert not verified["outstanding"]
    for category, axes in routing.CAPS.items():
        for axis, count in axes.items():
            assert verified["counts"][category, axis] == {"attempt": count, "returned": count, "failed": 0}
    assert not summary["native_execution_authenticated"]


@pytest.mark.parametrize("category,axis", [
    ("composition_inherited", "conditional_prefix_transition"),
    ("common_composition", "predict"),
    ("common_no_composition_sra", "predictor"),
])
def test_caps_reject_before_dispatch_and_latch(tmp_path, category, axis):
    journal = core.Ledger(tmp_path / "events.jsonl")
    meter = routing.RoutedMeter(journal)
    called = []
    try:
        with meter.ledger_view.phase(category):
            run = meter.call if axis in routing.HOOKS else meter.ledger_view.call
            for _ in range(routing.CAPS[category][axis]):
                run(axis, lambda: None)
            with pytest.raises(ValueError, match="cap"):
                run(axis, lambda: called.append(True))
        assert not called and meter.failed
        with pytest.raises(ValueError, match="incomplete"):
            meter.finish()
    finally:
        journal.close()
    records = [json.loads(x)["record"] for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(x["kind"] == "routed_meter_rejected" and x["payload"]["axis"] == axis for x in records)


def test_failures_are_durable_and_no_subsequent_dispatch(tmp_path):
    journal = core.Ledger(tmp_path / "events.jsonl")
    meter = routing.RoutedMeter(journal)
    def failing():
        raise RuntimeError("model failed")
    try:
        with meter.ledger_view.phase("composition_inherited"):
            with pytest.raises(RuntimeError, match="model failed"):
                meter.call("predict", failing)
            with pytest.raises(ValueError, match="already failed"):
                meter.call("predictor", lambda: pytest.fail("must not execute"))
    finally:
        journal.close()
    verified = core.verify_ledger(tmp_path / "events.jsonl")
    assert verified["counts"]["composition_inherited", "predict"] == {
        "attempt": 1, "returned": 0, "failed": 1}
    assert not verified["outstanding"]


def test_wrong_route_or_category_fails_closed(tmp_path):
    for i, (category, axis, call_hook) in enumerate([
        ("shared_setup", "predict", True),
        ("composition_inherited", "predict", False),
        ("common_composition", "conditional_prefix_transition", False),
        ("common_composition", "action_encoder", False),
    ]):
        journal = core.Ledger(tmp_path / f"events{i}.jsonl")
        meter = routing.RoutedMeter(journal)
        try:
            with meter.ledger_view.phase(category):
                with pytest.raises(ValueError, match="invalid route"):
                    (meter.call if call_hook else meter.ledger_view.call)(axis, lambda: None)
        finally:
            journal.close()


def test_nested_caught_failure_cannot_be_accepted(tmp_path):
    journal = core.Ledger(tmp_path / "events.jsonl")
    meter = routing.RoutedMeter(journal)
    def outer():
        try:
            meter.call("predict", lambda: (_ for _ in ()).throw(RuntimeError("inner")))
        except RuntimeError:
            return 1
    try:
        with meter.ledger_view.phase("composition_inherited"):
            with pytest.raises(RuntimeError, match="nested dispatch"):
                meter.ledger_view.call("conditional_prefix_transition", outer)
        assert meter.failed
    finally:
        journal.close()


def test_public_meter_api_supports_native_objective_cpu(tmp_path):
    journal = core.Ledger(tmp_path / "events.jsonl")
    meter = routing.RoutedModelMeter(journal)
    model = Model()
    fixed = bridge.native.verified_input()
    objective = bridge.native.make_objective(fixed, fixture_transition(model), meter, torch.device("cpu"))
    import numpy as np
    raw = np.frombuffer(fixed["bank"][0], dtype="<f4").reshape(10, 2).copy()
    try:
        with hooks.model_scope(model, meter):
            _, grad = objective("composition", json.loads(fixed["context_json"]),
                                json.loads(fixed["goal_json"]), raw, True)
        assert grad.shape == (10, 2)
        assert meter.counts["common_composition"]["transition_packet"] == 2
        assert meter.counts["common_composition"]["autograd_grad"] == 1
        assert meter.counts["common_composition"]["predict"] == 2
        snapshot = meter.counts
        snapshot["common_composition"]["predict"] = 999
        assert meter.counts["common_composition"]["predict"] == 2
    finally:
        journal.close()


def test_model_scope_restores_hooks_after_exception(tmp_path):
    class Broken(Model):
        def predict(self, state, encoded):
            raise RuntimeError("injected model error")
    model = Broken()
    journal = core.Ledger(tmp_path / "events.jsonl")
    meter = routing.RoutedModelMeter(journal)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            with hooks.model_scope(model, meter):
                with meter.phase("composition_inherited"):
                    model.predict(torch.zeros(1, 3, 192), torch.zeros(1, 10))
        assert "predict" not in vars(model)
        assert meter.failed
    finally:
        journal.close()
    audit = core.verify_ledger(tmp_path / "events.jsonl")
    assert audit["counts"]["composition_inherited", "predict"]["failed"] == 1


def test_phase_validation_failure_records_and_latches(tmp_path):
    journal = core.Ledger(tmp_path / "events.jsonl")
    meter = routing.RoutedModelMeter(journal)
    try:
        with pytest.raises(ValueError, match="bad shape"):
            with meter.phase("composition_inherited"):
                raise ValueError("bad shape")
        assert meter.failed
        with meter.phase("common_composition"):
            with pytest.raises(ValueError, match="already failed"):
                meter.call("predict", lambda: pytest.fail("dispatch after failure"))
    finally:
        journal.close()
    records = [json.loads(x)["record"] for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(row["kind"] == "routed_phase_failed" for row in records)
