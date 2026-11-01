import pytest
import torch

from experiments.spectral_reachability import (
    codex_p3_chunk_extension_20261101 as chunk,
    p3_c3_live_executor_bridge_20261102_exclusive as bridge,
    p3_c3_model_boundary_20261102_exclusive as boundary,
)


class Model:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def action_encoder(self, value):
        self.calls.append("encoder")
        if self.fail == "encoder":
            raise RuntimeError("encoder failure")
        return value

    def predict(self, context, encoded):
        self.calls.append("predict")
        if self.fail == "predict":
            raise RuntimeError("predict failure")
        return context


class FakeEnvironment:
    def __init__(self):
        self.calls = 0

    def step(self, action):
        self.calls += 1
        return None, 0, False, False, {}


def setup(model, cap=8):
    ledger = boundary.BoundaryAdapter(model, point=20, arm="signal",
        caps={"action_encoder": cap, "predict": cap, "predictor_sample": cap},
        sync=lambda: None)
    args = {
        "env": FakeEnvironment(), "states": torch.zeros((1, 1, 2)),
        "history_actions": torch.zeros((1, 0, 2)),
        "action_tokens": torch.zeros((1, 2, 2)), "physical_actions": [0, 0],
        "goal": torch.ones(2), "projection": torch.eye(2), "token_steps": 1,
        "config": chunk.Config(1, 1, 1, 0, 0), "arm": "dynamic",
        "meter": chunk.Meter({"predict": 8, "predictor_samples": 8,
                              "encode": 0, "ad": 0, "svd": 0}),
    }
    return ledger, args


def test_existing_executor_counts_each_real_model_boundary_on_cpu():
    model = Model()
    ledger, args = setup(model)
    output = bridge.run_candidate(ledger, root=0, index=0, **args)
    assert output["chunk_steps"] == 2
    assert model.calls == ["encoder", "predict"] * 3
    assert ledger.snapshot()["attempted"] == ledger.snapshot()["returned"] == {
        "action_encoder": 3, "predict": 3, "predictor_sample": 3}
    assert args["env"].calls == 2  # Fake environment, not a physical rollout.
    assert ledger.snapshot()["P3_admission"] is False


def test_existing_executor_rejects_cap_before_extra_model_call():
    model = Model()
    ledger, args = setup(model, cap=1)
    with pytest.raises(ValueError, match="resource cap"):
        bridge.run_candidate(ledger, root=0, index=0, **args)
    assert model.calls == ["encoder", "predict"]
    assert args["env"].calls == 0
    assert ledger.snapshot()["failed"] is True


def test_existing_executor_exception_records_actual_boundary():
    model = Model(fail="predict")
    ledger, args = setup(model)
    with pytest.raises(RuntimeError, match="predict failure"):
        bridge.run_candidate(ledger, root=0, index=0, **args)
    saved = ledger.snapshot()
    assert saved["attempted"]["predict"] == 1
    assert saved["returned"]["predict"] == 0
    assert saved["returned"]["action_encoder"] == 1
    assert args["env"].calls == 0
