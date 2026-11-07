import pytest
import torch

from experiments import pc_input_permissions as v1
from experiments import pc_input_permissions_v2_20261107_codex as v2


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.eval()

    def encode(self, inputs):
        return {"emb": inputs["pixels"] * self.weight}

    def get_cost(self, inputs, actions):
        inputs["predicted_emb"] = actions * self.weight
        return actions.square().sum()


@pytest.fixture
def bindings(tmp_path):
    source = tmp_path / "fixture"
    source.write_text("not a checkpoint")
    paths = [source, v1.__file__, v2.__file__]
    return source, {
        v1.RGB: v1.freeze_audit(v1.RGB, "lewm", paths, preprocessing="fixture", clock="fixture"),
        v1.STATE: v1.freeze_audit(v1.STATE, "pointmaze_jepa", paths, preprocessing="fixture",
                                 clock="fixture", proprio_fields=("x", "y", "vx", "vy")),
    }


@pytest.mark.parametrize("operation", ["encode", "score"])
def test_failed_model_call_has_no_success_receipt(bindings, operation):
    class Broken(Dummy):
        def encode(self, inputs):
            raise RuntimeError("failed")

        def get_cost(self, context, actions):
            raise RuntimeError("failed")

    wrapper = v2.RGBLeWMv2(Broken(), bindings[1][v1.RGB])
    with pytest.raises(RuntimeError, match="failed"):
        if operation == "encode":
            wrapper.encode({"pixels": torch.ones(1)})
        else:
            wrapper.score({k: torch.ones(1) for k in ("pixels", "goal", "action")}, torch.ones(1))
    assert wrapper.receipts == []
    with pytest.raises(ValueError, match="actual call receipts"):
        v2.call_manifest(bindings[1][v1.RGB], algorithm_version="fixture",
                         protocol_path=bindings[0], task_manifest_path=bindings[0], receipts=wrapper.receipts)


@pytest.mark.parametrize("drift", ["train", "unfreeze", "during_call"])
def test_model_mode_or_parameter_drift_rejected(bindings, drift):
    class ModeChanging(Dummy):
        def encode(self, inputs):
            if drift == "during_call":
                self.train()
            return inputs
    model = ModeChanging()
    wrapper = v2.RGBLeWMv2(model, bindings[1][v1.RGB])
    if drift == "train":
        model.train()
    elif drift == "unfreeze":
        model.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        wrapper.encode({"pixels": torch.ones(1)})
    assert not wrapper.receipts


def test_state_unroll_failure_has_no_success_receipt(bindings):
    class BrokenState(Dummy):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.use_proprio = True
            self.model.proprio_encoder = torch.nn.Identity()
            self.eval()

        def encode(self, inputs):
            return inputs

        def unroll(self, encoded, act_suffix):
            raise RuntimeError("unroll failure")

    wrapper = v2.StateAssistedPointMazev2(BrokenState(), bindings[1][v1.STATE])
    with pytest.raises(RuntimeError):
        wrapper.predict({"visual": torch.zeros(1, 1, 3, 8, 8), "proprio": torch.zeros(1, 1, 4)},
                        torch.zeros(1, 1, 10))
    assert not wrapper.receipts


def test_success_gradient_track_separation_and_no_episode_claim(bindings):
    wrapper = v2.RGBLeWMv2(Dummy(), bindings[1][v1.RGB])
    inputs = {k: torch.ones(1) for k in ("pixels", "goal", "action")}
    actions = torch.ones(1, requires_grad=True)
    cost, context = wrapper.score(inputs, actions)
    cost.backward()
    assert actions.grad.item() == 2
    assert "predicted_emb" not in inputs and "predicted_emb" in context
    assert len(wrapper.receipts) == 1
    record = v2.call_manifest(bindings[1][v1.RGB], algorithm_version="fixture",
                              protocol_path=bindings[0], task_manifest_path=bindings[0], receipts=wrapper.receipts)
    assert record["execution_status"] == "model_calls_recorded"
    with pytest.raises(ValueError):
        v1.validate_evidence(record, target_track=v1.RGB, algorithm_version="fixture",
                             protocol_hash=record["protocol_hash"])
    with pytest.raises(ValueError):
        v1.validate_evidence(record, target_track=v1.STATE, algorithm_version="fixture",
                             protocol_hash=record["protocol_hash"])


def test_v2_requires_its_own_source_binding(bindings):
    audit = v1.freeze_audit(v1.RGB, "lewm", [bindings[0], v1.__file__],
                            preprocessing="fixture", clock="fixture")
    with pytest.raises(ValueError, match="both v1 and v2"):
        v2.RGBLeWMv2(Dummy(), audit)
