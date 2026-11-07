import copy

import pytest
import torch

from experiments.pc_input_permissions import (
    RGB, STATE, RGBLeWM, StateAssistedPointMaze, cache_key, freeze_audit,
    manifest, track, validate_evidence,
)


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.eval()

    def encode(self, inputs):
        return {"emb": inputs["pixels"] * self.weight}

    def get_cost(self, context, actions):
        context["predicted_emb"] = actions * self.weight
        return actions.square().sum()


@pytest.fixture
def audits(tmp_path):
    source = tmp_path / "binding"
    source.write_text("fixture only, not a checkpoint certification")
    return source, {
        RGB: freeze_audit(RGB, "lewm", [source], preprocessing="fixture", clock="fixture"),
        STATE: freeze_audit(STATE, "pointmaze_jepa", [source], preprocessing="fixture", clock="fixture",
                            proprio_fields=("x", "y", "vx", "vy")),
    }


@pytest.mark.parametrize("name", [None, "", "rgb", "legacy_unclassified"])
def test_explicit_track(name):
    with pytest.raises(ValueError):
        track(name)


@pytest.mark.parametrize("extra", ["proprio", "state", "env", "oracle"])
def test_rgb_rejects_extra(audits, extra):
    wrapper = RGBLeWM(Dummy(), audits[1][RGB])
    with pytest.raises(ValueError):
        wrapper.encode({"pixels": torch.ones(1), extra: torch.ones(4)})
    assert wrapper.receipts == []


def test_no_rgb_history_default(audits):
    wrapper = RGBLeWM(Dummy(), audits[1][RGB])
    with pytest.raises(ValueError):
        wrapper.score({"pixels": torch.ones(1), "goal": torch.ones(1)}, torch.ones(1))


def test_rgb_no_context_mutation_and_preserves_gradient(audits):
    wrapper = RGBLeWM(Dummy(), audits[1][RGB])
    inputs = {k: torch.ones(1) for k in ("pixels", "goal", "action")}
    actions = torch.ones(1, requires_grad=True)
    score, context = wrapper.score(inputs, actions)
    score.backward()
    assert actions.grad.item() == 2
    assert "predicted_emb" not in inputs
    assert "predicted_emb" in context


def test_bad_track_checkpoint_pair(audits):
    with pytest.raises(ValueError):
        RGBLeWM(Dummy(), audits[1][STATE])
    with pytest.raises(ValueError):
        StateAssistedPointMaze(Dummy(), audits[1][RGB])


def test_unfrozen_rejected(audits):
    model = Dummy()
    model.weight.requires_grad_(True)
    with pytest.raises(ValueError):
        RGBLeWM(model, audits[1][RGB])


def test_manifest_and_cross_track_promotion(audits):
    source, bindings = audits
    wrapper = RGBLeWM(Dummy(), bindings[RGB])
    wrapper.encode({"pixels": torch.ones(1)})
    record = manifest(bindings[RGB], algorithm_version="fixture", protocol_path=source,
                      task_manifest_path=source, receipts=wrapper.receipts)
    assert validate_evidence(record, target_track=RGB, algorithm_version="fixture",
                             protocol_hash=record["protocol_hash"])
    with pytest.raises(ValueError):
        validate_evidence(record, target_track=STATE, algorithm_version="fixture",
                          protocol_hash=record["protocol_hash"])
    forged = copy.deepcopy(record)
    forged["track_id"] = STATE
    forged["evidence_scope"] = "state_assisted_mechanism"
    with pytest.raises(ValueError):
        validate_evidence(forged, target_track=STATE, algorithm_version="fixture",
                          protocol_hash=record["protocol_hash"])


def test_cache_separation(audits):
    keys = [cache_key(a, algorithm_version="fixture", inputs={}) for a in audits[1].values()]
    assert keys[0] != keys[1]


def test_dependency_drift(audits):
    source, bindings = audits
    source.write_text("changed")
    with pytest.raises(ValueError):
        bindings[RGB].verify()


def test_state_missing_proprio_rejected(audits):
    class Model(Dummy):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.use_proprio = True
            self.model.proprio_encoder = torch.nn.Identity()
            self.eval()
    wrapper = StateAssistedPointMaze(Model(), audits[1][STATE])
    with pytest.raises(ValueError):
        wrapper.predict({"visual": torch.zeros(1, 1, 3, 8, 8)}, torch.zeros(1, 1, 10))
    assert wrapper.receipts == []
