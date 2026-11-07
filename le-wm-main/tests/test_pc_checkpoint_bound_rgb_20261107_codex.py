import json

import pytest
import torch

from experiments.pc_checkpoint_bound_rgb_20261107_codex import CheckpointBoundRGB
from experiments.pc_input_permissions import RGB, RGBLeWM, freeze_audit


class Fixture(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([weight]), requires_grad=False)
        self.eval()

    def encode(self, inputs):
        return {"emb": inputs["pixels"] * self.weight}


@pytest.fixture
def setup_binding(tmp_path):
    weights = tmp_path / "weights.pt"
    config = tmp_path / "config.json"
    torch.save(Fixture(1.0).state_dict(), weights)
    config.write_text(json.dumps({"fixture": True}))
    audit = freeze_audit(RGB, "lewm", [weights, config], preprocessing="fixture", clock="fixture")
    return audit, weights, config


def test_original_boundary_allows_wrong_frozen_model(setup_binding):
    audit, _, _ = setup_binding
    # Existing boundary enforces track and freezing, but does not inspect
    # whether these weights came from the audited file.
    wrapper = RGBLeWM(Fixture(2.0), audit)
    assert wrapper.encode({"pixels": torch.ones(1)})["emb"].item() == 2.0


def test_checkpoint_binding_rejects_wrong_weights_and_missing_dependencies(setup_binding, tmp_path):
    audit, weights, config = setup_binding
    with pytest.raises(ValueError, match="weight mismatch"):
        CheckpointBoundRGB(Fixture(2.0), audit, weights, config)
    other = tmp_path / "other.json"
    other.write_text("{}")
    with pytest.raises(ValueError, match="not an audited dependency"):
        CheckpointBoundRGB(Fixture(1.0), audit, weights, other)


def test_success_and_runtime_mutation_rejection(setup_binding):
    audit, weights, config = setup_binding
    model = Fixture(1.0)
    wrapper = CheckpointBoundRGB(model, audit, weights, config)
    wrapper.encode({"pixels": torch.ones(1)})
    result = wrapper.finish()
    assert result["model_state_sha256"] == result["initial_model_state_sha256"]
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(ValueError, match="weight mismatch"):
        wrapper.finish()


def test_requires_runtime_call_and_does_not_accept_replacement(setup_binding):
    audit, weights, config = setup_binding
    wrapper = CheckpointBoundRGB(Fixture(1.0), audit, weights, config)
    with pytest.raises(ValueError, match="no runtime receipts"):
        wrapper.finish()
    wrapper.encode({"pixels": torch.ones(1)})
    wrapper._model = Fixture(1.0)
    with pytest.raises(ValueError, match="model object replaced"):
        wrapper.finish()


def test_config_and_weight_file_drift_fail_closed(setup_binding):
    audit, weights, config = setup_binding
    wrapper = CheckpointBoundRGB(Fixture(1.0), audit, weights, config)
    wrapper.encode({"pixels": torch.ones(1)})
    config.write_text('{"changed": true}')
    with pytest.raises(ValueError, match="dependency drift"):
        wrapper.finish()
