import numpy as np
import pytest
import torch

from experiments.pc_rgb_deadline_scoring_20261107_codex import compile_rgb
from experiments.pc_input_permissions import InputAudit, RGBLeWM


class DummyModel(torch.nn.Module):
    def get_cost(self, context, actions):
        return actions.sum(dim=(-1, -2))


class IdentityScaler:
    def transform(self, values):
        return values


def test_remaining_horizon_and_future_action_noninterference(tmp_path):
    dependency = tmp_path / "weight"
    dependency.write_text("frozen")
    from experiments.pc_input_permissions import file_hash
    audit = InputAudit("primary_rgb", "lewm", ((str(dependency), file_hash(dependency)),), (), "identity", "clock")
    model = DummyModel().eval()
    boundary = RGBLeWM(model, audit)
    ctx = {"pixels": torch.zeros(1), "goal": torch.zeros(1), "action": torch.zeros(1)}
    parents = np.zeros((3, 25, 2), np.float32)
    probes = np.zeros((2, 15, 2), np.float32)
    tails = np.zeros((2, 15, 2), np.float32)
    result = compile_rgb(boundary, ctx, parents, probes, tails, scaler=IdentityScaler(), prefix_steps=10,
                         retain_count=2, strategy="lookahead", budget=13, seed=2, remaining_steps=5)
    assert result["model_packet_steps"] == 13
    assert all(x["inputs"]["candidate_actions"]["shape"][2] == 1 for x in boundary.receipts)
    parents[:, 5:] = 1
    result2 = compile_rgb(boundary, ctx, parents, probes, tails, scaler=IdentityScaler(), prefix_steps=10,
                          retain_count=2, strategy="lookahead", budget=13, seed=2, remaining_steps=5)
    np.testing.assert_array_equal(result["parent_cost"], result2["parent_cost"])
    with pytest.raises(ValueError):
        compile_rgb(boundary, ctx, parents, probes, tails, scaler=IdentityScaler(), prefix_steps=10,
                    retain_count=2, strategy="lookahead", budget=13, seed=2, remaining_steps=6)
