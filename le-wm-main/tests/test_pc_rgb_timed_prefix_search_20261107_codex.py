import numpy as np
import pytest
import torch

from experiments.pc_input_permissions import InputAudit, RGBLeWM, file_hash
from experiments.pc_rgb_timed_prefix_search_20261107_codex import build, compile_timed


class Model(torch.nn.Module):
    def get_cost(self, ctx, actions):
        return actions.sum(dim=(-1, -2))


class Scaler:
    def transform(self, actions):
        return actions


def test_prefix_clock_fallback_and_old_tail(tmp_path):
    f = tmp_path / "weight"
    f.write_text("frozen")
    audit = InputAudit("primary_rgb", "lewm", ((str(f), file_hash(f)),), (), "pre", "clock")
    ctx = {"pixels": torch.zeros(1), "goal": torch.zeros(1), "action": torch.zeros(1)}
    old = np.full((5, 2), .2, np.float32)
    outputs = []
    for arm in ("parent", "lookahead", "vector", "shuffled"):
        boundary = RGBLeWM(Model().eval(), audit)
        result = compile_timed(boundary, ctx, scaler=Scaler(), old_tail=old, candidate_seed=7,
                               strategy_seed=8, strategy=arm, remaining_steps=5)
        assert result["mode"] == "direct_fallback" and result["model_queries"] == 13
        assert result["model_packet_steps"] == 13
        outputs.append(result["actions"])
    assert all(np.array_equal(outputs[0], x) for x in outputs)
    parents, probes, tails = build(7, old, 5)
    np.testing.assert_array_equal(parents[0, :5], old)
    assert parents.shape == (3, 25, 2) and probes.shape == tails.shape == (2, 20, 2)
    with pytest.raises(ValueError):
        compile_timed(RGBLeWM(Model().eval(), audit), ctx, scaler=Scaler(), old_tail=None, candidate_seed=7,
                      strategy_seed=8, strategy="parent", remaining_steps=6)
