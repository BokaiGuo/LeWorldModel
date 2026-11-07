import numpy as np
import pytest
import torch

from experiments.pc_input_permissions import InputAudit, RGBLeWM, file_hash
from experiments.pc_rgb_timed_prefix_clock_fixed_20261107_codex import compile_timed_clock_fixed
from experiments.pc_rgb_timed_prefix_search_20261107_codex import compile_timed


class Model(torch.nn.Module):
    def get_cost(self, context, actions):
        return actions.sum(dim=(-1, -2))


class Scaler:
    def transform(self, actions):
        return actions


@pytest.mark.parametrize("old_tail", [None, np.full((5, 2), 0.2, np.float32)])
@pytest.mark.parametrize("strategy", ["parent", "lookahead", "vector", "shuffled"])
def test_fallback_returns_exact_scored_chain_without_changing_choice(tmp_path, old_tail, strategy):
    binding = tmp_path / "binding"
    binding.write_text("fixture")
    audit = InputAudit("primary_rgb", "lewm", ((str(binding), file_hash(binding)),), (), "pre", "clock")
    context = {"pixels": torch.zeros(1), "goal": torch.zeros(1), "action": torch.zeros(1)}
    kwargs = dict(scaler=Scaler(), old_tail=old_tail, candidate_seed=7,
                  strategy_seed=8, strategy=strategy, remaining_steps=5)
    original = compile_timed(RGBLeWM(Model().eval(), audit), context, **kwargs)
    fixed = compile_timed_clock_fixed(RGBLeWM(Model().eval(), audit), context, **kwargs)
    assert original["actions"].shape == (25, 2)
    assert fixed["actions"].shape == (5, 2)
    np.testing.assert_array_equal(fixed["actions"], original["actions"][:5])
    assert fixed["choice"] == original["choice"]
    np.testing.assert_array_equal(fixed["parent_cost"], original["parent_cost"])
    assert fixed["model_queries"] == original["model_queries"] == 13


@pytest.mark.parametrize("remaining", [10, 15, 20, 25])
def test_program_branch_preserves_full_selected_chain(tmp_path, remaining):
    binding = tmp_path / "binding"
    binding.write_text("fixture")
    audit = InputAudit("primary_rgb", "lewm", ((str(binding), file_hash(binding)),), (), "pre", "clock")
    context = {"pixels": torch.zeros(1), "goal": torch.zeros(1), "action": torch.zeros(1)}
    kwargs = dict(scaler=Scaler(), old_tail=None, candidate_seed=7,
                  strategy_seed=8, strategy="lookahead", remaining_steps=remaining)
    original = compile_timed(RGBLeWM(Model().eval(), audit), context, **kwargs)
    fixed = compile_timed_clock_fixed(RGBLeWM(Model().eval(), audit), context, **kwargs)
    assert fixed["actions"].shape == (remaining, 2)
    assert original["actions"].shape == (25, 2)
    np.testing.assert_array_equal(fixed["actions"], original["actions"][:remaining])
    assert fixed["choice"] == original["choice"]
