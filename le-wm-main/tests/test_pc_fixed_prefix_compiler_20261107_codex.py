import numpy as np
import pytest
import torch

from experiments.pc_fixed_prefix_compiler_20261107_codex import compile_fixed_prefix
from experiments.pc_input_permissions import RGB, RGBLeWM, freeze_audit


class Model(torch.nn.Module):
    def get_cost(self, context, actions):
        return ((actions - 0.3) ** 2).sum(dim=(2, 3))


class Identity:
    def transform(self, value):
        return value


def setup_case(tmp_path):
    f = tmp_path / "checkpoint"
    f.write_text("frozen")
    audit = freeze_audit(RGB, "lewm", [f], preprocessing="test", clock="2x5")
    model = Model().eval()
    wrapper = RGBLeWM(model, audit)
    context = {"pixels": torch.zeros(1, 1, 1), "goal": torch.zeros(1, 1, 1),
               "action": torch.zeros(1, 1, 0)}
    parents = np.zeros((3, 10, 2), dtype=np.float32)
    parents[1] = 0.3
    parents[2] = -0.1
    probes = np.full((2, 5, 2), -0.2, dtype=np.float32)
    tail = np.full((2, 5, 2), 0.6, dtype=np.float32)
    return wrapper, context, parents, probes, tail


def test_real_boundary_selection_and_query_receipts(tmp_path):
    wrapper, context, parents, probes, tails = setup_case(tmp_path)
    out = compile_fixed_prefix(wrapper, context, parents, probes, tails, scaler=Identity(),
                               prefix_steps=5, retain_count=1, strategy="lookahead",
                               max_queries=11, seed=1)
    assert out["choice"].source == "parent"
    np.testing.assert_array_equal(out["actions"], parents[1])
    assert out["model_queries"] == 11 and out["forward_calls"] == len(wrapper.receipts) == 3
    assert set(context) == {"pixels", "goal", "action"}
    assert all(x["track_id"] == RGB for x in wrapper.receipts)


def test_budget_and_privilege_fail_before_query(tmp_path):
    wrapper, ctx, parents, probes, tails = setup_case(tmp_path)
    opts = dict(scaler=Identity(), prefix_steps=5, retain_count=1, strategy="lookahead", seed=1)
    with pytest.raises(ValueError, match="budget"):
        compile_fixed_prefix(wrapper, ctx, parents, probes, tails, max_queries=10, **opts)
    with pytest.raises(ValueError, match="observation"):
        compile_fixed_prefix(wrapper, {**ctx, "proprio": torch.ones(4)}, parents, probes, tails,
                             max_queries=11, **opts)
    assert not wrapper.receipts


def test_retained_prefixes_and_strategy_are_deterministic(tmp_path):
    wrapper, ctx, parents, probes, tails = setup_case(tmp_path)
    opts = dict(scaler=Identity(), prefix_steps=5, retain_count=2, strategy="shuffled",
                max_queries=13, seed=16)
    a = compile_fixed_prefix(wrapper, ctx, parents, probes, tails, **opts)
    b = compile_fixed_prefix(wrapper, ctx, parents, probes, tails, **opts)
    assert a["retained_prefixes"] == b["retained_prefixes"]
    assert len(set(a["retained_prefixes"])) == 2
    np.testing.assert_array_equal(a["actions"], b["actions"])
