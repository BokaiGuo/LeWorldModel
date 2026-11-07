import numpy as np
import pytest
import torch

from experiments.pc_fixed_prefix_state_adapter_20261107_codex import compile_state
from experiments.pc_fixed_prefix_state_batched_20261107_codex import compile_state_batched
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, freeze_audit


class Core(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_proprio = True
        self.proprio_encoder = torch.nn.Identity()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Core()

    def encode(self, data):
        return data

    def unroll(self, encoded, act_suffix):
        assert encoded["visual"].shape[0] == act_suffix.shape[1]
        return {"cost": ((act_suffix - 0.3) ** 2).sum(dim=(0, 2))}


def test_batch_matches_serial_and_charges_logical_candidates(tmp_path):
    dep = tmp_path / "weights"
    dep.write_text("frozen")
    audit = freeze_audit(STATE, "pointmaze_jepa", [dep], proprio_fields=("x", "y", "vx", "vy"),
                         preprocessing="test", clock="2x5")
    ctx = {"visual": torch.zeros(1, 1, 3, 2, 2), "proprio": torch.zeros(1, 1, 4)}
    rng = np.random.default_rng(10)
    parent = rng.uniform(-1, 1, (3, 10, 2)).astype(np.float32)
    probes = rng.uniform(-1, 1, (2, 5, 2)).astype(np.float32)
    tails = rng.uniform(-1, 1, (2, 5, 2)).astype(np.float32)
    args = dict(mean=torch.zeros(2), std=torch.ones(2),
                objective_factory=lambda _: lambda pred, actions: pred["cost"],
                prefix_steps=5, retain_count=1, strategy="lookahead", budget=11, seed=5)
    serial = compile_state(StateAssistedPointMaze(Model().eval(), audit), ctx, ctx, parent, probes, tails, **args)
    batched = compile_state_batched(StateAssistedPointMaze(Model().eval(), audit), ctx, ctx, parent, probes, tails,
                                   batch_size=4, **args)
    for field in ("parent_cost", "probe_cost", "tail_cost"):
        np.testing.assert_allclose(serial[field], batched[field], atol=1e-5)
    assert (serial["choice"].source, serial["choice"].index) == (batched["choice"].source, batched["choice"].index)
    assert serial["choice"].cost == pytest.approx(batched["choice"].cost, abs=1e-5)
    np.testing.assert_array_equal(serial["actions"], batched["actions"])
    assert batched["model_queries"] == 11 and batched["predict_calls"] == 4
    assert batched["batch_lengths"] == [3, 4, 2, 2]
    with pytest.raises(ValueError, match="positive batch"):
        compile_state_batched(StateAssistedPointMaze(Model().eval(), audit), ctx, ctx, parent, probes, tails,
                              batch_size=0, **args)


@pytest.mark.parametrize("strategy", ["parent", "lookahead", "vector", "shuffled"])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 32])
def test_batch_strategies_and_sizes(tmp_path, strategy, batch_size):
    dep = tmp_path / "weights"
    dep.write_text("frozen")
    audit = freeze_audit(STATE, "pointmaze_jepa", [dep], proprio_fields=("x", "y", "vx", "vy"),
                         preprocessing="test", clock="2x5")
    ctx = {"visual": torch.zeros(1, 1, 3, 2, 2), "proprio": torch.zeros(1, 1, 4)}
    rng = np.random.default_rng(137)
    parent = rng.uniform(-1, 1, (3, 10, 2)).astype(np.float32)
    probes = rng.uniform(-1, 1, (2, 5, 2)).astype(np.float32)
    tails = rng.uniform(-1, 1, (2, 5, 2)).astype(np.float32)
    args = dict(mean=torch.zeros(2), std=torch.ones(2),
                objective_factory=lambda _: lambda pred, actions: pred["cost"],
                prefix_steps=5, retain_count=2, strategy=strategy, budget=13, seed=5)
    a = compile_state(StateAssistedPointMaze(Model().eval(), audit), ctx, ctx, parent, probes, tails, **args)
    b = compile_state_batched(StateAssistedPointMaze(Model().eval(), audit), ctx, ctx, parent, probes, tails,
                              batch_size=batch_size, **args)
    assert a["retained_prefixes"] == b["retained_prefixes"]
    assert (a["choice"].source, a["choice"].index) == (b["choice"].source, b["choice"].index)
    np.testing.assert_array_equal(a["actions"], b["actions"])
    assert b["model_queries"] == sum(b["batch_lengths"]) == 13
