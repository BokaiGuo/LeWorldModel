import json
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium.spaces import Box

from experiments.pc_native_prefix_conditional_20261105_codex import NativePrefixConditional
from experiments.pc_prefix_heldout_effect_20261105_codex import NativePrefixHeldoutEffect
from experiments.pc_prefix_heldout_physical_20261105_codex import NativePrefixHeldoutPhysical
from experiments.pc_prefix_heldout_optimization_20261105_codex import NativePrefixHeldoutOptimization


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.counts = []

    def get_cost(self, info, actions):
        self.counts.append(actions.shape[1])
        return actions.square().sum((-1, -2))


def test_heldout_queries_do_not_change_search(tmp_path):
    models = [Model(), Model()]
    solvers = []
    for cls, model, name in zip((NativePrefixConditional, NativePrefixHeldoutEffect),
                                models, ("normal", "heldout")):
        solver = cls(model=model, batch_size=2, num_samples=20, n_steps=3,
                     topk=3, device="cpu", seed=5, audit_path=tmp_path / name)
        solver.configure(action_space=Box(-1, 1, shape=(3, 2), dtype=np.float32),
                         n_envs=3, config=SimpleNamespace(action_block=1, horizon=4))
        solvers.append(solver)
    info = {"x": torch.zeros(3, 1)}
    normal, heldout = [solver.solve(info)["actions"] for solver in solvers]
    assert torch.equal(normal, heldout)
    assert models[0].counts == [20, 16, 4, 16, 4] * 2
    assert models[1].counts == [20, 16, 4, 10, 16, 4] * 2
    record = json.loads((tmp_path / "heldout").read_text())
    observed = [item["heldout_effect"] for item in record["iterations"]
                if item["heldout_effect"] is not None]
    assert len(observed) == 2
    assert all(item["additional_full_chain_scores_per_environment"] == 10 for item in observed)


def test_physical_capture_records_chains_without_changing_solver(tmp_path):
    results = []
    for cls, name in ((NativePrefixHeldoutEffect, "heldout"),
                      (NativePrefixHeldoutPhysical, "physical")):
        solver = cls(model=Model(), batch_size=2, num_samples=20, n_steps=3,
                     topk=3, device="cpu", seed=5, audit_path=tmp_path / name)
        solver.configure(action_space=Box(-1, 1, shape=(3, 2), dtype=np.float32),
                         n_envs=3, config=SimpleNamespace(action_block=1, horizon=4))
        results.append(solver.solve({"x": torch.zeros(3, 1)})["actions"])
    assert torch.equal(*results)
    record = json.loads((tmp_path / "physical").read_text())
    heldout = next(item["heldout_effect"] for item in record["iterations"]
                   if item["heldout_effect"] is not None)
    assert torch.tensor(heldout["parent_actions"]).shape == (2, 2, 4, 2)
    assert torch.tensor(heldout["heldout_b_actions"]).shape == (2, 2, 4, 4, 2)
    assert torch.equal(torch.tensor(record["returned_actions"]), results[1])


def test_independent_tail_optimization_is_not_in_main_selection(tmp_path):
    results = []
    for cls, name, extra in (
        (NativePrefixConditional, "normal", {}),
        (NativePrefixHeldoutOptimization, "optimized", {"action_mean": [0., 0.],
                                                        "action_scale": [0.01, 0.01]}),
    ):
        solver = cls(model=Model(), batch_size=2, num_samples=20, n_steps=3,
                     topk=3, device="cpu", seed=5, audit_path=tmp_path / name, **extra)
        solver.configure(action_space=Box(-1, 1, shape=(3, 2), dtype=np.float32),
                         n_envs=3, config=SimpleNamespace(action_block=1, horizon=4))
        results.append(solver.solve({"x": torch.zeros(3, 1)})["actions"])
    assert torch.equal(*results)
    record = json.loads((tmp_path / "optimized").read_text())
    observed = next(x["heldout_effect"] for x in record["iterations"]
                    if x["heldout_effect"] is not None)
    parent = torch.tensor(observed["parent_actions"])
    optimized = torch.tensor(observed["optimized_actions"])
    assert torch.equal(parent[:, :, :2], optimized[:, :, :2])
    assert observed["additional_full_chain_scores_per_environment"] == 192
    assert torch.all(torch.tensor(observed["optimized_cost"]) <= torch.tensor(observed["parent_cost"]))
