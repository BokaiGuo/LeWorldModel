from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from stable_worldmodel.solver.cem import CEMSolver

from experiments.pc_native_internal_response_20261105_codex import NativeInternalResponse


class Model(torch.nn.Module):
    def __init__(self, invalid=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []
        self.invalid = invalid

    def get_cost(self, info, actions):
        self.calls.append(actions.clone())
        cost = ((actions - info["target"]) ** 2).sum((-1, -2))
        return cost * float("nan") if self.invalid else cost


def solver(cls, model, **kwargs):
    instance = cls(model=model, batch_size=2, num_samples=20, n_steps=3, topk=3,
                   device="cpu", seed=5, **kwargs)
    instance.configure(action_space=Box(-1, 1, shape=(3, 2), dtype=np.float32),
                       n_envs=3, config=SimpleNamespace(action_block=1, horizon=2))
    return instance


def test_native_mode_exactly_matches_base():
    info = {"target": torch.ones(3, 1, 1)}
    native = solver(CEMSolver, Model()).solve(info)
    tested = solver(NativeInternalResponse, Model(), mode="native").solve(info)
    assert torch.equal(native["actions"], tested["actions"])
    assert native["costs"] == tested["costs"]


@pytest.mark.parametrize("mode", ["response", "random"])
def test_budget_batch_and_new_combinations(mode):
    model = Model()
    result = solver(NativeInternalResponse, model, mode=mode).solve(
        {"target": torch.ones(3, 1, 1)})
    assert result["actions"].shape == (3, 2, 2)
    assert [x.shape[1] for x in model.calls] == [20, 16, 4, 16, 4] * 2
    assert all(not torch.equal(p, q) for p in model.calls[2][0] for q in model.calls[1][0, -8:])


def test_nonfinite_cost_fails_closed():
    with pytest.raises(ValueError, match="invalid candidate costs"):
        solver(NativeInternalResponse, Model(invalid=True)).solve(
            {"target": torch.zeros(3, 1, 1)})


def test_audit_budget_and_overwrite(tmp_path):
    path = tmp_path / "audit.jsonl"
    solver(NativeInternalResponse, Model(), audit_path=path).solve(
        {"target": torch.zeros(3, 1, 1)})
    record = json.loads(path.read_text())
    assert len(record["iterations"]) == 6
    assert all(item["scored_per_environment"] == 20 for item in record["iterations"])
    assert all(sum(item["probe_elites"]) == 0 for item in record["iterations"]
               if item["step"] == 0)
    with pytest.raises(FileExistsError):
        solver(NativeInternalResponse, Model(), audit_path=path)


def test_zero_response_is_finite():
    class Flat(Model):
        def get_cost(self, info, actions):
            self.calls.append(actions.clone())
            return torch.zeros(actions.shape[:2])

    model = Flat()
    result = solver(NativeInternalResponse, model).solve({"target": torch.zeros(3, 1, 1)})
    assert torch.isfinite(result["actions"]).all()
    assert all(torch.isfinite(call).all() for call in model.calls)


def test_response_and_random_share_initial_population_and_probes():
    models = [Model(), Model()]
    for mode, model in zip(("response", "random"), models):
        solver(NativeInternalResponse, model, mode=mode).solve(
            {"target": torch.ones(3, 1, 1)})
    assert torch.equal(models[0].calls[0], models[1].calls[0])
    assert torch.equal(models[0].calls[1], models[1].calls[1])
    assert not torch.equal(models[0].calls[2], models[1].calls[2])
