from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from stable_worldmodel.solver.cem import CEMSolver

from experiments.pc_native_cem_incumbent_20261105_codex import NativeCEMIncumbent
from experiments.pc_native_finite_response_20261105_codex import NativeFiniteResponse
from experiments.pc_native_mean_selection_20261105_codex import NativeMeanSelection


class RecordingModel(torch.nn.Module):
    def __init__(self, invalid=False):
        super().__init__()
        self.register_parameter("anchor", torch.nn.Parameter(torch.zeros(())))
        self.calls = []
        self.invalid = invalid

    def get_cost(self, info, actions):
        costs = ((actions - info["target"]) ** 2).sum(dim=(-1, -2))
        if self.invalid:
            costs[:] = float("nan")
        self.calls.append((actions.clone(), costs.clone()))
        return costs


def make(solver_class, model, **kwargs):
    solver = solver_class(model=model, batch_size=2, num_samples=8, n_steps=3,
                          topk=3, device="cpu", seed=5, **kwargs)
    solver.configure(action_space=Box(-1, 1, shape=(3, 2), dtype=np.float32),
                     n_envs=3, config=SimpleNamespace(action_block=1, horizon=2))
    return solver


def info():
    return {"target": torch.tensor([[[0.1]], [[0.2]], [[0.3]]]),
            "env_name": ["a", "b", "c"]}


def test_disabled_matches_native():
    native = make(CEMSolver, RecordingModel()).solve(info())
    off = make(NativeCEMIncumbent, RecordingModel(), return_best=False).solve(info())
    assert torch.equal(native["actions"], off["actions"])
    assert native["costs"] == off["costs"]


def test_best_is_scored_and_no_extra_calls_across_batches():
    model = RecordingModel()
    result = make(NativeCEMIncumbent, model, return_best=True).solve(info())
    assert len(model.calls) == 6
    for batch, indices in enumerate((range(2), range(2, 3))):
        calls = model.calls[batch * 3:batch * 3 + 3]
        for local, global_index in enumerate(indices):
            expected = torch.cat([cost[local] for _, cost in calls]).min()
            assert result["best_scored_cost"][global_index] == expected
            assert any(torch.equal(result["actions"][global_index], candidates[local, j])
                       for candidates, _ in calls for j in range(candidates.shape[1]))


def test_fail_closed_on_nonfinite():
    with pytest.raises(ValueError, match="non-finite"):
        make(NativeCEMIncumbent, RecordingModel(invalid=True), return_best=True).solve(info())


def test_incumbent_is_reset_when_observation_changes():
    model = RecordingModel()
    solver = make(NativeCEMIncumbent, model, return_best=True)
    solver.solve(info())
    first_call_count = len(model.calls)
    changed = {"target": torch.full((3, 1, 1), 10.0)}
    second = solver.solve(changed)
    new_calls = model.calls[first_call_count:]
    assert len(new_calls) == 6
    for batch, indices in enumerate((range(2), range(2, 3))):
        calls = new_calls[batch * 3:batch * 3 + 3]
        for local, global_index in enumerate(indices):
            assert second["best_scored_cost"][global_index] == torch.cat(
                [cost[local] for _, cost in calls]
            ).min()


def test_response_and_random_share_probes_and_cost():
    models = [RecordingModel(), RecordingModel()]
    outcomes = [make(NativeFiniteResponse, model, mode=mode).solve(info())
                for model, mode in zip(models, ("response", "random"))]
    assert len(models[0].calls) == len(models[1].calls) == 10
    for index in (0, 1, 2, 3, 4, 5, 6, 8):
        assert torch.equal(models[0].calls[index][0], models[1].calls[index][0])
    assert not torch.equal(models[0].calls[7][0], models[1].calls[7][0])
    for model, result in zip(models, outcomes):
        for batch, indices in enumerate((range(2), range(2, 3))):
            calls = model.calls[batch * 3:batch * 3 + 3] + model.calls[6 + batch * 2:8 + batch * 2]
            for local, global_index in enumerate(indices):
                assert result["best_scored_cost"][global_index] == torch.cat(
                    [cost[local] for _, cost in calls]
                ).min()


def test_audit_records_scored_selection_and_refuses_overwrite(tmp_path):
    path = tmp_path / "audit.jsonl"
    result = make(NativeFiniteResponse, RecordingModel(), mode="response",
                  audit_path=str(path)).solve(info())
    record = json.loads(path.read_text())
    selected_actions = []
    for batch in record["batches"]:
        for actions, costs, idx in zip(batch["actions"], batch["costs"], batch["selected"]):
            assert costs[idx] == min(costs)
            selected_actions.append(actions[idx])
    assert torch.allclose(result["actions"], torch.tensor(selected_actions))
    with pytest.raises(FileExistsError):
        make(NativeFiniteResponse, RecordingModel(), audit_path=str(path))


def test_fixed_parent_and_directions_respond_to_goal(monkeypatch):
    def fixed_parent(self, info_dict, init_action=None):
        return {
            "actions": torch.zeros(3, 2, 2),
            "best_scored_cost": 4 * info_dict["target"].flatten().square(),
        }

    monkeypatch.setattr(NativeCEMIncumbent, "solve", fixed_parent)
    models = [RecordingModel(), RecordingModel()]
    targets = (torch.full((3, 1, 1), 1.0), torch.full((3, 1, 1), -1.0))
    for model, target in zip(models, targets):
        make(NativeFiniteResponse, model, mode="response").solve({"target": target})
    # First calls use exactly the same finite-difference probes; second calls
    # are target-dependent combined proposals.
    assert torch.equal(models[0].calls[0][0], models[1].calls[0][0])
    assert not torch.equal(models[0].calls[1][0], models[1].calls[1][0])


def test_mean_sample_joint_use_same_pool_and_score_mean():
    models = [RecordingModel() for _ in range(3)]
    outputs = [make(NativeMeanSelection, model, mode=mode).solve(info())
               for model, mode in zip(models, ("mean", "sample", "joint"))]
    assert all(len(model.calls) == 8 for model in models)
    for i in range(8):
        assert torch.equal(models[0].calls[i][0], models[1].calls[i][0])
        assert torch.equal(models[0].calls[i][0], models[2].calls[i][0])
    mean, sample, joint = outputs
    assert torch.equal(mean["actions"], sample["native_final_mean"])
    assert torch.equal(mean["actions"], joint["native_final_mean"])
    use_sample = joint["best_scored_cost"] < joint["native_mean_cost"]
    assert torch.equal(joint["actions"], torch.where(
        use_sample[:, None, None], sample["actions"], mean["actions"]))
    assert torch.equal(mean["native_mean_cost"], joint["native_mean_cost"])


def test_mean_selection_audit_and_no_overwrite(tmp_path):
    path = tmp_path / "mean.jsonl"
    result = make(NativeMeanSelection, RecordingModel(), mode="joint",
                  audit_path=str(path)).solve(info())
    record = json.loads(path.read_text())
    for i, chosen_sample in enumerate(record["choose_sample"]):
        expected = record["sample_actions"][i] if chosen_sample else record["mean_actions"][i]
        assert torch.allclose(result["actions"][i], torch.tensor(expected))
    with pytest.raises(FileExistsError):
        make(NativeMeanSelection, RecordingModel(), audit_path=str(path))
