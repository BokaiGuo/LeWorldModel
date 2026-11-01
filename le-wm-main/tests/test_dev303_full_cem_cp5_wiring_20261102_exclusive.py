import numpy as np
import pytest
import torch

from experiments.spectral_reachability import dev303_full_cem_cp5_wiring_20261102_exclusive as m


def test_frozen_dev303_task_and_budget():
    task = m.load_task()
    assert task["environment"] == "swm/PushT-v1"
    assert m.budget(3)["predict_calls"] == 195
    assert m.budget(2)["predict_calls"] == 130
    assert m.contract()["feedback_steps"] == [5]
    assert m.contract()["physical_execution_authorized"] is False


def test_complete_horizon_search_and_replay():
    seen = []
    def score(x):
        seen.append(x.copy())
        return float(np.square(x).sum())
    result = m.optimize(3, score, seed=9)
    assert len(seen) == 65
    assert all(x.shape == (3, 5, 2) for x in seen)
    assert result["selected_index"] == 0
    np.testing.assert_array_equal(result["actions"], np.zeros((3, 5, 2)))
    np.testing.assert_array_equal(seen[-1], result["actions"])


def test_cp5_replanning_uses_actual_receipt_and_issues_complete_tail():
    calls = []
    def observe(step, actions):
        calls.append(("observe", step, actions.copy()))
        return {"step": step, "actual_actions": actions.copy(), "rgb_sha256": "a"*64}
    def score(obs, plan):
        calls.append(("score", obs["step"], len(plan)))
        return float(np.square(plan).sum())
    def issue(step, plan):
        calls.append(("issue", step, len(plan)))
        return {"executed_actions": plan.copy(), "stopped": False}
    result = m.run_callbacks(observe, score, issue, seed=3)
    assert [x[:3] for x in calls if x[0] == "issue"] == [
        ("issue", 0, 5), ("issue", 5, 10)]
    assert [x[1] for x in calls if x[0] == "observe"] == [0, 5]
    assert [row["horizon"] for row in result["nodes"]] == [3, 2]
    assert result["fairness_proven"] is False


@pytest.mark.parametrize("failure", ["mismatch", "short", "missing_rgb"])
def test_bad_feedback_or_action_receipt_stops_without_fallback(failure):
    issued = []
    def observe(step, actions):
        data = {"step": step, "actual_actions": actions.copy(), "rgb_sha256": "a"*64}
        if step == 5 and failure == "missing_rgb":
            data["rgb_sha256"] = ""
        return data
    def issue(step, plan):
        issued.append(step)
        actual = plan.copy()
        if failure == "mismatch":
            actual[0, 0] += 0.5
        if failure == "short":
            actual = actual[:-1]
        return {"executed_actions": actual, "stopped": False}
    with pytest.raises(ValueError):
        m.run_callbacks(observe, lambda obs, plan: float(np.square(plan).sum()), issue, seed=1)
    assert issued == [0]


def test_torch_scorer_uses_real_leaf_calls_and_actual_normalization():
    class Leaf:
        def action_encode(self, actions, *, phase):
            return actions
        def predict(self, states, actions, *, phase):
            out = states.clone()
            out[:, -1, 0] = actions[:, -1].sum(dim=-1)
            return out
    scorer = m.TorchScorer(Leaf(), {
        "states": torch.zeros(1, 3, 2), "history": torch.zeros(1, 2, 10),
        "goal": torch.zeros(1, 2), "mean": torch.zeros(2),
        "std": torch.ones(2), "matrix": torch.eye(2),
    })
    score = scorer(np.ones((2, 5, 2), np.float32))
    assert np.isfinite(score)
    assert scorer.attempted == scorer.returned == {"action_encode": 2, "predict": 2}


def test_nonfinite_replay_and_unsupported_horizon_fail_closed():
    with pytest.raises(ValueError):
        m.optimize(3, lambda _: float("nan"), seed=1)
    with pytest.raises(ValueError):
        m.optimize(4, lambda _: 0., seed=1)


def test_model_callback_wiring_charges_each_node_and_retains_actual_counts():
    class Leaf:
        def action_encode(self, actions, *, phase):
            return actions
        def predict(self, states, actions, *, phase):
            return states
    ctx = {"states": torch.zeros(1, 3, 2), "history": torch.zeros(1, 2, 10),
           "goal": torch.zeros(1, 2), "mean": torch.zeros(2),
           "std": torch.ones(2), "matrix": torch.eye(2)}
    events = []
    def observe(step, actions):
        return {"step": step, "actual_actions": actions, "rgb_sha256": "a"*64, "context": ctx}
    result = m.run_model_callbacks(
        Leaf(), observe, lambda step, actions: {"executed_actions": actions, "stopped": False},
        seed=3, emit=events.append)
    assert [row["attempted"]["predict"] for row in result["model_usage"]] == [195, 130]
    assert all(row["attempted"] == row["returned"] for row in result["model_usage"])
    assert len([e for e in events if e["type"] == "node_model_usage"]) == 2


def test_model_failure_is_counted_without_dispatch():
    class Broken:
        def action_encode(self, actions, *, phase):
            raise RuntimeError("injected")
    ctx = {"states": torch.zeros(1, 3, 2), "history": torch.zeros(1, 2, 10),
           "goal": torch.zeros(1, 2), "mean": torch.zeros(2),
           "std": torch.ones(2), "matrix": torch.eye(2)}
    events = []
    with pytest.raises(RuntimeError, match="injected"):
        m.run_model_callbacks(Broken(), lambda step, actions: {
            "step": step, "actual_actions": actions, "rgb_sha256": "a"*64, "context": ctx},
            lambda *_: pytest.fail("must not dispatch"), seed=3, emit=events.append)
    usage = next(e for e in events if e["type"] == "node_model_usage")
    assert usage["attempted"]["action_encode"] == 1
    assert usage["returned"]["action_encode"] == 0
