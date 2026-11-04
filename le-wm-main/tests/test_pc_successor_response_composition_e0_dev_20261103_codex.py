import json

import numpy as np
import pytest

from experiments import pc_successor_response_composition_e0_dev_20261103_codex as experiment


def test_wall_endpoint_matches_original_step_index_and_rejects_bad_actions():
    rng = np.random.default_rng(10)
    task = experiment.wall.tasks()[0]
    for length in (5, 10, 15):
        actions = rng.uniform(-1, 1, (3, length, 2))
        padded = np.pad(actions, ((0, 0), (0, 16 - length), (0, 0))) * experiment.wall.HIGH
        trace, _ = experiment.wall.rollout(task["start"], padded)
        np.testing.assert_allclose(experiment.endpoints(task, actions), trace[:, length, :2])
    with pytest.raises(ValueError):
        experiment.endpoints(task, np.full((1, 5, 2), 1.1))


@pytest.mark.parametrize("arm", experiment.ARMS)
@pytest.mark.parametrize("budget", experiment.BUDGETS)
def test_candidate_composition_budget_and_trace(arm, budget):
    row = experiment.plan_one(experiment.wall.tasks()[0], 0, budget, arm)
    assert row["counts"]["prediction_token_samples"] <= budget
    assert sum(batch["tokens"] for batch in row["trace"]) == row["counts"]["prediction_token_samples"]
    assert np.asarray(row["selected"]["chain"]).shape == (15, 2)
    if arm.startswith("successor_"):
        assert len(row["bank"]) == 9
        for item in row["bank"]:
            assert np.asarray(item["prefix"]).shape == (10, 2)
    if arm == "successor_response":
        assert len(row["successor_response"]) == 3
        assert row["structure_counts"]["prediction_samples"] == 168
        assert row["structure_counts"]["svd"] == 8
        for item in row["bank"]:
            name = item["name"].split("/")[0]
            prefix = next(x["prefix"] for x in row["first_bank"] if x["name"] == name)
            np.testing.assert_array_equal(np.asarray(item["prefix"])[:5], prefix)
    if row["incumbent"] is not None:
        assert row["selected"]["score"] <= row["incumbent"]["score"]


def test_sealed_selected_chain_intervention_and_accounting():
    if not experiment.INTERVENTION.exists():
        pytest.skip("development intervention not yet sealed")
    source = json.loads(experiment.PLANS.read_text())
    audit = json.loads(experiment.AUDIT.read_text())
    data = json.loads(experiment.INTERVENTION.read_text())
    assert len(source["rows"]) == len(audit["rows"]) == 96
    assert data["plans_sha256"] == audit["plans_sha256"] == experiment.sha(experiment.PLANS)
    assert data["audit_sha256"] == experiment.sha(experiment.AUDIT)
    assert source["dependencies"] == experiment.hashes()
    originals = {(x["root"], x["budget"]): x for x in source["rows"] if x["arm"] == "successor_response"}
    assert len(data["rows"]) == 16
    for row in data["rows"]:
        original = originals[row["root"], row["budget"]]
        chain = np.asarray(row["chain"])
        np.testing.assert_array_equal(chain, original["selected"]["chain"])
        greedy = original["first_bank"][int(np.argmin(original["first_scores"]))]["prefix"]
        np.testing.assert_array_equal(row["controls"]["greedy"]["prefix"], greedy)
        goal = np.asarray(original["task"]["goal"])
        for control in row["controls"].values():
            prefix = np.asarray(control["prefix"])
            full = np.concatenate((prefix, chain[5:]))[None]
            distance = np.linalg.norm(experiment.endpoints(original["task"], full)[0] - goal)
            np.testing.assert_allclose(distance, control["common_suffix_distances"][-1], atol=1e-12)
            np.testing.assert_allclose(control["active_changes"],
                                       np.asarray(control["common_suffix_distances"])
                                       - control["common_suffix_distances"][0])
            assert len(control["own_scores"]) == 120
            assert control["own_optimized_distance"] == min(control["own_scores"])
        assert row["limited_witness"] == (
            row["nonprogress"] and row["active_response"] and row["own_plan_advantage"]
        )
