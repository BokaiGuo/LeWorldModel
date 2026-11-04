import copy

import pytest

from experiments.spectral_reachability.audit_pc_fd_saved_rgb_replay_20261102 import analyze, verify_goal_q


def fake_reports():
    replay_rows, physical_rows = [], []
    for seed in (20263160, 20263161):
        branches, truth = [], []
        for arm, prefix in (("selected", 4), ("parameter_zero", 0)):
            for tail in range(5):
                checkpoints = []
                trajectory = [{"rgb": {"sha256": str(i)}, "qpos_distance": float(i)} for i in range(16)]
                for step in (5, 10, 15):
                    checkpoints.append({"step": step, "rgb_sha256": str(step), "qpos_distance": float(step),
                        "observed_latent": [1.0, 2.0], "open_latent": [2.0, 2.0],
                        "refreshed_latent": [1.0, 2.0], "open_mse": 0.5, "refreshed_mse": 0.0,
                        "observed_goal_q": 1.0, "open_goal_q": 1.0, "refreshed_goal_q": 1.0})
                identity = {"arm": arm, "prefix_index": prefix, "tail_index": tail}
                branches.append({**identity, "status": "complete", "checkpoints": checkpoints})
                truth.append({**identity, "trajectory": trajectory})
        replay_rows.append({"seed": seed, "branches": branches})
        physical_rows.append({"seed": seed, "branches": truth})
    return ({"status": "complete", "frozen_parameters_unchanged": True,
             "parameter_hash_before": "abc", "parameter_hash_after": "abc", "rows": replay_rows},
            {"rows": physical_rows})


def test_valid_results_are_paired():
    replay, physical = fake_reports()
    result = analyze(replay, physical)
    assert result["roots"][0]["steps"][0]["refresh_strictly_lower_count"] == 10
    assert result["roots"][0]["observed_vs_qpos_reversals"] == 0


@pytest.mark.parametrize("kind", ["mse", "rgb", "missing", "model"])
def test_mismatches_fail_closed(kind):
    replay, physical = copy.deepcopy(fake_reports())
    if kind == "mse":
        replay["rows"][0]["branches"][0]["checkpoints"][0]["open_mse"] = 3.0
    elif kind == "rgb":
        replay["rows"][0]["branches"][0]["checkpoints"][0]["rgb_sha256"] = "bad"
    elif kind == "missing":
        replay["rows"][0]["branches"].pop()
    else:
        replay["parameter_hash_after"] = "bad"
    with pytest.raises(ValueError):
        analyze(replay, physical)


def test_q_is_independently_recomputed():
    replay, _ = fake_reports()
    metric = {"rows": []}
    for row in replay["rows"]:
        row["goal_coordinate"] = [0.0, 0.0]
        metric["rows"].append({"seed": row["seed"], "goal_metric": {"gram": [[0.0, 0.0], [0.0, 0.0]]}})
        for branch in row["branches"]:
            for point in branch["checkpoints"]:
                for role in ("open", "refreshed", "observed"):
                    point[role + "_coordinate"] = [1.0, 1.0]
                    point[role + "_goal_q"] = 1.0
    assert verify_goal_q(replay, metric) == 180
    replay["rows"][0]["branches"][0]["checkpoints"][0]["open_goal_q"] = 1.1
    with pytest.raises(ValueError):
        verify_goal_q(replay, metric)
