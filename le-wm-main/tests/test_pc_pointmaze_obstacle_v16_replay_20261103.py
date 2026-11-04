"""Validate sealed replay arithmetic and inputs without new physics."""
import json

import numpy as np
import torch

from experiments.pc_pointmaze_obstacle_v16_replay_20261103 import (
    DATA, OUTPUT, PANEL, PROTOCOL, SOURCE, sha256_file,
)


def test_obstacle_replay_saved_bindings():
    report = json.loads(OUTPUT.read_text())
    source, panel = (json.loads(p.read_text()) for p in (SOURCE, PANEL))
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert sha256_file(SOURCE) == report["source_sha256"]
    assert sha256_file(PANEL) == report["panel_sha256"]
    assert sha256_file(DATA / "states.pth") == report["states_sha256"]
    assert report["replayed_trajectories"] == 96
    assert not report["new_model_inference"]
    assert len({(x["root"], x["method"]) for x in report["rows"]}) == 96
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu").numpy().astype(np.float64)
    expected = {(r["episode_id"], r["method"]): r for r in source["rows"]}
    goal = np.asarray(source["selected_goal_qpos"])
    for row in report["rows"]:
        old = expected[row["root"], row["method"]]
        pos = np.asarray(row["positions"])
        assert row["root"] in panel["selected_roots"]
        assert row["blocks"] == old["blocks_executed"] == len(old["trace"])
        assert pos.shape == (row["blocks"] + 1, 4)
        np.testing.assert_array_equal(pos[0], states[row["root"], 40])
        distances = np.linalg.norm(pos[:, :2] - goal, axis=1)
        assert distances[0] == row["initial_distance"]
        assert distances[1] == row["after_one_distance"]
        assert distances[min(5, row["blocks"])] == row["after_five_distance"]
        assert distances[-1] == row["final_distance"]
        assert abs(distances[-1] - old["final_distance"]) < 1e-10
        assert row["reported_success"] == old["success"]
