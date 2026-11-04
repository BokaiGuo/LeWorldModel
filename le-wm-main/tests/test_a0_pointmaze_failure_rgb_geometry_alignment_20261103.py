"""Recheck sealed geometry calculations; this is not a new physical rollout."""
import json

import numpy as np

from experiments.a0_pointmaze_failure_rgb_geometry_alignment_20261103 import (
    ARRAYS, CAMERA, OLD, ORACLE, OUTPUT, PLANS, PROTOCOL, RGB, pairwise_ranking,
    project, sha256_file,
)


def test_saved_geometry_alignment():
    report = json.loads(OUTPUT.read_text())
    camera = json.loads(CAMERA.read_text())["metadata"]
    offset = json.loads(OLD.read_text())["body_offset_from_static_scene"]
    plans = {row["root"]: row for row in json.loads(PLANS.read_text())["rows"]}
    assert sha256_file(ARRAYS) == report["arrays_sha256"]
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert report["image_count"] == 624
    assert len(report["rows"]) == 12
    with np.load(ARRAYS, allow_pickle=False) as saved, \
            np.load(RGB.with_suffix(".npz"), allow_pickle=False) as rgb, \
            np.load(ORACLE.with_suffix(".npz"), allow_pickle=False) as oracle, \
            np.load(PLANS.with_suffix(".npz"), allow_pickle=False) as inputs:
        cursor = 0
        for row in report["rows"]:
            key = f"r{row['root']}_{row['method']}_{row['phase']}"
            goal, error = project(inputs[f"r{row['root']}_goal_rgb"], camera, offset)
            assert error == row["goal_detection_error"] is None
            positions, failures = [], []
            for index, image in enumerate(rgb["rgb10"][cursor:cursor + row["n"]]):
                position, error = project(image, camera, offset)
                positions.append(position if error is None else [np.nan, np.nan])
                if error is not None:
                    failures.append({"index": index, "error": error})
            cursor += row["n"]
            assert failures == row["image_detection_errors"] == []
            np.testing.assert_array_equal(saved[f"{key}_rgb_positions"], positions)
            distance = np.linalg.norm(np.asarray(positions) - goal, axis=1)
            np.testing.assert_array_equal(saved[f"{key}_rgb_distances"], distance)
            physical = oracle[f"{key}_distances"]
            assert row["rgb_distance_vs_physical"] == pairwise_ranking(distance, physical)
            assert row["latent_vs_rgb_distance"] == pairwise_ranking(rgb[f"{key}_real_cost"], distance)
            choice = int(np.argmin(distance))
            assert row["rgb_oracle_choice"] == choice
            assert row["rgb_oracle_physical_distance"] == physical[choice]
            goal_eval = np.asarray(plans[row["root"]]["goal_position_eval_only"])
            assert row["goal_position_error"] == np.linalg.norm(goal - goal_eval)
            errors = np.linalg.norm(np.asarray(positions) - oracle[f"{key}_endpoints"][:, :2], axis=1)
            assert row["terminal_position_error_mean"] == errors.mean()
            assert row["terminal_position_error_max"] == errors.max()
        assert cursor == 624
