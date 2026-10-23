from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.decision_benchmark.aggregate_pointmaze_v16_obstacle_challenge import aggregate


def _report(parent: Path, name: str, roots: list[int], *, design: str = "sealed") -> Path:
    rows = []
    for root in roots:
        for method, distance, candidate in (
            ("rgb_greedy", 1.0, 0),
            ("rgb_position_anchored_velocity", 0.5, 1),
            ("v14_full_state_dynamics_diagnostic", 1.1, 2),
            ("privileged_one_step_oracle", 0.25, 1),
        ):
            rows.append({"episode_id": root, "method": method, "final_distance": distance, "success": distance < .75, "trace": [{"block": 0, "selected_candidate_id": candidate}, {"block": 1, "selected_candidate_id": candidate}]})
    path = parent / name
    path.write_text(json.dumps({"schema": "pointmaze_v16_position_anchored_score.v1", "roots": roots, "rows": rows, "design_sha256": design, "selected_goal_qpos": [1.8, 2.8], "paired_position_anchored_over_rgb_greedy": {"mean_gain": .5, "lower": .1, "upper": .9}}))
    return path


def _manifest(path: Path, *, count: int = 20) -> Path:
    labels = [index % 2 for index in range(count)]
    path.write_text(json.dumps({"schema": "pointmaze_geometry_stratified_obstacle_extension.v1", "base_root_count": 100, "new_episode_count": count, "static_start_cells_qpos": [[.8, .8], [2.8, .8]], "sampled_start_cell_indices": labels}))
    return path


def test_aggregate_pools_roots_not_trace_blocks_and_reports_strata(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    first = _report(tmp_path, "first.json", list(range(100, 108)))
    second = _report(tmp_path, "second.json", list(range(108, 116)))
    result = aggregate([first, second], manifest, samples=100, seed=7)
    assert result["outcome_root_count"] == 16
    assert result["pooled_root_level_paired_position_anchored_over_rgb_greedy"]["mean_gain"] == pytest.approx(.5)
    assert sum(value["root_count"] for value in result["static_start_cell_strata"].values()) == 16
    assert result["oracle_action_match_diagnostics"]["rgb_position_anchored_velocity"]["first_decision_candidate_match_rate_to_oracle"] == pytest.approx(1.0)


def test_aggregate_rejects_overlapping_or_differently_sealed_panels(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    first = _report(tmp_path, "first.json", list(range(100, 108)))
    overlap = _report(tmp_path, "overlap.json", list(range(107, 115)))
    with pytest.raises(ValueError, match="overlap"):
        aggregate([first, overlap], manifest, samples=100)
    other = _report(tmp_path, "other.json", list(range(108, 116)), design="other-sealed")
    with pytest.raises(ValueError, match="same goal/design"):
        aggregate([first, other], manifest, samples=100)
