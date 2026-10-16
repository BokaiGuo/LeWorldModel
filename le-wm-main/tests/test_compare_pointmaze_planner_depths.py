from __future__ import annotations

import json

from experiments.compare_pointmaze_planner_depths import compare


def _report(path, *, successes, distances, blocks, depth, beam):
    rows = []
    for episode, (success, distance, count) in enumerate(
        zip(successes, distances, blocks), start=10
    ):
        rows.append({
            "episode_id": episode,
            "method": "perfect_dynamics_mpc",
            "success": success,
            "final_distance": distance,
            "blocks_executed": count,
        })
    path.write_text(json.dumps({
        "protocol": {"action_bank": {"oracle_depth": depth, "oracle_beam_width": beam}},
        "rows": rows,
    }))


def test_depth_comparison_is_episode_paired(tmp_path) -> None:
    shallow = tmp_path / "shallow.json"
    deep = tmp_path / "deep.json"
    _report(
        shallow,
        successes=[False, True],
        distances=[2.0, 1.0],
        blocks=[4, 5],
        depth=1,
        beam=32,
    )
    _report(
        deep,
        successes=[True, True],
        distances=[1.0, 0.5],
        blocks=[5, 5],
        depth=4,
        beam=8,
    )
    result = compare(shallow, deep, draws=1000, seed=7)
    assert result["paired"]["deep_minus_shallow_success"]["mean"] == 0.5
    assert result["paired"]["shallow_minus_deep_final_distance"]["mean"] == 0.75
    assert result["paired"]["discordant_success"] == {
        "deep_only": 1,
        "shallow_only": 0,
    }
