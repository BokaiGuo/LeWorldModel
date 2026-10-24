from __future__ import annotations

from pathlib import Path

import torch

from experiments.decision_benchmark.build_pointmaze_velocity_matched_state_pool import (
    SOURCE_FRAME,
    _require_unique_source_observations,
    build_pool,
)


def test_zero_velocity_pool_emits_independent_source_observations(tmp_path: Path) -> None:
    """Regression for the former +/- zero-velocity duplicate-root bug."""

    source = tmp_path / "source"
    source.mkdir()
    (source / "obses").mkdir()
    torch.save(torch.zeros((1, 100, 4), dtype=torch.float32), source / "states.pth")
    torch.save(torch.zeros((1, 100, 2), dtype=torch.float32), source / "actions.pth")
    output = tmp_path / "direct"
    report = build_pool(
        source,
        output,
        pair_count=3,
        seed=31,
        velocity_magnitude=0.0,
        target_goal=(0.8, 0.8),
        target_radius=0.30,
        history_qpos_bounds=(0.05, 3.95),
        regime="zero_velocity",
    )
    states = torch.load(output / "states.pth", map_location="cpu", weights_only=True)
    generated = [row.numpy() for row in states[1:]]
    assert report["new_episode_count"] == 6
    assert report["unique_source_signature_count"] == 6
    assert _require_unique_source_observations(generated) == 6
    assert all(float(row[SOURCE_FRAME, 2:].norm()) == 0.0 for row in states[1:])
