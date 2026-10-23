from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.decision_benchmark.select_pointmaze_obstacle_stratified_roots import select


def _manifest(path: Path) -> Path:
    path.write_text(json.dumps({
        "schema": "pointmaze_geometry_stratified_obstacle_extension.v1",
        "base_root_count": 10,
        "new_episode_count": 12,
        "static_start_cells_qpos": [[.8, .8], [2.8, .8]],
        "sampled_start_cell_indices": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }))
    return path


def test_selects_equal_static_start_cell_allocation_without_replacement(tmp_path: Path) -> None:
    result = select(_manifest(tmp_path / "manifest.json"), excluded_roots=[10, 11], per_cell=3, seed=7)
    assert result["selected_root_count"] == 6
    assert all(len(values) == 3 for values in result["selected_roots_by_start_cell_index"].values())
    assert not ({10, 11} & set(result["selected_roots"]))


def test_rejects_insufficient_unused_stratum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="insufficient"):
        select(_manifest(tmp_path / "manifest.json"), excluded_roots=[10, 12, 14, 16, 18], per_cell=2, seed=7)
