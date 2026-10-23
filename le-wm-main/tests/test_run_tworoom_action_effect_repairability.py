from __future__ import annotations

import numpy as np
import pytest

from experiments.run_tworoom_action_effect_repairability import _effect_scores, _feature_views, _require_fresh_output, _root_split


def test_root_split_is_disjoint_and_complete() -> None:
    split = _root_split(range(100, 140), seed=7)
    assert set(split.values()) == {"train", "calibration", "outcome"}
    assert len(split) == 40
    assert sum(value == "train" for value in split.values()) == 24
    assert sum(value == "calibration" for value in split.values()) == 8
    assert sum(value == "outcome" for value in split.values()) == 8


def test_effect_feature_views_are_action_conditioned_and_dynamics_ablation_is_smaller() -> None:
    source = np.zeros((3, 4), dtype=np.float32)
    predicted = np.ones((3, 4), dtype=np.float32)
    action = np.ones((3, 2), dtype=np.float32)
    views = _feature_views(source, predicted, action)
    assert views["effect_adapter"].shape == (3, 10)
    assert views["no_dynamics"].shape == (3, 6)
    assert views["action_only"].shape == (3, 2)
    assert np.allclose(views["effect_adapter"][:, 4:8], 1.0)


def test_effect_scores_rank_goal_directed_displacement() -> None:
    scores = _effect_scores(np.array([0.0, 0.0]), np.array([3.0, 0.0]), np.array([[3.0, 0.0], [-1.0, 0.0]]))
    assert int(np.argmin(scores)) == 0


def test_root_split_rejects_too_few_roots() -> None:
    with pytest.raises(ValueError, match="at least ten"):
        _root_split(range(9), seed=1)


def test_tworoom_repairability_refuses_to_replace_existing_evidence_panel(tmp_path) -> None:
    output = tmp_path / "completed-panel"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _require_fresh_output(output)
    _require_fresh_output(tmp_path / "fresh-panel")
