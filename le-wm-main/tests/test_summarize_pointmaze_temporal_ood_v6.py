from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.decision_benchmark.summarize_pointmaze_temporal_ood_v6 import summarize


def _report(path: Path, root: str, recurrent: float, action_only: float) -> Path:
    value = {
        "outcome_root_ids": [root], "active_policy": "action_only_fallback",
        "recurrent": {"roots": {root: {"regret": recurrent}}},
        "action_only": {"roots": {root: {"regret": action_only}}},
    }
    path.write_text(json.dumps(value))
    return path


def test_summary_uses_root_paired_gain_and_keeps_a_declared_regime(tmp_path: Path) -> None:
    first = _report(tmp_path / "first.json", "root-1", .1, .3)
    second = _report(tmp_path / "second.json", "root-2", .4, .3)
    result = summarize([first, second], regime="numeric_ood", samples=100)
    assert result["candidate_generalization_regime"] == "numeric_ood"
    assert result["pooled_root_level_paired_recurrent_over_action_only"]["mean_gain"] == pytest.approx(.05)
    assert result["outcome_root_count"] == 2


def test_summary_rejects_reused_outcome_roots(tmp_path: Path) -> None:
    first = _report(tmp_path / "first.json", "root", .1, .3)
    second = _report(tmp_path / "second.json", "root", .2, .4)
    with pytest.raises(ValueError, match="overlap"):
        summarize([first, second], regime="shared_candidate_in_support", samples=100)
