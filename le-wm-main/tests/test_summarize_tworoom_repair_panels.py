from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.gdd_wm.summarize_tworoom_repair_panels import summarize


def _planning(path: Path, seeds: list[int], frozen: list[int], hybrid: list[int]) -> None:
    path.write_text(json.dumps({
        "protocol": {"checkpoint": "fixed", "candidate_bank": "fixed", "role": path.stem.split("-")[-1]},
        "paired": {"seed_order": seeds},
        "frozen_lewm": {"episode_successes": frozen, "seeds": seeds},
        "hybrid_rgb_fallback": {"episode_successes": hybrid, "seeds": seeds},
    }), encoding="utf-8")


def _panel(path: Path, baseline: Path, intervention: Path, frozen: list[int], hybrid: list[int]) -> None:
    path.write_text(json.dumps({
        "schema": "gdd_wm_paired_tworoom_repair_consequence.v0.1",
        "protocol": {
            "baseline": str(baseline), "intervention": str(intervention),
            "intervention_key": "hybrid_rgb_fallback", "outcome_used_for_selection": False,
            "same_seed_panel": True,
        },
        "n": len(frozen),
        "success": {
            "baseline": sum(frozen), "intervention": sum(hybrid),
            "baseline_only": sum(left == 1 and right == 0 for left, right in zip(frozen, hybrid)),
            "intervention_only": sum(left == 0 and right == 1 for left, right in zip(frozen, hybrid)),
        },
        "runtime_diagnostics": {},
    }), encoding="utf-8")


def _valid_panel(tmp_path: Path, name: str, seeds: list[int], frozen: list[int], hybrid: list[int]) -> Path:
    baseline, intervention, panel = (tmp_path / f"{name}-{suffix}.json" for suffix in ("base", "intervention", "panel"))
    _planning(baseline, seeds, frozen, hybrid)
    _planning(intervention, seeds, frozen, hybrid)
    _panel(panel, baseline, intervention, frozen, hybrid)
    return panel


def test_summary_requires_trace_backed_disjoint_paired_panels(tmp_path: Path) -> None:
    left = _valid_panel(tmp_path, "left", [1, 2], [1, 0], [1, 1])
    right = _valid_panel(tmp_path, "right", [3, 4], [0, 1], [1, 1])
    result = summarize([left, right], tmp_path / "summary.json")
    assert result["combined"]["n"] == 4
    assert result["combined"]["success_delta"] == 2
    assert result["protocol"]["trace_backed_success_counts"] is True


def test_summary_rejects_duplicate_or_overlapping_panels(tmp_path: Path) -> None:
    left = _valid_panel(tmp_path, "left", [1, 2], [1, 0], [1, 1])
    overlapping = _valid_panel(tmp_path, "overlap", [2, 3], [0, 1], [1, 1])
    with pytest.raises(ValueError, match="duplicate"):
        summarize([left, left], tmp_path / "duplicate.json")
    with pytest.raises(ValueError, match="overlap"):
        summarize([left, overlapping], tmp_path / "overlap.json")


def test_summary_rejects_cross_panel_protocol_drift(tmp_path: Path) -> None:
    left = _valid_panel(tmp_path, "left", [1, 2], [1, 0], [1, 1])
    right = _valid_panel(tmp_path, "right", [3, 4], [0, 1], [1, 1])
    trace = tmp_path / "right-base.json"
    payload = json.loads(trace.read_text(encoding="utf-8"))
    payload["protocol"]["candidate_bank"] = "changed"
    trace.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline planning protocol differs"):
        summarize([left, right], tmp_path / "drift.json")
