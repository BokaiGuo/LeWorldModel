from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.decision_benchmark import run_pointmaze_c3_regret_budget_v23_cuda as v23
from experiments.gdd_wm import preflight_pointmaze_c3_regret_budget_v23 as preflight


def test_v23_predeclares_new_roots_and_a_nonzero_regret_budget() -> None:
    assert preflight.SCHEMA == panel.C3_REGRET_BUDGET_V23_PREFLIGHT_SCHEMA
    assert (preflight.HIGH_RANGE, preflight.DIRECT_RANGE) == ((30664, 31112), (31112, 31496))
    assert preflight.REGRET_BUDGET == 0.12
    assert v23.REGRET_BUDGET == preflight.REGRET_BUDGET


def test_v23_uses_independent_fit_and_regret_certificate_not_v22_outcome_selection() -> None:
    source = inspect.getsource(v23.main)
    assert "calibrate_fixed_policy" in source
    assert "certificate_kind=\"regret_budget\"" in source
    assert "test_outcomes_opened\": False" in source
    assert "outcome_path.exists()" in source
    assert "require_cuda_recurrent_cell(\"cuda\")" in source


def test_v23_binds_its_frozen_ledger_and_shared_support_panel_at_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the runner-to-builder interface without materializing CUDA features."""

    for relative in (
        "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar",
        "results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "external" / "local_dinov2").mkdir(parents=True)
    captured: dict[str, object] = {}
    monkeypatch.setattr(v23, "ROOT", tmp_path)
    monkeypatch.setattr(v23, "OUTPUT", tmp_path / "results" / "v23")
    monkeypatch.setattr(v23, "DATASET", tmp_path / "data" / "v72")
    monkeypatch.setattr(v23, "PREFLIGHT", tmp_path / "results" / "v23_preflight.json")
    monkeypatch.setattr(v23.builder, "build_panel", lambda *args, **kwargs: captured.update(kwargs) or {"built": True})

    assert v23._build(tuple()) == {"built": True}
    assert captured["candidate_generalization_regime"] == "shared_candidate_in_support"
    assert captured["candidate_design"] == "velocity_matched_braking_pairs"
    assert captured["frozen_preflight_ledger"] == v23.PREFLIGHT
    assert captured["split_counts"] == {"train": 256, "calibration": 256, "outcome": 256}
