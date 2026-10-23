from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.decision_benchmark import build_pointmaze_temporal_ood_v6_matched_panel as panel
from experiments.decision_benchmark import run_pointmaze_c3_stratified_repair_v24_cuda as v24
from experiments.gdd_wm import preflight_pointmaze_c3_stratified_repair_v24 as preflight


def test_v24_freezes_new_ranges_motion_route_and_stratum_calibration() -> None:
    assert preflight.SCHEMA == panel.C3_STRATIFIED_REPAIR_V24_PREFLIGHT_SCHEMA
    assert (preflight.HIGH_RANGE, preflight.DIRECT_RANGE) == ((32456, 32904), (32904, 33288))
    assert v24.POLICY["direction"] == "recurrent_when_high"
    assert v24.POLICY["threshold"] == .02
    source = inspect.getsource(v24.main)
    assert "_calibration_gate(calibration_records)" in source
    assert "test_outcomes_opened\": False" in source
    assert "certificate_kind=\"regret_budget\"" in source


def test_v24_motion_strata_are_root_constant_and_observable() -> None:
    history_low = np.zeros((8, 4), dtype=np.float32).tolist()
    history_high = np.zeros((8, 4), dtype=np.float32)
    history_high[-1, 0] = .1
    rows = [
        {"instance_id": "low", "candidate_id": candidate, "source_history_visual_state": history_low}
        for candidate in range(2)
    ] + [
        {"instance_id": "high", "candidate_id": candidate, "source_history_visual_state": history_high.tolist()}
        for candidate in range(2)
    ]
    assert v24._strata(rows) == {"low": "direct_zero_motion", "high": "high_motion"}


def test_v24_build_binds_new_frozen_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for relative in (
        "data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar",
        "results/pointmaze_same_backbone_full_support_20261013/checkpoints/jepa_full_mlp/seed_8411/stage_005.pt",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "external" / "local_dinov2").mkdir(parents=True)
    captured: dict[str, object] = {}
    monkeypatch.setattr(v24, "ROOT", tmp_path)
    monkeypatch.setattr(v24, "OUTPUT", tmp_path / "results" / "v24")
    monkeypatch.setattr(v24, "DATASET", tmp_path / "data" / "v75")
    monkeypatch.setattr(v24, "PREFLIGHT", tmp_path / "results" / "v24_preflight.json")
    monkeypatch.setattr(v24.builder, "build_panel", lambda *args, **kwargs: captured.update(kwargs) or {"built": True})
    assert v24._build(tuple()) == {"built": True}
    assert captured["frozen_preflight_ledger"] == v24.PREFLIGHT
    assert captured["candidate_generalization_regime"] == "shared_candidate_in_support"
    assert captured["candidate_design"] == "velocity_matched_braking_pairs"


def test_v24_host_wrapper_persists_a_diagnosable_runtime_log() -> None:
    wrapper = Path(v24.__file__).with_name("run_pointmaze_c3_stratified_repair_v24_host.sh")
    source = wrapper.read_text(encoding="utf-8")
    assert "host_runtime.log" in source
    assert "exec >>" in source
    assert "gpu_preflight --device cuda --recurrent-cell-smoke" in source
