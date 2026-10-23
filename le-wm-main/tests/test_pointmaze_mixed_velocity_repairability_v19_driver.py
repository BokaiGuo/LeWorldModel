from __future__ import annotations

import inspect

from experiments.decision_benchmark import run_pointmaze_mixed_velocity_repairability_v19_cuda as v19


def test_v19_freezes_a_visual_motion_route_before_calibration_and_outcome() -> None:
    assert v19.ROOT_RANGE == (26_376, 27_208)
    assert v19.SPLIT_COUNTS == {"train": 256, "calibration": 256, "outcome": 256}
    assert v19.MOTION_POLICY == {
        "kind": "history_energy_gate", "direction": "recurrent_when_high", "threshold": 0.02,
        "motion_source": "source_history_visual_state", "fixed_threshold": True,
    }
    source = inspect.getsource(v19._fixed_calibration)
    assert "hidden=32" in source
    assert "contrast_weight=1.0" in source
    assert "policy=MOTION_POLICY" in source


def test_v19_uses_the_host_gpu_runner_and_exact_static_ledger() -> None:
    source = inspect.getsource(v19.main)
    assert "runner.HISTORY_GATE = True" in source
    assert "runner.CLAIM_BOUNDARY = CLAIM_BOUNDARY" in source
    assert "runner.PREFLIGHT_MODULE = preflight" in source
    assert "runner._ensure_extension = _pool" in source


def test_v19_declares_its_c2_boundary_without_overclaiming_c3() -> None:
    assert "C2 mixed-mechanism prospective-repairability" in v19.CLAIM_BOUNDARY
    assert "Not C3 calibration" in v19.CLAIM_BOUNDARY
