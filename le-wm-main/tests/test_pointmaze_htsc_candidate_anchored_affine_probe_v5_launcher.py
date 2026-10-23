from __future__ import annotations

from pathlib import Path


LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_pointmaze_htsc_candidate_anchored_affine_probe_v5_cuda.sh"
)


def test_v5_candidate_anchored_launcher_is_cuda_only_and_root_disjoint() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "experiments.gpu_preflight --device cuda --recurrent-smoke" in text
    assert "nvidia-smi" in text
    assert "optional nvidia-smi telemetry query fails" in text
    assert 'ACTIVE_METHOD="effect_adapter_rgb_online_affine_candidate_anchored_verified_local"' in text
    assert "V4_PROVENANCE=" in text
    assert "PREFLIGHT_OBSERVED_ROOTS=" in text
    assert "ADDITIONAL_EXCLUDED_ROOTS" in text
    assert ".current_split.held_out_test_roots[]" in text
    assert "ADDITIONAL_EXCLUDED_ARGS=(--additional-excluded-roots" in text
    assert "refusing to reuse immutable active-probe output path" in text
    assert "audit_pointmaze_active_probe_trace_contract" in text
    assert "audit_pointmaze_active_probe_identifiability" in text
