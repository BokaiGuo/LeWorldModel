from __future__ import annotations

from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_counterfactual_endpoint_v6_cuda.sh"


def test_v6_endpoint_launcher_is_gpu_only_and_freezes_larger_certificate_protocol() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "nvidia-smi is unavailable; refusing CPU fallback" in text
    assert "experiments.gpu_preflight --device cuda" in text
    assert 'RUNNER="experiments.gdd_wm.run_pointmaze_counterfactual_endpoint_control"' in text
    assert 'STATIC_PREFLIGHT="experiments.gdd_wm.preflight_pointmaze_counterfactual_endpoint_v6"' in text
    assert 'FROZEN_PREFLIGHT="${ROOT}/results/pointmaze_counterfactual_endpoint_v6_static_preflight_20261021.json"' in text
    assert 'CONTRACT_AUDIT="experiments.gdd_wm.audit_pointmaze_counterfactual_endpoint_v6"' in text
    assert 'DISAGREEMENT_AUDIT="experiments.decision_benchmark.audit_counterfactual_arm_disagreement"' in text
    assert "--train-root-count 128 --calibration-root-count 192 --outcome-root-count 64" in text
    assert "--candidate-design goal_balanced_interaction_pairs --goal-qpos 0.8 0.8" in text
    assert "--root-id-min 9656 --root-id-max-exclusive 10296" in text
    assert "--root-id-min 9656 --root-id-max-exclusive 10296 --device cuda" in text
    assert "--ensemble-size 4" in text
    assert "--risk-alpha 0.1" in text
    assert "--minimum-summary-collision-pairs-per-root 8" in text
    assert "refusing to reuse a V6 panel or endpoint-control output path" in text
    assert "live prior-context discovery differs from the frozen V6 preflight ledger" in text
    assert "--frozen-preflight-ledger \"${FROZEN_PREFLIGHT}\"" in text
    assert "decision_aliasing_calibration_outcome.json" in text
    assert "cost_endpoint_disagreement_audit.json" in text
    assert "contract_audit.json" in text
