from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "experiments" / "gdd_wm" / "run_pointmaze_planning_shaped_advantage_v7_cuda.sh"


def test_v7_launcher_is_fresh_cuda_only_and_uses_regret_advantage_router() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "point_maze_v23_planning_shaped_advantage_extension_20261022" in text
    assert "--episode-count 512" in text
    assert "--root-id-min 10296 --root-id-max-exclusive 10808" in text
    assert "--router-kind advantage_ridge" in text
    assert "--action-representation full_ordered_20x2" in text
    assert "--require-shared-candidate-support" in text
    assert "static_preflight_v7_alias_contract_r2_context38_20261022.json" in text
    assert "--device cuda" in text
    assert "experiments.gpu_preflight --device cuda --recurrent-smoke" in text
    assert "NVML telemetry is not an execution contract" in text
    assert "GPU_PREFLIGHT_ATTEMPTS" in text
    assert "refusing CPU fallback" in text
    assert "audit_planning_shaped_repairability_mechanism" in text
    assert "run_pointmaze_decision_aliasing_audit" in text
    assert "decision_aliasing_calibration_outcome.json" in text
    assert "--decision-aliasing-audit \"${OUTPUT}/decision_aliasing_calibration_outcome.json\"" in text
    assert "--max-reported-pairs 64" in text
    assert "declared full-coverage aliasing contract" in text
    assert "preflight_pointmaze_planning_shaped_advantage_v7" in text
    assert "--frozen-preflight-ledger \"${PREFLIGHT}\"" in text
    assert "if [[ -e \"${PANEL}\" || -e \"${OUTPUT}\" ]]" in text
    assert "extension_manifest.json" in text
    assert "matching completed extension is safely reusable" in text
