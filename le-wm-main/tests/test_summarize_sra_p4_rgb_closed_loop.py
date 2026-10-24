import json

from experiments.spectral_reachability.summarize_sra_p4_rgb_closed_loop import summarize


def test_p4_summary_requires_complete_matched_root_pairs(tmp_path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "schema": "sra_pointmaze_p4_rgb_closed_loop.v1", "status": "admission_pass",
        "inputs": {"calibration_roots": [1, 2], "outcome_roots": [3, 4], "runner_sha256": "a" * 64, "runtime_contract_sha256": "b" * 64, "rgb_calibration_manifest_sha256": "c" * 64, "state_extension_manifest_sha256": "d" * 64},
        "gates": {"cuda_only": True},
        "rows": [
            {"root": 3, "method": "spectral", "final_distance": 1.0, "minimum_distance": 0.5, "success": False, "total_candidate_rollouts": 100, "total_planner_seconds": 1.0},
            {"root": 3, "method": "immediate", "final_distance": 2.0, "minimum_distance": 1.0, "success": False, "total_candidate_rollouts": 100, "total_planner_seconds": 1.0},
            {"root": 4, "method": "spectral", "final_distance": 1.0, "minimum_distance": 0.5, "success": False, "total_candidate_rollouts": 100, "total_planner_seconds": 1.0},
        ],
    }), encoding="utf-8")
    result = summarize(receipt)
    assert result["status"] == "invalid"
    assert not result["checks"]["complete_paired_rows"]
