"""Read-only audit of saved SRA evidence; does not load a model or execute physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL = ROOT / "results/p2_no_composition_sra_physical_20261027_v2_cuda_main_v1/report.json"
PROOF = ROOT / "results/codex_sra_checkpoint_identity_20261030_CPU02.json"
ANCHORS = (
    "20554d967c49eec3e93e7ee846772d48be4967002d0d609cd70a8479678bcfd5",
    "ec42cfc0f32efc344fef05ccb7ac232209c5f32564f4ed8d8770fbc79c04a9cc",
)


def anchored_json(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("saved evidence SHA mismatch")
    return json.loads(raw)


def audit(historical=HISTORICAL, proof=PROOF, anchors=ANCHORS):
    old = anchored_json(historical, anchors[0])
    identity = anchored_json(proof, anchors[1])
    accounting = old["planning_accounting"]
    per_root = accounting["per_root"]
    if (per_root["compilation_predict_samples"], per_root["candidate_predict_samples"],
        per_root["replay_predict_samples"], per_root["total_predict_samples"],
        accounting["four_root"]["total_predict_samples"],
        accounting["fair_total_budget_comparison"]) != (4, 80, 4, 88, 352, False):
        raise ValueError("historical accounting mismatch")
    if identity["fixture"] is not False or identity["boundaries"]["equal_budget"] is not False:
        raise ValueError("identity proof boundary mismatch")
    assets = {}
    for name in ("weights", "config"):
        path = Path(identity["checkpoint"][name + "_path"])
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        assets[name] = {"path": str(path), "current_file_size_bytes": size,
                        "identity_proof_sha256": identity["checkpoint"][name + "_sha256"],
                        "current_content_reverified": False}
    return {
        "schema": "codex_sra_resource_coverage_audit_v1",
        "sources": [{"path": str(historical), "sha256": anchors[0]},
                    {"path": str(proof), "sha256": anchors[1]}],
        "historical_physical_report": {
            "scope": "historical run, not current online guard",
            "per_root_predict_samples": per_root["total_predict_samples"],
            "four_root_predict_samples": accounting["four_root"]["total_predict_samples"],
            "fair_total_budget_comparison": False,
            "unmeasured": accounting["unmeasured"],
        },
        "cpu_identity_proof": {"fixture": False, "checkpoint_tensor_count": identity["checkpoint_tensor_count"],
                               "remapped_keys": identity["remapped_keys"], "assets": assets},
        "current_asset_sizes_are_not_runtime_io": True,
        "available_local_measurements": {
            "guard": "setup steps and guard wall time; io_bytes and FLOPs unknown",
            "other_runs": "some load times and tensor payload counts are run-specific, not total SRA costs",
            "new_meter": "explicit operations only; not connected automatically to guard or loader",
        },
        "unavailable": {"total_file_io_bytes": None, "physical_disk_io_bytes": None,
                        "total_device_transfer_bytes": None, "physical_bus_bytes": None,
                        "model_load_wall_ns": None, "model_load_internal_io_bytes": None,
                        "model_load_internal_transfer_bytes": None, "total_compute_flops": None},
        "equal_total_budget": False,
        "audit_io_is_not_production_cost": True,
        "gpu_or_physical_execution": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit()
    output = Path(args.output)
    with output.open("xb") as f:
        f.write((json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())
    print(output)


if __name__ == "__main__":
    main()
