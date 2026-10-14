"""Build a claim-safe manifest for the unified ATM -> HTSC baseline.

This is a post-hoc aggregator.  It never selects checkpoints from held-out
outcomes and never converts an offline task-contrast or inverse-risk result
into a planning certificate.  The manifest records the actual GPU preflight
state so a CPU post-hoc audit cannot be confused with learned GPU inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_preflight() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        record: dict[str, Any] = {
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": available,
            "device_count": int(torch.cuda.device_count()),
        }
        if not available:
            record.update({"status": "environment_blocked", "allocation": "not_attempted_after_unavailable"})
            return record
        try:
            value = torch.ones(8, device="cuda")
            record.update({"status": "passed", "allocation": str(value.device), "sum": float(value.sum())})
        except Exception as exc:  # pragma: no cover - host-specific runtime
            record.update({"status": "environment_blocked", "allocation_error": f"{type(exc).__name__}: {exc}"})
        return record
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"status": "environment_blocked", "probe_error": f"{type(exc).__name__}: {exc}"}


def build_manifest(root: Path, *, cross_environment: Path) -> dict[str, Any]:
    p0 = root / "p0_pointmaze" / "task_contrast_audit.json"
    p1_atm = root / "p1_atm" / "atm_cross_domain_audit.json"
    p1_panel = root / "p1_learned_panel" / "oral_level_unified_diagnostic_audit.json" / "oral_level_unified_diagnostic_audit.json"
    required = (p0, p1_atm, p1_panel, cross_environment)
    payloads = {str(path): _load(path) for path in required}
    p0_data = payloads[str(p0)]
    p1_data = payloads[str(p1_atm)]
    panel = payloads[str(p1_panel)]
    cross = payloads[str(cross_environment)]
    strict = cross.get("global_audit", {})
    summary = {
        "schema": "unified_decision_transfer_p0p1_manifest.v1",
        "gpu_preflight": _gpu_preflight(),
        "stages": {
            "P0_pointmaze_task_contrast": {
                "status": "partial",
                "inversion_rate": p0_data["outcome"]["inversion_rate"],
                "contrast_error_mean": p0_data["outcome"]["contrast_error_mean"],
                "calibration_bound": p0_data["calibration"]["bound"],
                "certificate_status": p0_data["certificate_preview"]["status"],
                "claim_boundary": "offline candidate contrast; no RGB closed-loop planning",
            },
            "P1_ATM": {
                "status": "diagnostic_only",
                "inverse_risk_matrix": p1_data["inverse_risk_matrix"],
                "effect_only_inverse_risk_matrix": p1_data["effect_only_inverse_risk_matrix"],
                "ranking_status": p1_data["ranking_audit"]["status"],
                "claim_boundary": p1_data["protocol"]["claim_boundary"],
            },
            "P1_learned_matched_panel": {
                "status": panel["strict_matched_audit"]["summary"].get("status", "unknown"),
                "identifiable_strata": panel["strict_matched_audit"]["summary"].get("identifiable_strata"),
                "claim_boundary": panel["claim_boundary"],
            },
            "P1_cross_environment": {
                "status": strict.get("summary", {}).get("status", "unknown"),
                "environment_count": len(strict.get("inventory", {}).get("environments", [])),
                "claim_boundary": strict.get("claim_boundary"),
            },
        },
        "estimand": "Delta R | MSE, spectrum, training stage, action risk",
        "decision": {
            "GDD_WM": "not_run_proposed",
            "HTSC": "abstain_until_support_and_realized_successor_are_calibrated",
            "RGB_online_planning": "blocked",
            "universal_world_model": "not_established",
        },
        "inputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in required
        ],
        "protocol": {
            "posthoc_cpu_audit": True,
            "learned_inference_rerouted_to_cpu": False,
            "root_disjoint_task_contrast": True,
            "heldout_outcome_used_for_metric_selection": False,
            "support_gate_required": True,
            "claim_boundary": "implementation and baseline evidence only; no positive GDD-WM or planning claim",
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cross-environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_manifest(args.root, cross_environment=args.cross_environment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gpu": report["gpu_preflight"], "stages": report["stages"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
