"""Fail-closed root-paired summary for an immutable RGB-only SRA P4 receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "sra_pointmaze_p4_rgb_closed_loop_summary.v1"
BOOTSTRAP_SAMPLES = 20_000
SIGN_FLIP_SAMPLES = 100_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval(values: np.ndarray, *, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _sign_flip_pvalue(values: np.ndarray, *, seed: int) -> float:
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(SIGN_FLIP_SAMPLES, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (1 + len(null)))


def summarize(receipt: Path) -> dict[str, object]:
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    inputs, gates, rows = payload.get("inputs"), payload.get("gates"), payload.get("rows")
    if payload.get("schema") != "sra_pointmaze_p4_rgb_closed_loop.v1" or not isinstance(inputs, dict) or not isinstance(gates, dict) or not isinstance(rows, list):
        raise ValueError("not a P4 closed-loop receipt")
    outcome_roots = [int(value) for value in inputs.get("outcome_roots", [])]
    calibration_roots = [int(value) for value in inputs.get("calibration_roots", [])]
    by_root: dict[int, dict[str, dict[str, object]]] = {root: {} for root in outcome_roots}
    for row in rows:
        if not isinstance(row, dict) or row.get("root") not in by_root or row.get("method") not in {"spectral", "immediate"}:
            raise ValueError("row has unexpected root or method")
        by_root[int(row["root"])][str(row["method"])] = row
    complete_pairs = all(set(values) == {"spectral", "immediate"} for values in by_root.values())
    paired = []
    for root, values in by_root.items():
        if set(values) != {"spectral", "immediate"}:
            continue
        spectral, immediate = values["spectral"], values["immediate"]
        paired.append({
            "root": root,
            "final_distance_difference": float(spectral["final_distance"]) - float(immediate["final_distance"]),
            "minimum_distance_difference": float(spectral["minimum_distance"]) - float(immediate["minimum_distance"]),
            "success_difference": int(bool(spectral["success"])) - int(bool(immediate["success"])),
            "candidate_rollout_difference": int(spectral["total_candidate_rollouts"]) - int(immediate["total_candidate_rollouts"]),
            "planner_seconds_difference": float(spectral["total_planner_seconds"]) - float(immediate["total_planner_seconds"]),
        })
    final = np.asarray([row["final_distance_difference"] for row in paired], dtype=np.float64)
    minimum = np.asarray([row["minimum_distance_difference"] for row in paired], dtype=np.float64)
    checks = {
        "source_status_admission_pass": payload.get("status") == "admission_pass",
        "source_gates_all_pass": all(bool(value) for value in gates.values()),
        "runner_and_input_identities_bound": all(
            isinstance(inputs.get(key), str) and len(str(inputs[key])) == 64
            for key in ("runner_sha256", "runtime_contract_sha256", "rgb_calibration_manifest_sha256", "state_extension_manifest_sha256")
        ),
        "calibration_outcome_root_disjoint": not bool(set(calibration_roots) & set(outcome_roots)),
        "complete_paired_rows": complete_pairs and len(paired) == len(outcome_roots) and len(outcome_roots) >= 2,
        "matched_candidate_rollout_count_per_root": all(row["candidate_rollout_difference"] == 0 for row in paired),
        "finite_paired_outcomes": bool(np.isfinite(final).all() and np.isfinite(minimum).all()),
    }
    if not all(checks.values()):
        status, estimates = "invalid", {}
    else:
        status = "outcome_complete"
        estimates = {
            "spectral_minus_immediate_final_distance": {
                "mean": float(final.mean()), "bootstrap_95_ci": _interval(final, seed=2026102404),
                "two_sided_sign_flip_p": _sign_flip_pvalue(final, seed=2026102405),
            },
            "spectral_minus_immediate_minimum_distance": {
                "mean": float(minimum.mean()), "bootstrap_95_ci": _interval(minimum, seed=2026102406),
                "two_sided_sign_flip_p": _sign_flip_pvalue(minimum, seed=2026102407),
            },
            "success_counts": {
                method: int(sum(bool(values[method]["success"]) for values in by_root.values()))
                for method in ("spectral", "immediate")
            },
        }
    return {
        "schema": SCHEMA, "status": status,
        "scope": "Root-paired PointMaze RGB-only P4 outcome summary; a method conclusion remains limited to this frozen checkpoint, new archive, action bank, and four-block protocol.",
        "source_receipt": str(receipt.resolve()), "source_receipt_sha256": _sha256(receipt),
        "checks": checks, "root_count": len(outcome_roots), "paired": paired, "estimates": estimates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable P4 summary: {args.output}")
    result = summarize(args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["checks"], "estimates": result["estimates"]}, indent=2))
    if result["status"] != "outcome_complete":
        raise RuntimeError("P4 outcome receipt failed a protocol check")


if __name__ == "__main__":
    main()
