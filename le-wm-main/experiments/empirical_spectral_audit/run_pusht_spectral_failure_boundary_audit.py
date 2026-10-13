"""Connect the spectral branch to the PushT contact failure boundary.

This audit does not claim that spectral geometry solves contact control.  It
tests the narrower bridge required by the paper: whether the declared
input-output object ``G_h = D A^(h-1) B`` exposes the identifiability boundary
that separates action-correlated latent dynamics from reliable contact pose
control.  The spectral and contact artifacts remain independently produced;
the contact result is an outcome, never used to fit A/B/D or tune a gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
    return array


def _operator_row(report: dict[str, Any], horizons: tuple[int, ...]) -> dict[str, Any]:
    model = report["fit"]["model"]
    A, B, D = _matrix(model["A"], "A"), _matrix(model["B"], "B"), _matrix(model["D"], "D")
    singular_values = np.linalg.svd(B, compute_uv=False)
    row: dict[str, Any] = {
        "action_condition_number": float(report["fit"]["action_excitation"]["condition_number"]),
        "action_excitation_passes_declared": bool(report["fit"]["action_excitation"]["passes"]),
        "task_dynamics_rmse": float(model["task_rmse"]),
        "latent_dynamics_rmse": float(model["dynamics_rmse"]),
        "B_rank": int(np.linalg.matrix_rank(B, tol=1.0e-8)),
        "B_min_singular_value": float(singular_values[-1]),
        "G": {},
        "controllability": {},
    }
    for horizon in horizons:
        G = D @ np.linalg.matrix_power(A, horizon - 1) @ B
        gramian = sum(
            np.linalg.matrix_power(A, index) @ B @ B.T @ np.linalg.matrix_power(A, index).T
            for index in range(horizon)
        )
        gramian_singular_values = np.linalg.svd(gramian, compute_uv=False)
        row["G"][str(horizon)] = {
            "frobenius_gain": float(np.linalg.norm(G, "fro")),
            "spectral_gain": float(np.linalg.svd(G, compute_uv=False)[0]),
            "rank": int(np.linalg.matrix_rank(G, tol=1.0e-8)),
        }
        row["controllability"][str(horizon)] = {
            "minimum_singular_value": float(gramian_singular_values[-1]),
            "effective_rank": int(np.sum(gramian_singular_values > 1.0e-8)),
            "condition_number": float(
                gramian_singular_values[0] / max(gramian_singular_values[-1], 1.0e-30)
            ),
        }
    return row


def run(spectral_paths: list[Path], contact_path: Path) -> dict[str, Any]:
    if not spectral_paths:
        raise ValueError("at least one spectral report is required")
    spectral_rows = [_operator_row(json.loads(path.read_text()), (1, 2, 4)) for path in spectral_paths]
    contact = json.loads(contact_path.read_text())
    closed_loop = contact.get("closed_loop", {})
    mode_lookahead = closed_loop.get("contact_chart_lookahead", {}).get("metrics", {})
    mode_accuracy = contact.get("held_out", {}).get("mode_accuracy")
    pose_success = mode_lookahead.get("pose_success", {}).get("estimate")
    angle_error = mode_lookahead.get("final_angle", {}).get("estimate")
    contact_loss = mode_lookahead.get("contact_loss_rate", {}).get("estimate")
    condition_numbers = np.asarray([row["action_condition_number"] for row in spectral_rows])
    gramian_h4 = np.asarray([row["controllability"]["4"]["minimum_singular_value"] for row in spectral_rows])
    task_rmse = np.asarray([row["task_dynamics_rmse"] for row in spectral_rows])
    return {
        "schema": "pusht_spectral_failure_boundary_audit.v1",
        "status": "failure_boundary_bridge_not_control_success",
        "spectral_source": [str(path) for path in spectral_paths],
        "contact_source": str(contact_path),
        "spectral_rows": spectral_rows,
        "aggregate": {
            "n_spectral_splits": len(spectral_rows),
            "action_condition_number_mean": float(condition_numbers.mean()),
            "action_condition_number_min": float(condition_numbers.min()),
            "action_condition_number_max": float(condition_numbers.max()),
            "h4_controllability_min_singular_value_max": float(gramian_h4.max()),
            "h4_controllability_min_singular_value_median": float(np.median(gramian_h4)),
            "task_dynamics_rmse_mean": float(task_rmse.mean()),
        },
        "contact_outcome": {
            "heldout_mode_accuracy": None if mode_accuracy is None else float(mode_accuracy),
            "pose_success": None if pose_success is None else float(pose_success),
            "final_angle": None if angle_error is None else float(angle_error),
            "contact_loss_rate": None if contact_loss is None else float(contact_loss),
        },
        "bridge_tests": {
            "action_excitation_is_formally_full_rank_but_ill_conditioned": bool(
                all(row["action_excitation_passes_declared"] for row in spectral_rows)
                and float(condition_numbers.min()) > 1.0e3
            ),
            "finite_horizon_reachability_is_numerically_degenerate": bool(float(gramian_h4.max()) < 1.0e-8),
            "mode_accuracy_does_not_imply_pose_closure": bool(
                mode_accuracy is not None and pose_success is not None
                and float(mode_accuracy) >= 0.9 and float(pose_success) < 0.5
            ),
        },
        "interpretation": (
            "PushT is a failure-boundary instantiation of the spectral framework: "
            "a formally action-excited latent operator can remain ill-conditioned "
            "for finite-horizon task transfer, and successor-mode prediction can "
            "be accurate while within-mode G_h/action-effect identification is "
            "insufficient for pose closure."
        ),
        "claim_boundary": (
            "This artifact connects the branches by a predeclared diagnostic and "
            "negative boundary test. It does not prove that geometry predicts every "
            "PushT failure, and it does not establish automatic contact control, "
            "causal dynamics, or a universal certificate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral-dir", type=Path, required=True)
    parser.add_argument("--contact-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.spectral_dir.glob("jepa_pusht_windows4_composite_splitseed*.json"))
    report = run(paths, args.contact_report)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "pusht_spectral_failure_boundary_audit.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["bridge_tests"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
