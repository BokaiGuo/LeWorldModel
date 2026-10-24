"""Aggregate independent PointMaze SRA panels bound to one replay contract.

This is an evidence reducer, not a planner.  It rejects mixed physical
evaluator contracts, overlapping roots, or failed CUDA/frozen-model gates
before reporting whether P1/P2 are stable enough to justify later stages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.spectral_reachability.run_sra_p2_action_order_cuda import spearman


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "mechanism_ready":
        raise ValueError(f"panel is not mechanism-ready: {path}")
    return payload


def _contract_key(payload: dict[str, Any]) -> tuple[str, str, str]:
    backend = payload["inputs"]["physical_position_backend"]
    scripts = backend["endpoint_evaluator_scripts"]
    if backend["position_label_contract"].get("status") != "pass":
        raise ValueError("physical position contract did not pass")
    return (
        backend["contract_sha256"],
        scripts["replay_backend_script_sha256"],
        scripts["endpoint_backend_script_sha256"],
    )


def _bootstrap_mean(values: np.ndarray, *, seed: int, draws: int = 20_000) -> list[float]:
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("need at least two values for a bootstrap interval")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _validate_panels(panels: list[dict[str, Any]], *, expected_gates: tuple[str, ...]) -> tuple[str, str, str]:
    roots: set[int] = set()
    keys: set[tuple[str, str, str]] = set()
    for panel in panels:
        if not all(panel["gates"].get(gate) is True for gate in expected_gates):
            raise ValueError("a panel failed a required implementation/protocol gate")
        panel_roots = {int(root) for root in panel["inputs"]["roots"]}
        if roots & panel_roots:
            raise ValueError("independent panels share roots")
        roots |= panel_roots
        keys.add(_contract_key(panel))
    if len(keys) != 1:
        raise ValueError("panels use different evaluator contracts or endpoint scripts")
    return next(iter(keys))


def _stage_gate_reason(*, stable_p1: bool, stable_p2: bool, p1_panel_means: list[float]) -> str:
    """State the failed mechanism gate without inventing a sign pattern."""

    if stable_p1 and stable_p2:
        return "independent P1 and P2 evidence is positive under one bound evaluator contract"
    if not stable_p1 and not stable_p2:
        signs = {int(np.sign(value)) for value in p1_panel_means if value != 0.0}
        p1_reason = "P1 panel signs disagree" if len(signs) > 1 else "P1 does not establish a spectral-selection advantage"
        return f"{p1_reason}; P2 action-order correlations do not establish a positive effect"
    if not stable_p1:
        return "P1 does not establish a spectral-selection advantage"
    return "P2 action-order correlations do not establish a positive effect"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-report", type=Path, action="append", required=True)
    parser.add_argument("--p2-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    p1, p2 = [_read(path) for path in args.p1_report], [_read(path) for path in args.p2_report]
    p1_contract = _validate_panels(
        p1,
        expected_gates=("cuda_only", "frozen_parameter_hash_unchanged", "every_chart_has_finite_nonzero_jacobian", "no_outcome_labels_used_for_selection"),
    )
    p2_contract = _validate_panels(
        p2,
        expected_gates=("cuda_only", "frozen_parameter_hash_unchanged", "model_order_effect_nonzero_every_root", "physical_order_effect_nonzero_every_root"),
    )
    if p1_contract != p2_contract:
        raise ValueError("P1 and P2 do not share the same evaluator/script contract")

    p1_delta = np.asarray([
        row["selection"]["spectral_physical_regret"] - row["selection"]["immediate_physical_regret"]
        for panel in p1 for row in panel["rows"]
    ], dtype=np.float64)
    p1_panel_means = [float(panel["aggregate"]["paired_regret_difference_spectral_minus_immediate"]) for panel in p1]
    p2_pred = np.asarray([
        pair["predicted_latent_order_effect_l2"]
        for panel in p2 for row in panel["rows"] for pair in row["pairs"]
    ], dtype=np.float64)
    p2_actual = np.asarray([
        pair["physical_position_order_effect_l2"]
        for panel in p2 for row in panel["rows"] for pair in row["pairs"]
    ], dtype=np.float64)
    p2_root = np.asarray([
        row["root_spearman"] for panel in p2 for row in panel["rows"]
    ], dtype=np.float64)
    if not np.isfinite(p2_root).all():
        raise ValueError("P2 contains undefined root-level rank correlations")

    stable_p1 = bool(len(p1_panel_means) >= 2 and all(value < 0.0 for value in p1_panel_means))
    stable_p2 = bool(spearman(p2_pred, p2_actual) > 0.0 and _bootstrap_mean(p2_root, seed=20261024)[0] > 0.0)
    report = {
        "schema": "sra_pointmaze_identity_contract_cross_panel.v1",
        "contract": {
            "contract_sha256": p1_contract[0],
            "replay_backend_script_sha256": p1_contract[1],
            "endpoint_backend_script_sha256": p1_contract[2],
        },
        "p1": {
            "reports": [str(path.resolve()) for path in args.p1_report],
            "panel_paired_regret_means_spectral_minus_immediate": p1_panel_means,
            "pooled_paired_regret_mean_spectral_minus_immediate": float(p1_delta.mean()),
            "pooled_paired_regret_ci95": _bootstrap_mean(p1_delta, seed=20261025),
            "stable_positive_evidence": stable_p1,
        },
        "p2": {
            "reports": [str(path.resolve()) for path in args.p2_report],
            "global_spearman_predicted_vs_physical_order_effect": spearman(p2_pred, p2_actual),
            "mean_root_spearman": float(p2_root.mean()),
            "mean_root_spearman_ci95": _bootstrap_mean(p2_root, seed=20261026),
            "stable_positive_evidence": stable_p2,
        },
        "stage_gate": {
            "p3_unlocked": bool(stable_p1 and stable_p2),
            "reason": _stage_gate_reason(stable_p1=stable_p1, stable_p2=stable_p2, p1_panel_means=p1_panel_means),
        },
        "claim_boundary": "This aggregates evaluator-state-labelled fixed-bank diagnostics. It is not RGB-only closed-loop success, a full-state simulator equivalence result, or evidence for a general planning compiler.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "p1": report["p1"], "p2": report["p2"], "stage_gate": report["stage_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
