"""Summarize root-disjoint P1/P2 protocol-repair panels without mixing contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.spectral_reachability.run_sra_p2_action_order_cuda import spearman
from experiments.spectral_reachability.summarize_sra_pointmaze_identity_panels import _bootstrap_mean


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "mechanism_ready":
        raise ValueError(f"not a mechanism-ready panel: {path}")
    return payload


def _validate_group(panels: list[dict[str, Any]], gates: tuple[str, ...], score_key: str, expected_score: str) -> dict[str, Any]:
    roots: set[int] = set()
    contracts: set[str] = set()
    for panel in panels:
        if not all(panel["gates"].get(gate) is True for gate in gates):
            raise ValueError("an implementation gate failed")
        if panel["inputs"].get(score_key) != expected_score:
            raise ValueError(f"unexpected {score_key}")
        panel_roots = {int(root) for root in panel["inputs"]["roots"]}
        if roots & panel_roots:
            raise ValueError("panels share roots")
        roots |= panel_roots
        contracts.add(panel["inputs"]["physical_position_backend"]["contract_sha256"])
    if len(contracts) != 1:
        raise ValueError("a protocol-repair panel group mixes physical contracts")
    return {"roots": sorted(roots), "contract_sha256": next(iter(contracts))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1-report", type=Path, action="append", required=True)
    parser.add_argument("--p2-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    p1, p2 = [_read(path) for path in args.p1_report], [_read(path) for path in args.p2_report]
    p1_contract = _validate_group(p1, ("cuda_only", "frozen_parameter_hash_unchanged", "every_chart_has_finite_nonzero_jacobian", "no_outcome_labels_used_for_selection"), "continuation_score", "discrete_reachable_set")
    p2_contract = _validate_group(p2, ("cuda_only", "frozen_parameter_hash_unchanged", "model_order_effect_nonzero_every_root", "physical_order_effect_nonzero_every_root"), "effect_score", "goal_projection")
    p1_values = np.asarray([row["selection"]["spectral_physical_regret"] - row["selection"]["immediate_physical_regret"] for panel in p1 for row in panel["rows"]])
    p1_panel_means = [float(panel["aggregate"]["paired_regret_difference_spectral_minus_immediate"]) for panel in p1]
    p2_pred = np.asarray([pair["predicted_latent_order_effect_l2"] for panel in p2 for row in panel["rows"] for pair in row["pairs"]])
    p2_actual = np.asarray([pair["physical_position_order_effect_l2"] for panel in p2 for row in panel["rows"] for pair in row["pairs"]])
    p2_roots = np.asarray([row["root_spearman"] for panel in p2 for row in panel["rows"]])
    if not np.isfinite(p2_roots).all():
        raise ValueError("P2 contains an undefined root-level correlation")
    p1_positive = bool(all(value < 0.0 for value in p1_panel_means))
    p2_positive = bool(spearman(p2_pred, p2_actual) > 0.0 and _bootstrap_mean(p2_roots, seed=20261024)[0] > 0.0)
    report = {
        "schema": "sra_pointmaze_protocol_repair_summary.v1",
        "status": "mechanism_supported" if p1_positive and p2_positive else "method_negative_after_protocol_repairs",
        "p1_discrete_reachable_set": {"reports": [str(path.resolve()) for path in args.p1_report], "contract": p1_contract, "panel_means_spectral_minus_immediate": p1_panel_means, "pooled_mean_spectral_minus_immediate": float(p1_values.mean()), "pooled_ci95": _bootstrap_mean(p1_values, seed=20261027), "positive_evidence": p1_positive},
        "p2_goal_projection": {"reports": [str(path.resolve()) for path in args.p2_report], "contract": p2_contract, "global_spearman": spearman(p2_pred, p2_actual), "mean_root_spearman": float(p2_roots.mean()), "mean_root_spearman_ci95": _bootstrap_mean(p2_roots, seed=20261028), "positive_evidence": p2_positive},
        "stage_gate": {"p3_unlocked": bool(p1_positive and p2_positive), "reason": "P1 discrete reachable-set selection and P2 goal-projected action-order effects must both be positive on root-disjoint receipt-bound panels"},
        "claim_boundary": "This rules on two explicitly repaired fixed-bank diagnostics. It is not a general spectral-theory conclusion, RGB-only closed-loop success claim, or full-state simulator-equivalence result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"], "stage_gate": report["stage_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
