"""Separate fixed-neutral non-greediness from same-feedback strategy contrasts.

Tracking the zero plan generally sends nonzero compensation actions. It is
not the fixed sealed neutral prefix. Neither contrast is same-action model
ranking fidelity because feedback changes the candidate's commands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.analyze_reacher_d2p_waypoint_tracking import load
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def summarize(summary):
    rows = sorted(summary["rows"], key=lambda r: r["seed"])
    seeds = sorted(summary["root_seeds"])
    if len(seeds) != 8 or len(set(seeds)) != 8 or [r["seed"] for r in rows] != seeds:
        raise ValueError("all eight unique fixed root identities required")
    output = []
    for row in rows:
        costs = row["prefix_observed_costs"]
        values = [costs[k] for k in ("native_dispatch_replay", "native_dispatch_tracking", "zero_replay", "zero_tracking")]
        if not np.isfinite(values).all() or min(values) < 0 or not np.isfinite(row["original_planned_prefix_margin"]):
            raise ValueError("finite nonnegative observed costs required")
        native, tracked_native, neutral, tracked_zero = values
        output.append({"seed": row["seed"], "original_model_margin": row["original_planned_prefix_margin"],
            "replay_minus_fixed_neutral_margin": native - neutral,
            "tracking_minus_fixed_neutral_margin": tracked_native - neutral,
            "tracking_minus_tracking_zero_margin": tracked_native - tracked_zero,
            "fixed_neutral_method": "zero_replay", "same_feedback_control_method": "zero_tracking",
            "same_action_model_fidelity_comparison_valid": False})
    return {"root_seeds": seeds, "rows": output,
        "scope": "Known/fresh-root execution contrasts under their upstream provenance. Fixed neutral and feedback zero are different programs; no P1 admission or same-action model fidelity claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new immutable reference-contrast receipt required")
    result = summarize(load(args.report))
    write_new(args.output, {"schema": "reacher_d2p_waypoint_reference_margins.v1", "status": "completed",
        "report_path": str(args.report.resolve()), "report_sha256": digest(args.report), "script_sha256": digest(Path(__file__)), **result})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
