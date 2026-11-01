"""Replay a model-only routing rule against already executed physical branches.

This is a post-development trace diagnostic, never fresh efficacy evidence.
The old neutral-fallback report and its mechanism gates remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments.spectral_reachability.reacher_d2p_program_dispatch import POLICY, dispatch_decision


def summarize(report: dict) -> dict:
    if report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()):
        raise ValueError("completed integrity-passing source report required")
    roots = report["rows"]
    seeds = [row["seed"] for row in roots]
    if len(set(seeds)) != len(seeds) or sorted(seeds) != sorted(report["sealed_protocol"]["root_seeds"]):
        raise ValueError("missing or duplicate root identity")
    cells = []
    for root in sorted(roots, key=lambda row: row["seed"]):
        decision = dispatch_decision(root["model_selection"])
        witness = decision["reconfiguration_witness_found"]
        selected = root["reconfiguration"]
        if witness != (selected is not None):
            raise ValueError("model witness differs from physical branch existence")
        zero, greedy = root["zero_prefix"], root["same_bank_greedy"]
        if greedy["first_receipt"]["selected_proposal_index"] != decision["same_bank_greedy_index"]:
            raise ValueError("executed greedy packet differs from model reference")
        if witness and selected["first_receipt"]["selected_proposal_index"] != decision["selected_program_index"]:
            raise ValueError("executed reconfiguration differs from model decision")
        identities = [branch["execution_identity"]["initial"] for branch in (zero, greedy, selected) if branch is not None]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError("physical branches do not share exact root and goal identity")
        dispatched, neutral = (selected, selected) if witness else (greedy, zero)
        distance = lambda branch: float(branch["physical"]["final_qpos_distance"])
        final, old, baseline = distance(dispatched), distance(neutral), distance(greedy)
        if not all(math.isfinite(value) for value in (final, old, baseline)):
            raise ValueError("nonfinite physical distances")
        cells.append({"seed": root["seed"], "source_episode": root["source_episode"], "dispatch": decision,
            "final_qpos_distance": final, "greedy_final_qpos_distance": baseline,
            "neutral_policy_final_qpos_distance": old,
            "dispatch_minus_greedy": final - baseline, "neutral_minus_greedy": old - baseline,
            "dispatch_minus_neutral": final - old})
    mean = lambda key: sum(row[key] for row in cells) / len(cells)
    return {"schema": "reacher_d2p_program_dispatch_trace_replay.v1", "status": "completed",
        "policy": POLICY, "root_count": len(cells),
        "reconfiguration_count": sum(row["dispatch"]["reconfiguration_witness_found"] for row in cells),
        "progress_fallback_count": sum(not row["dispatch"]["reconfiguration_witness_found"] for row in cells),
        "aggregate": {"mean_dispatch_minus_greedy": mean("dispatch_minus_greedy"),
            "mean_neutral_minus_greedy": mean("neutral_minus_greedy"),
            "mean_dispatch_minus_neutral": mean("dispatch_minus_neutral"),
            "wins_against_greedy": sum(row["dispatch_minus_greedy"] < 0 for row in cells),
            "losses_against_greedy": sum(row["dispatch_minus_greedy"] > 0 for row in cells),
            "ties_against_greedy": sum(row["dispatch_minus_greedy"] == 0 for row in cells)},
        "rows": cells,
        "claim_boundary": "Post-development trace replay on reused diagnostic roots, not fresh efficacy, formal P1 admission, or resource-matched comparison. Physical labels never enter routing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.report.read_text()))
    result["inputs"] = {"report": str(args.report.resolve()),
        "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
