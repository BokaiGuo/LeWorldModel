"""Scalar/trajectory audit, no inference or replacement efficacy experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability import reacher_d2p_refined_dispatch as selector


def audit(report, selections):
    if report.get("status") != "completed" or not report.get("gates") or not all(report["gates"].values()):
        raise ValueError("complete integrity-passing physical evidence required")
    roots = set(report["sealed_protocol"]["root_seeds"])
    physical = {row["seed"]: row for row in report["rows"]}
    model = {row["seed"]: row for row in selections["rows"]}
    if len(physical) != len(report["rows"]) or len(model) != len(selections["rows"]) or set(physical) != roots or set(model) != roots:
        raise ValueError("duplicate-free full root-identity joins required")
    rows = []
    for root in sorted(roots):
        actual, sealed = physical[root], model[root]
        if any(actual[key] != sealed[key] for key in ("source_episode", "source_start_row")):
            raise ValueError("physical source-root identity differs from sealed model root")
        receipt = sealed["receipt"]
        decision = selector.select_refined_bank(receipt["proposal_rows"], non_greedy_margin=1e-6)
        if actual["decision"] != decision or receipt["dispatch"] != decision:
            raise ValueError("sealed selector cannot be reproduced")
        bank = actual["word_results"]
        if set(bank) != {"word_" + str(i) for i in range(17)}:
            raise ValueError("all17 word physical outcomes required")
        for name, cell in list(bank.items()) + list(actual["methods"].items()):
            actions = np.asarray(cell["action_witness_raw"], np.float64)
            if actions.shape != (30, 2) or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1 + 1e-6:
                raise ValueError("complete bounded30-step action witness required")
            if [boundary["raw_steps_executed"] for boundary in cell["prefix_boundaries"]] != [5, 10, 15, 20]:
                raise ValueError("four distinct causal observation boundaries required")
            if any(len(boundary["history_rgb_sha256"]) != 3 for boundary in cell["prefix_boundaries"]):
                raise ValueError("three-frame history required at every boundary")
            if cell["execution_identity"]["reset_rgb_sha256"] != sealed["root_rgb_sha256"] or cell["execution_identity"]["goal_rgb_sha256"] != sealed["goal_rgb_sha256"]:
                raise ValueError("root/goal RGB identity differs across branches")
        for name, index in (("dispatch", decision["selected_program_index"]), ("same_bank_greedy", decision["same_bank_greedy_index"]), ("zero", 0)):
            if actual["methods"][name] != bank["word_" + str(index)]:
                raise ValueError("method alias differs from its actually executed word")
        predicted = np.asarray([row["immediate_goal_cost"] for row in receipt["proposal_rows"]])
        observed = np.asarray([bank["word_" + str(i)]["observed_prefix_rgb_cost"] for i in range(17)])
        pm, om = predicted - predicted[0], observed - observed[0]
        order_pairs = []
        for first in range(1, 17, 2):
            second = first + 1
            labels = receipt["word_support"]["labels"]
            if labels[first]["signs"] != labels[second]["signs"] or labels[first]["amplitude"] != labels[second]["amplitude"]:
                raise ValueError("matched amplitude/sign order pairs required")
            pd, od = float(predicted[first] - predicted[second]), float(observed[first] - observed[second])
            order_pairs.append({"first_word": first, "second_word": second, "predicted_order_margin": pd,
                "observed_order_margin": od, "strict_agreement": pd * od > 0, "strict_reversal": pd * od < 0})
        chosen = decision["selected_program_index"]
        rows.append({"seed": root, "model_witness": decision["reconfiguration_witness_found"],
            "selected_predicted_prefix_margin": float(pm[chosen]), "selected_observed_prefix_margin": float(om[chosen]),
            "strict_nonzero_word_margin_agreements": int(np.sum(pm[1:] * om[1:] > 0)),
            "strict_nonzero_word_margin_reversals": int(np.sum(pm[1:] * om[1:] < 0)), "order_pairs": order_pairs})
    eligible = [row for row in rows if row["model_witness"]]
    return {"rows": rows, "aggregate": {"root_count": len(rows), "nonzero_word_cells": len(rows) * 16,
        "prefix_sign_agreements": sum(row["strict_nonzero_word_margin_agreements"] for row in rows),
        "prefix_sign_reversals": sum(row["strict_nonzero_word_margin_reversals"] for row in rows),
        "model_witnesses": len(eligible), "witness_actual_rgb_non_greedy": sum(row["selected_observed_prefix_margin"] >= 1e-6 for row in eligible),
        "order_pair_count": len(rows) * 8, "strict_order_pair_agreements": sum(cell["strict_agreement"] for row in rows for cell in row["order_pairs"]),
        "strict_order_pair_reversals": sum(cell["strict_reversal"] for row in rows for cell in row["order_pairs"])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new immutable scalar audit required")
    report = json.loads(args.report.read_text())
    selection_path = args.report.with_suffix(".selection.json")
    if digest(selection_path) != report["selection_sha256"]:
        raise ValueError("model selection SHA changed")
    summary = audit(report, json.loads(selection_path.read_text()))
    result = {"schema": "reacher_d2p_action_word_result_audit.v1", "status": "completed", "input_sha256": digest(args.report),
        "selection_sha256": report["selection_sha256"], **summary,
        "scope": "All-root scalar/trajectory verification, not new GPU inference, new efficacy or a P1 admission decision."}
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregate"]))


if __name__ == "__main__":
    main()
