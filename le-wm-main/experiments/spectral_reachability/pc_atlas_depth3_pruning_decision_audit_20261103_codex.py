"""Explain saved pruning choices without re-running or changing the experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .pc_atlas_depth3_pruning_alternative_20261103_codex import choose_intermediate
from .pc_exact_unicycle_development import TURN


def explain(trace, mode, tolerance):
    candidates = trace["candidates"]
    selected = trace["selected_indices_in_beam_order"]
    if len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError("two distinct recorded selections required")
    if mode != "random":
        expected = choose_intermediate(candidates, mode, np.random.default_rng(0),
                                       [2., 2 * TURN], tolerance)
        if selected != expected:
            raise ValueError("recorded selection does not match declared policy")
    ordered_cost = sorted(range(len(candidates)), key=lambda i: candidates[i]["cost"])
    ordered_look = sorted(range(len(candidates)), key=lambda i: candidates[i]["lookahead_cost"])
    cost_rank = {index: rank for rank, index in enumerate(ordered_cost, 1)}
    look_rank = {index: rank for rank, index in enumerate(ordered_look, 1)}
    anchor = np.asarray(candidates[ordered_look[0]]["last_action"])
    best = candidates[ordered_look[0]]["lookahead_cost"]
    rows = []
    for i, node in enumerate(candidates):
        distance = float(np.linalg.norm((np.asarray(node["last_action"]) - anchor) / [2., 2 * TURN]))
        rows.append({"index": i, "current_cost_rank": cost_rank[i],
                     "lookahead_cost_rank": look_rank[i],
                     "within_lookahead_tolerance": node["lookahead_cost"] <= best + tolerance,
                     "normalized_action_distance_to_lookahead_best": distance,
                     "selected": i in selected})
    explanation = {
        "v2": "first two by current cost; lookahead scores discarded",
        "lookahead": "first two by lookahead cost",
        "lookahead_diverse": "best lookahead then most action-distant eligible node, or second lookahead if none eligible",
        "random": "two uniformly sampled distinct candidates; lookahead scores discarded",
    }[mode]
    return {"root_index": trace["root_index"], "root_name": trace["root_name"],
            "selection_reason": explanation, "selected_indices_in_beam_order": selected,
            "lookahead_best_index": ordered_look[0], "candidates": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.input.read_bytes()
    source = json.loads(raw)
    if source["schema"] != "pc_atlas_depth3_pruning_alternative.v1":
        raise ValueError("unexpected source schema")
    pairs = []
    for pair in source["pairs"]:
        arms = {}
        for mode, result in pair["arms"].items():
            arms[mode] = [explain(trace, mode, result["configuration"]["tolerance"])
                          for trace in result["intermediate_trace"]]
        pairs.append({"index": pair["index"], "seed": pair["seed"], "arms": arms})
    report = {"schema": "pc_atlas_depth3_pruning_decision_audit.v1",
              "source_sha256": hashlib.sha256(raw).hexdigest(),
              "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({"pairs": len(pairs), "source_sha256": report["source_sha256"]}))


if __name__ == "__main__":
    main()
