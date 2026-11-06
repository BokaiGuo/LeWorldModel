"""Prespecified exploratory matched-parent rank diagnostic from saved logs."""
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sign(value):
    return 0 if abs(value) <= 1e-8 else 1 if value > 0 else -1


def main():
    path = (Path(__file__).resolve().parents[1] / "results" /
            "pc_prefix_heldout_optimization_seed42_20261105_codex.audit.jsonl")
    runs = [json.loads(line) for line in path.read_text().splitlines()]
    if not runs or runs[0]["solve"] != 0:
        raise ValueError("missing first solve")
    rows = []
    for iteration in runs[0]["iterations"]:
        item = iteration["heldout_effect"]
        if item is None:
            continue
        for p, a, optimized in zip(item["parent_cost"], item["construction_probe_cost"],
                                   item["optimized_cost"]):
            parent = np.asarray(p, dtype=float)
            amin = np.asarray(a, dtype=float).min(axis=1)
            target = parent - np.asarray(optimized, dtype=float)
            gap = abs(parent[0] - parent[1]) / max(abs(parent[0]), abs(parent[1]), 1e-8)
            rows.append((gap, parent, amin, target))
    if len(rows) != 50:
        raise ValueError(f"expected 50 pairs, got {len(rows)}")
    output = {}
    for cohort, selected in (("all", rows), ("parent_gap_at_most_10pct",
                                             [row for row in rows if row[0] <= 0.1])):
        counts = {key: Counter() for key in ("negative_parent", "negative_a_min", "a_gain",
                                              "a_gain_swapped")}
        for _, parent, amin, target in selected:
            gain = parent - amin
            features = {"negative_parent": -parent, "negative_a_min": -amin,
                        "a_gain": gain, "a_gain_swapped": gain[::-1]}
            y = sign(target[0] - target[1])
            for key, feature in features.items():
                x = sign(feature[0] - feature[1])
                counts[key]["tie" if x == 0 or y == 0 else "agree" if x == y else "opposite"] += 1
        output[cohort] = {"pairs": len(selected), "ranking": counts}
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
