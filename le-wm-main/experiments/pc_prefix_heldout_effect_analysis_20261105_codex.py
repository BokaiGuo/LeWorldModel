"""Analyze predeclared paired ranking diagnostics; never perform selection."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def direction(value, tolerance=1e-8):
    return 0 if abs(value) <= tolerance else (1 if value > 0 else -1)


def run(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    totals = {key: Counter() for key in ("parent_to_b_absolute", "a_to_b_absolute",
                                          "a_gain_to_b_gain")}
    samples = 0
    for record in records:
        for iteration in record["iterations"]:
            data = iteration["heldout_effect"]
            if data is None:
                continue
            for parent, a, b in zip(data["parent_cost"], data["construction_probe_cost"],
                                    data["heldout_b_cost"]):
                parent, a, b = np.asarray(parent), np.asarray(a), np.asarray(b)
                amin, bmin = a.min(axis=1), b.min(axis=1)
                comparisons = {
                    "parent_to_b_absolute": (parent, bmin),
                    "a_to_b_absolute": (amin, bmin),
                    "a_gain_to_b_gain": (parent - amin, parent - bmin),
                }
                for key, (predictor, target) in comparisons.items():
                    x, y = direction(predictor[0] - predictor[1]), direction(target[0] - target[1])
                    totals[key]["tie"] += int(x == 0 or y == 0)
                    totals[key]["agree"] += int(x != 0 and y != 0 and x == y)
                    totals[key]["opposite"] += int(x != 0 and y != 0 and x != y)
                samples += 1
    print(json.dumps({"paired_contexts": samples, "ranking": totals},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    run(parser.parse_args().path)
