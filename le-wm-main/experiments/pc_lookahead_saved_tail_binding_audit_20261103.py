"""Audit whether saved tail packets and reported costs share the final CEM batch."""
import argparse
import json
import re

import numpy as np


def audit(data):
    keys = sorted(k for k in data if re.fullmatch(r"r\d+_lookahead_tails", k))
    if not keys:
        raise ValueError("no saved roots")
    rows = []
    for key in keys:
        root = key.split("_")[0]
        tails = data[key]
        terminal = data[f"{root}_terminal_scores"]
        if tails.shape != (9, 10) or terminal.shape != (9, 2):
            raise ValueError(f"invalid saved arrays for {root}")
        for i in range(9):
            stem = f"{root}_prefix{i}_lookahead3"
            proposals, scores = data[f"{stem}_proposals"], data[f"{stem}_scores"]
            if proposals.shape != (1, 2, 1, 10) or scores.shape != (2,):
                raise ValueError(f"invalid final CEM batch for {root}/{i}")
            if not all(np.isfinite(x).all() for x in (tails[i], terminal[i], proposals, scores)):
                raise ValueError("nonfinite saved data")
            if not np.array_equal(terminal[i], scores):
                raise ValueError(f"reported scores differ from final batch: {root}/{i}")
            winner = int(scores[1] < scores[0])
            if not np.array_equal(tails[i], proposals[0, winner, 0]):
                raise ValueError(f"saved tail differs from winner: {root}/{i}")
            rows.append({"root": root, "prefix": i, "winner": winner,
                         "score_margin": float(scores[0] - scores[1])})
    return {"roots": len(keys), "checked_tails": len(rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()
    with np.load(args.archive, allow_pickle=False) as data:
        print(json.dumps(audit(data), indent=2))


if __name__ == "__main__":
    main()
