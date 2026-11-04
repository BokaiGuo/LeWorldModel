"""Validate saved fixed-input scores and recompute fixed-bank physical regret."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(result, source):
    if len(result["rows"]) != 4 or len(source["rows"]) != 4:
        raise ValueError("incomplete root panel")
    if result["parameter_hash_before"] != result["parameter_hash_after"]:
        raise ValueError("model parameter mismatch")
    checks = []
    for row, old in zip(result["rows"], source["rows"], strict=True):
        if row["root"] != old["root"]:
            raise ValueError("root mismatch")
        scores = np.asarray(row["model_scores"], dtype=float)
        positions = np.asarray(old["physical_terminal_positions"], dtype=float)
        target = np.asarray(old["goal_position_evaluation_only"], dtype=float)
        if scores.shape != (8, 8) or positions.shape != (8, 8, 2) or target.shape != (2,):
            raise ValueError("unexpected candidate geometry")
        if not np.isfinite(scores).all() or not np.isfinite(positions).all():
            raise ValueError("non-finite inputs")
        choice = tuple(map(int, np.unravel_index(np.argmin(scores), scores.shape)))
        distances = np.linalg.norm(positions - target, axis=-1)
        regret = float(distances[choice] - distances.min())
        if list(choice) != row["model_pair"] or not np.isclose(regret, row["physical_regret"], atol=1e-12):
            raise ValueError("saved choice or regret mismatch")
        if not np.allclose(distances, old["physical_terminal_costs"], rtol=0, atol=1e-12):
            raise ValueError("source position costs mismatch")
        checks.append({"root": row["root"], "choice": list(choice), "physical_regret": regret})
    return checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result, source = (json.loads(path.read_text()) for path in (args.result, args.source))
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if result["source_sha256"] != source_hash:
        raise ValueError("source hash mismatch")
    checks = validate(result, source)
    report = {"scope": "saved-array validation only; no model inference or new physical replay",
              "result_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
              "source_sha256": source_hash,
              "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checks": checks}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
