"""Compare fixed t5 prediction and t5 actual RGB; no new candidate selection."""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path

from . import p1_candidate_step_rgb_analysis_20261031 as real
from . import p1_uphill_all_candidates_midpoint_20261031 as prior

NAME = "p1_candidate_short_horizon_alignment_20261031"
ROOT = real.source.ROOT
MID = ROOT / "results/p1_uphill_all_candidates_midpoint_20261031_RUN02.json"
MID_SHA = "e0e3062b6a0ef46a94b5be0c159e19547da5182745a7a4c343a88c4e6d2f3de0"
EPS = 1e-6


def relation(a, b):
    if not all(math.isfinite(x) for x in (a, b)):
        raise ValueError("nonfinite pair difference")
    if abs(a) <= EPS or abs(b) <= EPS:
        return "tie"
    return "agree" if a * b > 0 else "reverse"


def analyze(mid, actual):
    if mid["status"] != "complete" or actual["status"] != "complete" or \
            len(mid["rows"]) != 96 or len(actual["rows"]) != 96:
        raise ValueError("fixed roster")
    latest = {(r["index"], r["candidate"]): r for r in actual["rows"]}
    observed = {(r["index"], r["candidate"]): r for r in mid["rows"]}
    expected = {(i, c) for i in range(24) for c in range(4)}
    if set(latest) != expected or set(observed) != expected:
        raise ValueError("candidate identity")
    counts = Counter()
    selectors = []
    per_pair = []
    for index in range(24):
        candidates = []
        for candidate in range(4):
            a, b = latest[index, candidate], observed[index, candidate]
            if len(a["steps"]) < 5:
                raise ValueError("missing midpoint")
            frame = a["steps"][4]
            if any(a[k] != b[k] for k in ("index", "candidate", "kind", "role", "logical_pair", "status", "success")) \
                    or frame["pixel_sha256"] != b["midpoint_pixel_sha256"] \
                    or abs(frame["actual_goal_mse"] - b["actual_midpoint_goal_mse"]) > 1e-5 \
                    or abs(a["model_terminal_score"] - b["open_terminal_goal_mse"]) > 1e-5:
                raise ValueError("saved midpoint identity")
            candidates.append({"candidate": candidate, "success": a["success"], "kind": a["kind"],
                               "predicted5": b["predicted_midpoint_goal_mse"],
                               "actual5": frame["actual_goal_mse"],
                               "open10": b["open_terminal_goal_mse"],
                               "refreshed10": b["refreshed_terminal_goal_mse"]})
        winner = min(candidates, key=lambda row: (row["open10"], row["candidate"]))
        if winner["kind"] != 0:
            raise ValueError("original selector identity")
        selectors.append({"index": index, "role": latest[index, 0]["role"],
                          "open10": winner["candidate"],
                          "predicted5": min(candidates, key=lambda row: (row["predicted5"], row["candidate"]))["candidate"],
                          "actual5_oracle": min(candidates, key=lambda row: (row["actual5"], row["candidate"]))["candidate"],
                          "success_by_candidate": [c["success"] for c in candidates]})
        for a, b in combinations(candidates, 2):
            label = relation(a["predicted5"] - b["predicted5"], a["actual5"] - b["actual5"])
            counts[label] += 1
            per_pair.append({"index": index, "candidates": [a["candidate"], b["candidate"]],
                             "predicted5_delta": a["predicted5"] - b["predicted5"],
                             "actual5_delta": a["actual5"] - b["actual5"], "relation": label})
    return {"schema": NAME, "tolerance": EPS, "five_step_pair_counts": dict(counts),
            "pairwise": per_pair, "selectors": selectors,
            "limitations": "Two known roots and repeated trajectories, retrospective selection; "
                           "actual5 requires executing every candidate. Not P1 evidence.",
            "P1_admission": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if not out.is_absolute() or out.parent != ROOT / "results" or not out.name.startswith(NAME + "_") \
            or out.exists() or out.is_symlink():
        raise ValueError("exclusive output")
    for path, digest in ((MID, MID_SHA), (real.INPUT, real.SHA)):
        if prior.base.sha(path) != digest:
            raise ValueError("source hash")
    result = analyze(json.loads(MID.read_bytes()), json.loads(real.INPUT.read_bytes()))
    result["sources"] = {str(MID): MID_SHA, str(real.INPUT): real.SHA,
                         str(Path(__file__).resolve()): prior.base.sha(Path(__file__).resolve())}
    for path, digest in result["sources"].items():
        if prior.base.sha(path) != digest:
            raise ValueError("source drift")
    digest = prior.base.write(out, result)
    print(json.dumps({"pair_counts": result["five_step_pair_counts"], "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
