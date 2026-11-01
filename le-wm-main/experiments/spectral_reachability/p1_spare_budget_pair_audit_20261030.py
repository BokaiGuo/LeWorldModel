"""Pair saved known-root nodes and audit effects of extra optimization."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_four_arm_hooked_saved_check_20261030_main as old_check
from . import p1_four_arm_spare_budget_saved_check_20261030 as new_check

ROOT = Path(__file__).resolve().parents[2]
NAME = "p1_spare_budget_pair_audit_20261030"
OLD_SHA = "599407de034bed59d3c424fe5284f5317b524053e4af889ecc2c7bc5f432acb6"
NEW_SHA = "07476d922fcfbe6413061b2d7d2d57c2abbf955d6401e9b42f937b8ba685c7a4"
IDENTITY = ("root", "goal", "history", "action_domain", "model", "optimizer", "optimizer_config", "hardware")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def events(path):
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    kinds = ("node_header", "initial_candidates", "optimized_candidates", "final_selection")
    selected = [row for row in rows if row["kind"] in kinds]
    values = {row["kind"]: row["payload"] for row in selected}
    need(len(selected) == len(kinds) and set(values) == set(kinds), "event roster")
    return values


def identity(path):
    first = json.loads(path.read_bytes().splitlines()[0])
    return first["payload"]["contract"]["identity"]


def analyze(old, new):
    old_check.check(old, OLD_SHA)
    new_check.check(new, NEW_SHA)
    rows = []
    for arm in ("native", "random"):
        for boundary in (0, 5, 10, 15):
            old_ns = old / f"s0_{arm}_b{boundary:02}_common"
            new_ns = new / f"s0_{arm}_b{boundary:02}_spare"
            a, b = events(old_ns / "candidates.jsonl"), events(new_ns / "candidates.jsonl")
            ia, ib = identity(old_ns / "ledger/events.jsonl"), identity(new_ns / "ledger/events.jsonl")
            need(all(ia[key] == ib[key] for key in IDENTITY), "planning identity mismatch")
            need(a["node_header"]["arm"] == b["node_header"]["arm"] == arm
                 and a["node_header"]["config"] == b["node_header"]["config"]
                 and b["node_header"]["extra_indices"] == [0, 1], "node config")
            before = np.asarray(a["optimized_candidates"]["raw"], np.float32)
            after = np.asarray(b["optimized_candidates"]["raw"], np.float32)
            initial_a = np.asarray(a["initial_candidates"]["raw"], np.float32)
            initial_b = np.asarray(b["initial_candidates"]["raw"], np.float32)
            need(before.shape == after.shape == initial_a.shape == initial_b.shape == (8, 3, 5, 2)
                 and initial_a.tobytes() == initial_b.tobytes()
                 and before[2:].tobytes() == after[2:].tobytes(), "candidate binding")
            score_a = np.asarray(a["final_selection"]["scores"], np.float64)
            score_b = np.asarray(b["final_selection"]["scores"], np.float64)
            need(score_a.shape == score_b.shape == (8,) and np.isfinite(score_a).all()
                 and np.isfinite(score_b).all() and np.array_equal(score_a[2:], score_b[2:])
                 and int(np.argmin(score_a)) == a["final_selection"]["index"]
                 and int(np.argmin(score_b)) == b["final_selection"]["index"], "score binding")
            winner = int(np.argmin(score_b))
            rows.append({"arm": arm, "boundary": boundary, "matched_identity_fields": list(IDENTITY),
                         "whole_task_hash_equal": ia["task"] == ib["task"],
                         "old_winner": int(np.argmin(score_a)), "new_winner": winner,
                         "extra_candidate_update_l2": [float(np.linalg.norm(after[i] - before[i])) for i in (0, 1)],
                         "extra_candidate_old_scores": score_a[:2].tolist(),
                         "extra_candidate_new_scores": score_b[:2].tolist(),
                         "extra_candidate_score_delta": (score_b[:2] - score_a[:2]).tolist(),
                         "winner_score": float(score_b[winner]),
                         "best_extra_minus_winner": float(min(score_b[:2]) - score_b[winner])})
    return {"schema": NAME, "rows": rows, "old_report_sha256": OLD_SHA, "new_report_sha256": NEW_SHA,
            "scope": "known-root descriptive saved data; hashes do not prove complete runtime input equivalence",
            "fairness_proven": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    result = analyze(ROOT / "results/p1_four_arm_hooked_cuda_20261030_main_RUN01",
                     ROOT / "results/p1_four_arm_spare_budget_cuda_20261030_RUN01")
    result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with output.open("xb") as stream:
        need(stream.write(raw) == len(raw), "short write")
    print(json.dumps({"rows": len(result["rows"]), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
