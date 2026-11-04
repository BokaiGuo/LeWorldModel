"""Independently recompute FD endpoint Q using already authenticated goal embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FD = ROOT / "results/reacher_pc_fd_suffix_cuda_20261102_v1/report.json"
GOAL = ROOT / "results/reacher_pc_constrained_goal_q_replay_20261102_v1/report.json"
PLAN = ROOT / "results/reacher_pc_fd_suffix_cuda_20261102_PLAN_v1.json"
PREFLIGHT = ROOT / "results/reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json"
FD_SHA = "b76a55c6b6f8e13e7c4cbba0887e6ec6babc062c1a3245ddabf3991d39b00401"
PLAN_SHA = "12ff26c6c052e02fe487bde8affd21104b09b8950f6a5340f5538fea766a2307"
GOAL_SHA = "5b36c1ac7b35923fcd8f5d28e22340371ab9eb67d96acd72644cb573d8e6f066"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(fd, goal, q, plan):
    if not (fd["plan_postcheck"] and fd["frozen_parameters_unchanged"]
            and goal["frozen_parameters_unchanged"]
            and fd["parameter_hash_before"] == fd["parameter_hash_after"]
            == goal["parameter_hash_before"] == goal["parameter_hash_after"]
            == plan["parameters_sha256"]):
        raise ValueError("frozen model/source binding")
    if not ([r["seed"] for r in fd["rows"]] == [r["seed"] for r in goal["rows"]]
            == [r["seed"] for r in q["rows"]] == [20263160, 20263161]):
        raise ValueError("root roster")
    rows = []
    for f, g, r in zip(fd["rows"], goal["rows"], q["rows"], strict=True):
        if not (f["status"] == "complete" and g["allclose_1e-7_plus_1e-5_relative"]
                and f["goal_rgb_sha256"] == g["goal_rgb_sha256"] == r["goal_rgb_sha256"]):
            raise ValueError("goal identity or source replay")
        x = np.asarray(f["predicted_endpoints"], dtype=np.float64)
        target = np.asarray(g["target_coordinate"], dtype=np.float64).reshape(-1)
        gram = np.asarray(r["goal_metric"]["gram"], dtype=np.float64)
        expected = np.asarray([a["final_costs"] for a in f["compiler_receipt"]["proposal_rows"]])
        if (x.shape != (9, 5, 192) or target.shape != (192,)
                or gram.shape != (192, 192) or expected.shape != (9, 5)
                or not all(np.isfinite(a).all() for a in (x, target, gram, expected))
                or not np.allclose(gram, gram.T, atol=1e-12, rtol=1e-12)):
            raise ValueError("invalid arrays")
        residual = (x - target).reshape(-1, 192)
        eigenvalues = np.linalg.eigvalsh(gram)
        if eigenvalues.min() < -1e-10 * max(1., np.abs(eigenvalues).max()):
            raise ValueError("goal metric not positive semidefinite")
        values, vectors = np.linalg.eigh(gram)
        weights = .01 / (np.maximum(values, 0.) + .01)
        scores = ((residual @ vectors) ** 2 @ weights / 192).reshape(9, 5)
        if not np.allclose(scores, expected, atol=1e-7, rtol=1e-5):
            raise ValueError("recomputed Q mismatch")
        rows.append({"seed": f["seed"], "endpoints": 45,
                     "maximum_absolute_error": float(np.max(np.abs(scores - expected)))})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(FD) != FD_SHA or digest(PLAN) != PLAN_SHA:
        raise ValueError("sealed FD report or plan changed")
    fd, goal, plan, pf = (json.loads(path.read_text()) for path in (FD, GOAL, PLAN, PREFLIGHT))
    if (digest(PREFLIGHT) != plan["preflight_sha256"] or digest(GOAL) != GOAL_SHA
            or goal["preflight_sha256"] != plan["preflight_sha256"]):
        raise ValueError("preflight or saved goal replay mismatch")
    from experiments.spectral_reachability import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight
    if digest(preflight.Q_PATH) != plan["q_sha256"]:
        raise ValueError("Q source mismatch")
    q = json.loads(preflight.Q_PATH.read_text())
    result = {"status": "endpoint_scores_recomputed_from_saved_goal",
              "rows": audit(fd, goal, q, plan), "fd_sha256": FD_SHA,
              "goal_replay_sha256": digest(GOAL), "plan_sha256": PLAN_SHA,
              "q_sha256": plan["q_sha256"], "model_inference": False,
              "independent_goal_encoding": False, "P1_admission": False}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": result["status"], "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
