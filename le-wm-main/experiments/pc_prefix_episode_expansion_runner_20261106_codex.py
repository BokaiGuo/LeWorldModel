"""Frozen two-arm same-budget episode-disjoint *development* expansion."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("cem", "prefix", "cem300x10", "cem300x30"), required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.tag and not args.tag.isalnum():
        raise ValueError("tag must be alphanumeric")
    root = Path(__file__).resolve().parents[1]
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
        episodes = np.asarray(data["episode_idx"][:]).reshape(-1)
        previous = panel_rows(data, seed=42, count=10)
        current = panel_rows(data, seed=85, count=50)
        if (len(set(episodes[current].tolist())) != 50 or
                set(episodes[current].tolist()) & set(episodes[previous].tolist())):
            raise ValueError("episode disjointness check failed")
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
        raise ValueError("invalid normalizer")
    mean = "[" + ",".join(repr(float(x)) for x in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(x)) for x in scaler.scale_) + "]"
    stem = f"pc_episode_expansion_{args.arm}_task85_solver42_dev1106"
    if args.tag:
        stem += f"_{args.tag}"
    output = root / "results" / stem
    trace, audit, wall = (root / "results" / f"{stem}.{extension}" for extension in
                          ("trace.jsonl", "audit.jsonl", "wall.txt"))
    if any(p.exists() for p in (output, trace, audit, wall)):
        raise FileExistsError(stem)
    environment = os.environ.copy()
    environment.update(PC_PHYSICS_RESET_SEED="871001", PC_PHYSICS_TRACE_PATH=str(trace))
    if args.arm in ("cem", "cem300x10", "cem300x30"):
        samples, steps, topk = {"cem": (105, 6, 10), "cem300x10": (300, 10, 30),
                                "cem300x30": (300, 30, 30)}[args.arm]
        solver = ["solver._target_=experiments.pc_feedback_cem_20261106_codex.FeedbackCEM",
                  "+solver.selection=mean", f"solver.num_samples={samples}", f"solver.n_steps={steps}",
                  f"solver.topk={topk}"]
        budget = samples * steps + 2
    else:
        solver = ["solver._target_=experiments.pc_explicit_prefix_program_20261106_codex.ExplicitPrefixProgram",
                  "+solver.prefix_mode=lookahead", "solver.num_samples=100", "solver.n_steps=3", "solver.topk=10"]
        budget = 632
    print(f"Episode-disjoint development expansion {args.arm}: 50 tasks, {budget} scores per plan.", flush=True)
    started = time.perf_counter()
    subprocess.run([sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
                    "policy=pusht/lewm", "seed=85", "solver.seed=42", "eval.num_eval=50", *solver,
                    f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
                    f"+solver.audit_path={audit}", f"output.dir={output}"],
                   cwd=root, env=environment, check=True)
    with wall.open("x", encoding="utf-8") as stream:
        stream.write(f"{time.perf_counter() - started:.9f}\n")
    if not (output / "eval_metrics.json").exists():
        raise ValueError("missing evaluation metrics")


if __name__ == "__main__":
    main()
