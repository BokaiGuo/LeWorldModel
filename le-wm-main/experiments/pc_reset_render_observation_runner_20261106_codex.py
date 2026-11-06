"""Run one predeclared first-observation development arm."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dataset", "render"), required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("tag must be alphanumeric")
    root = Path(__file__).resolve().parents[1]
    stem = f"pc_reset_render_observation_{args.mode}_{args.tag}_20261106_codex"
    output = root / "results" / stem
    trace, audit, initial, wall = (root / "results" / f"{stem}.{suffix}" for suffix in
                                   ("trace.jsonl", "audit.jsonl", "initial.json", "wall.txt"))
    if any(p.exists() for p in (output, trace, audit, initial, wall)):
        raise FileExistsError(stem)
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    mean = "[" + ",".join(repr(float(x)) for x in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(x)) for x in scaler.scale_) + "]"
    environment = os.environ.copy()
    environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace),
                       PC_INITIAL_OBSERVATION_MODE=args.mode, PC_INITIAL_OBSERVATION_AUDIT=str(initial))
    print(f"First-observation development: {args.mode}, 10 tasks, CEM 105x6.", flush=True)
    started = time.perf_counter()
    subprocess.run([sys.executable, "-m", "experiments.pc_reset_render_observation_dev_20261106_codex",
                    "policy=pusht/lewm", "seed=42", "solver.seed=42", "eval.num_eval=10",
                    "solver._target_=experiments.pc_feedback_cem_20261106_codex.FeedbackCEM",
                    "+solver.selection=mean", "solver.num_samples=105", "solver.n_steps=6", "solver.topk=10",
                    f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
                    f"+solver.audit_path={audit}", f"output.dir={output}"], cwd=root, env=environment, check=True)
    with wall.open("x") as stream:
        stream.write(f"{time.perf_counter() - started:.9f}\n")
    if not (output / "eval_metrics.json").is_file() or not initial.is_file():
        raise ValueError("missing result or first-observation audit")


if __name__ == "__main__":
    main()
