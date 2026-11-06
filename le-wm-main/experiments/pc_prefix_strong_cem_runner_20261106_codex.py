"""Execute one preregistered bounded CEM development budget."""
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
    parser.add_argument("--budget", choices=("105x6", "315x2", "300x10", "300x30"), required=True)
    parser.add_argument("--solver-seed", type=int, choices=(42, 1042, 2042), required=True)
    args = parser.parse_args()
    if args.budget in ("300x10", "300x30") and args.solver_seed != 42:
        raise ValueError("large budgets predeclared for seed 42 only")
    root = Path(__file__).resolve().parents[1]
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
        raise ValueError("invalid action normalizer")
    mean = "[" + ",".join(repr(float(x)) for x in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(x)) for x in scaler.scale_) + "]"
    samples, steps = map(int, args.budget.split("x"))
    topk = {"105x6": 10, "315x2": 31, "300x10": 30, "300x30": 30}[args.budget]
    name = f"pc_strong_cem_{args.budget}_solver{args.solver_seed}_dev1106"
    output = root / "results" / name
    trace = root / "results" / f"{name}.trace.jsonl"
    audit = root / "results" / f"{name}.audit.jsonl"
    wall = root / "results" / f"{name}.wall.txt"
    if any(path.exists() for path in (output, trace, audit, wall)):
        raise FileExistsError(name)
    environment = os.environ.copy()
    environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
    print(f"Run CEM {args.budget}, solver seed {args.solver_seed}, 10 shared development tasks.", flush=True)
    start = time.perf_counter()
    subprocess.run([
        sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
        "policy=pusht/lewm", "seed=42", f"solver.seed={args.solver_seed}", "eval.num_eval=10",
        "solver._target_=experiments.pc_feedback_cem_20261106_codex.FeedbackCEM",
        "+solver.selection=mean", f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
        f"+solver.audit_path={audit}", f"solver.num_samples={samples}",
        f"solver.n_steps={steps}", f"solver.topk={topk}", f"output.dir={output}",
    ], cwd=root, env=environment, check=True)
    with wall.open("x", encoding="utf-8") as stream:
        stream.write(f"{time.perf_counter() - start:.9f}\n")
    if not (output / "eval_metrics.json").is_file():
        raise RuntimeError("missing evaluation metrics")


if __name__ == "__main__":
    main()
