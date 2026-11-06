"""Execute one arm of the predeclared explicit-prefix development panel."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("full_cem", "uniform", "lookahead"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--solver-seed", type=int, default=42)
    args = parser.parse_args()
    if not args.tag.isidentifier():
        raise ValueError("invalid tag")
    root = Path(__file__).resolve().parents[1]
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
        raise ValueError("invalid normalizer")
    mean = "[" + ",".join(repr(float(x)) for x in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(x)) for x in scaler.scale_) + "]"
    name = f"pc_explicit_prefix_{args.arm}_{args.tag}"
    output = root / "results" / name
    trace = root / "results" / f"{name}.trace.jsonl"
    audit = root / "results" / f"{name}.audit.jsonl"
    if any(path.exists() for path in (output, trace, audit)):
        raise FileExistsError(name)
    environment = os.environ.copy()
    environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
    if args.arm == "full_cem":
        solver = ["solver._target_=experiments.pc_feedback_cem_20261106_codex.FeedbackCEM",
                  "+solver.selection=mean", "solver.num_samples=210", "solver.topk=21"]
    else:
        solver = ["solver._target_=experiments.pc_explicit_prefix_program_20261106_codex.ExplicitPrefixProgram",
                  f"+solver.prefix_mode={args.arm}", "solver.num_samples=100", "solver.topk=10"]
    print(f"Explicit-prefix development: {args.arm}, 632 complete-chain scores per planning call.", flush=True)
    subprocess.run([
        sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
        "policy=pusht/lewm", "seed=42", f"solver.seed={args.solver_seed}", "eval.num_eval=10",
        *solver, f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
        f"+solver.audit_path={audit}", "solver.n_steps=3", f"output.dir={output}",
    ], cwd=root, env=environment, check=True)
    if not (output / "eval_metrics.json").is_file():
        raise RuntimeError("missing evaluation metrics")


if __name__ == "__main__":
    main()
