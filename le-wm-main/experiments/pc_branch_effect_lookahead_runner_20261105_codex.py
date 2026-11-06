"""Run matched branch-effect development comparisons without overwriting results."""
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
    parser.add_argument("--tag", default="codex")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1042, 2042])
    parser.add_argument("--modes", nargs="+", choices=["lookahead", "effect"],
                        default=["lookahead", "effect"])
    args = parser.parse_args()
    if not args.tag.isidentifier():
        raise ValueError("tag must be a valid identifier")
    root = Path(__file__).resolve().parents[1]
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
        raise ValueError("invalid action normalizer")
    mean = "[" + ",".join(repr(float(x)) for x in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(x)) for x in scaler.scale_) + "]"
    for seed in args.seeds:
        for mode in args.modes:
            name = f"pc_branch_effect_{mode}_task42_solver{seed}_20261105_{args.tag}"
            output = root / "results" / name
            trace, audit = output.with_suffix(".trace.jsonl"), output.with_suffix(".audit.jsonl")
            if output.exists() or trace.exists() or audit.exists():
                raise FileExistsError(f"refusing to overwrite outputs for {name}")
            environment = os.environ.copy()
            environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
            print(f"{mode}: task seed 42, solver seed {seed}", flush=True)
            subprocess.run([
                sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
                "policy=pusht/lewm", "seed=42", f"solver.seed={seed}",
                "solver._target_=experiments.pc_branch_effect_lookahead_20261105_codex.BranchEffectLookahead",
                f"+solver.comparison={mode}", f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
                f"+solver.audit_path={audit}", "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
                f"output.dir={output}",
            ], cwd=root, env=environment, check=True)
            if not (output / "eval_metrics.json").is_file():
                raise RuntimeError(f"missing metrics in {output}")


if __name__ == "__main__":
    main()
