"""Run action-constrained CEM against the predeclared fixed-task panel."""
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parents[1]
    dataset = Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5")
    with h5py.File(dataset, "r") as data:
        actions = np.asarray(data["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError("invalid dataset action normalizer")
    mean = "[" + ",".join(repr(float(x)) for x in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(x)) for x in scaler.scale_) + "]"
    for seed in (42, 1042, 2042):
        name = f"pc_bounded_cem_task42_solver{seed}_20261105_codex"
        output = root / "results" / name
        trace, audit = output.with_suffix(".trace.jsonl"), output.with_suffix(".audit.jsonl")
        if output.exists() or trace.exists() or audit.exists():
            raise FileExistsError(f"refusing to overwrite outputs for {name}")
        environment = os.environ.copy()
        environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
        print(f"Bounded CEM task seed 42, solver seed {seed}: physical action contract", flush=True)
        subprocess.run([
            sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
            "policy=pusht/lewm", "seed=42", f"solver.seed={seed}",
            "solver._target_=experiments.pc_bounded_cem_20261105_codex.BoundedCEM",
            f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
            f"+solver.audit_path={audit}",
            "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
            f"output.dir={output}",
        ], cwd=root, env=environment, check=True)
        if not (output / "eval_metrics.json").is_file():
            raise RuntimeError(f"missing metrics in {output}")


if __name__ == "__main__":
    main()
