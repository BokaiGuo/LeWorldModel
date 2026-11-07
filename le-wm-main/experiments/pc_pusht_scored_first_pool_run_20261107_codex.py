"""Run the instrumented first-plan capture on the fixed four-task panel."""
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments.pc_scored_incumbent_real_policy_parity_20261107_codex import DATA, ROOT

BASE = ROOT / "results/pc_pusht_scored_first_pool_20261107_codex"


def main():
    if BASE.exists():
        raise FileExistsError(BASE)
    BASE.mkdir()
    with h5py.File(DATA, "r") as data:
        actions = data["action"][:]
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
        raise ValueError("invalid action scaler")
    mean = "[" + ",".join(repr(float(v)) for v in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(v)) for v in scaler.scale_) + "]"
    env = os.environ.copy()
    env.update(PC_PHYSICS_RESET_SEED="871001", PC_PHYSICS_TRACE_PATH=str(BASE / "trace.jsonl"))
    print("Capture four complete first-plan pools before evaluator replay.", flush=True)
    subprocess.run([sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
                    "policy=pusht/lewm", "seed=85", "solver.seed=42", "eval.num_eval=4",
                    "solver._target_=experiments.pc_pusht_scored_first_pool_capture_20261107_codex.ScoredFirstPoolCapture",
                    "+solver.prefix_mode=lookahead", "solver.num_samples=100", "solver.n_steps=3", "solver.topk=10",
                    f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
                    f"+solver.capture_dir={BASE / 'capture'}", f"+solver.audit_path={BASE / 'audit.jsonl'}",
                    f"output.dir={BASE / 'eval'}"], cwd=ROOT, env=env, check=True)
    captures = list((BASE / "capture").glob("call_*.pt"))
    if len(captures) != 20:
        raise ValueError(f"expected 20 first-solve query batches, got {len(captures)}")


if __name__ == "__main__":
    main()
