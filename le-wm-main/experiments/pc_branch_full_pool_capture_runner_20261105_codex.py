"""Run the predeclared one-plan capture using the existing 50-task panel."""
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler


def main():
    root = Path(__file__).resolve().parents[1]
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    mean = "[" + ",".join(repr(float(x)) for x in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(x)) for x in scaler.scale_) + "]"
    for mode in ("lookahead", "effect"):
        name = f"pc_branch_full_pool_{mode}_seed42_20261105_codex"
        output = root / "results" / name
        capture = root / "results" / f"{name}.capture"
        trace = root / "results" / f"{name}.trace.jsonl"
        audit = root / "results" / f"{name}.audit.jsonl"
        if any(p.exists() for p in (output, capture, trace, audit)):
            raise FileExistsError(name)
        environment = os.environ.copy()
        environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
        print(f"Capture {mode}: first plan, first fixed task; same-query embeddings", flush=True)
        subprocess.run([
            sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
            "policy=pusht/lewm", "seed=42", "solver.seed=42",
            "solver._target_=experiments.pc_branch_full_pool_capture_20261105_codex.FullPoolCapture",
            f"+solver.comparison={mode}", f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
            f"+solver.capture_dir={capture}", f"+solver.audit_path={audit}",
            "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
            f"output.dir={output}",
        ], cwd=root, env=environment, check=True)
        if len(list(capture.glob("call_*.pt"))) != 20 or not (output / "eval_metrics.json").is_file():
            raise ValueError("incomplete capture")


if __name__ == "__main__":
    main()
