"""Run predeclared five-task pool capture and full-panel replan logging."""
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
    parser.add_argument("--offset5", action="store_true")
    parser.add_argument("--second-replan", action="store_true")
    args = parser.parse_args()
    if args.second_replan and args.offset5:
        raise ValueError("choose one capture protocol")
    root = Path(__file__).resolve().parents[1]
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
        raise ValueError("invalid normalizer")
    mean = "[" + ",".join(repr(float(x)) for x in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(x)) for x in scaler.scale_) + "]"
    for mode in ("lookahead", "effect"):
        name = (f"pc_branch_second_replan_{mode}_seed42_20261106_codex" if args.second_replan else
                f"pc_branch_panel_offset5_{mode}_seed42_20261106_codex" if args.offset5 else
                f"pc_branch_panel_{mode}_seed42_20261105_codex")
        output, capture = root / "results" / name, root / "results" / f"{name}.capture"
        trace, audit = root / "results" / f"{name}.trace.jsonl", root / "results" / f"{name}.audit.jsonl"
        if any(x.exists() for x in (output, capture, trace, audit)):
            raise FileExistsError(name)
        environment = os.environ.copy()
        environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
        print(f"Capture predeclared panel and replans: {mode}", flush=True)
        subprocess.run([
            sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
            "policy=pusht/lewm", "seed=42", "solver.seed=42",
            ("solver._target_=experiments.pc_branch_second_replan_capture_20261106_codex.SecondReplanCapture"
             if args.second_replan else "solver._target_=experiments.pc_branch_panel_capture_20261105_codex.PanelCapture"),
            f"+solver.comparison={mode}", f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
            *(["+solver.task_indices=[5,15,25,35,45]"] if args.offset5 else []),
            f"+solver.capture_dir={capture}", f"+solver.audit_path={audit}",
            "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30", f"output.dir={output}",
        ], cwd=root, env=environment, check=True)
        if not args.second_replan and len(list(capture.glob("task_*_call_*.pt"))) != 100:
            raise ValueError("incomplete first-plan capture")


if __name__ == "__main__":
    main()
