"""Run paired refinement diagnostic after the unchanged principal planner."""
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
    mean = "[" + ",".join(repr(float(x)) for x in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(x)) for x in scaler.scale_) + "]"
    output = root / "results" / "pc_prefix_novel_same_refinement_seed42_20261105_codex"
    audit, trace = output.with_suffix(".audit.jsonl"), output.with_suffix(".trace.jsonl")
    if output.exists() or audit.exists() or trace.exists():
        raise FileExistsError(f"refusing to overwrite outputs for {output}")
    environment = os.environ.copy()
    environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
    print("Refine novel and ordinary-pool chains with matching separate opportunities", flush=True)
    subprocess.run([
        sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
        "policy=pusht/lewm", "seed=42", "solver.seed=42",
        "solver._target_=experiments.pc_prefix_novel_same_refinement_20261105_codex.PrefixNovelSameRefinement",
        "+solver.mode=conditional", f"+solver.audit_path={audit}",
        f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
        "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
        f"output.dir={output}",
    ], cwd=root, env=environment, check=True)
    if not (output / "eval_metrics.json").is_file():
        raise RuntimeError(f"missing metrics in {output}")


if __name__ == "__main__":
    main()
