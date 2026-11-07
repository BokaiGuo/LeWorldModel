"""Run the single fixed episode-85 scored-incumbent development arm."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

from experiments import pc_explicit_prefix_scored_incumbent_20261107_codex as solver
from experiments import pc_prefix_episode_expansion_runner_20261106_codex as source
from experiments.pc_input_permissions import file_hash
from experiments.pc_standard_pusht_reference_replay_20261105_codex import panel_rows

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_EXPLICIT_PREFIX_SCORED_INCUMBENT_PROTOCOL_20261107_CODEX.md"
STEM = ROOT / "results/pc_pusht_explicit_prefix_scored_incumbent_20261107_codex"


def main():
    output = STEM
    trace, audit, manifest = [STEM.with_suffix(ext) for ext in (".trace.jsonl", ".audit.jsonl", ".manifest.json")]
    if any(p.exists() for p in (output, trace, audit, manifest)):
        raise FileExistsError("output path exists")
    with h5py.File("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5", "r") as data:
        actions = np.asarray(data["action"][:])
        episodes = np.asarray(data["episode_idx"][:]).reshape(-1)
        old, current = panel_rows(data, seed=42, count=10), panel_rows(data, seed=85, count=50)
        if len(set(episodes[current].tolist())) != 50 or set(episodes[current]) & set(episodes[old]):
            raise ValueError("panel overlap")
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or (scaler.scale_ <= 0).any():
        raise ValueError("invalid normalizer")
    mean = "[" + ",".join(repr(float(x)) for x in scaler.mean_) + "]"
    scale = "[" + ",".join(repr(float(x)) for x in scaler.scale_) + "]"
    environment = os.environ.copy()
    environment.update(PC_PHYSICS_RESET_SEED="871001", PC_PHYSICS_TRACE_PATH=str(trace))
    print("Run exposed seed-85 PushT tasks with all-scored-chain incumbent.", flush=True)
    started = time.perf_counter()
    subprocess.run([sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
                    "policy=pusht/lewm", "seed=85", "solver.seed=42", "eval.num_eval=50",
                    "solver._target_=experiments.pc_explicit_prefix_scored_incumbent_20261107_codex.ExplicitPrefixScoredIncumbent",
                    "+solver.prefix_mode=lookahead", "solver.num_samples=100", "solver.n_steps=3", "solver.topk=10",
                    f"+solver.action_mean={mean}", f"+solver.action_scale={scale}",
                    f"+solver.audit_path={audit}", f"output.dir={output}"],
                   cwd=ROOT, env=environment, check=True)
    if not (output / "eval_metrics.json").exists():
        raise RuntimeError("missing evaluation metrics")
    with manifest.open("x") as stream:
        json.dump({"protocol_sha256": file_hash(PROTOCOL), "solver_sha256": file_hash(solver.__file__),
                   "runner_sha256": file_hash(__file__), "previous_runner_sha256": file_hash(source.__file__),
                   "trace_sha256": file_hash(trace), "audit_sha256": file_hash(audit),
                   "metrics_sha256": file_hash(output / "eval_metrics.json"),
                   "subprocess_wall_seconds": time.perf_counter() - started,
                   "scope": "exposed episode-85 development; normalizer and training independence unverified"},
                  stream, indent=2)


if __name__ == "__main__":
    main()
