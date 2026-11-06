"""Trace a controlled mean-selected run and audit first-plan replay parity."""
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    result = root / "results" / "pc_seeded_trace_mean_seed42_20261105_codex"
    audit = result.with_suffix(".jsonl")
    trace = root / "results" / "pc_seeded_trace_physics_seed42_20261105_codex.jsonl"
    parity = root / "results" / "pc_seeded_trace_parity_seed42_20261105_codex.json"
    for path in (result, audit, trace, parity):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    environment = os.environ.copy()
    environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
    evaluation = [
        sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
        "policy=pusht/lewm", "seed=42",
        "solver._target_=experiments.pc_native_mean_selection_20261105_codex.NativeMeanSelection",
        "+solver.mode=mean", f"+solver.audit_path={audit}",
        "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
        f"output.dir={result}",
    ]
    print("Running traced controlled-mean evaluation.", flush=True)
    subprocess.run(evaluation, cwd=root, env=environment, check=True)
    print("Auditing recorded actual steps against first-plan replay.", flush=True)
    subprocess.run([
        sys.executable, "-m", "experiments.pc_seeded_candidate_replay_20261105_codex",
        "--audit", str(audit), "--trace", str(trace), "--output", str(parity),
    ], cwd=root, check=True)


if __name__ == "__main__":
    main()
