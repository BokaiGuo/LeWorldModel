"""Fixed task/reset with independent optimizer seeds; trace native executed actions."""
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for solver_seed in (42, 1042, 2042):
        name = f"pc_split_seed_action_audit_task42_solver{solver_seed}_20261105_codex"
        output = root / "results" / name
        trace = output.with_suffix(".jsonl")
        if output.exists() or trace.exists():
            raise FileExistsError(f"refusing to overwrite {output} or {trace}")
        environment = os.environ.copy()
        environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
        print(f"Native CEM task seed 42, solver seed {solver_seed}: action audit", flush=True)
        subprocess.run([
            sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
            "policy=pusht/lewm", "seed=42", f"solver.seed={solver_seed}",
            "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
            f"output.dir={output}",
        ], cwd=root, env=environment, check=True)
        if not (output / "eval_metrics.json").is_file():
            raise RuntimeError(f"missing metrics in {output}")


if __name__ == "__main__":
    main()
