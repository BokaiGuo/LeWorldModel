"""Record development held-out chains and corresponding true execution trace."""
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "pc_prefix_heldout_physical_seed42_20261105_codex"
    audit = output.with_suffix(".audit.jsonl")
    trace = output.with_suffix(".trace.jsonl")
    if output.exists() or audit.exists() or trace.exists():
        raise FileExistsError(f"refusing to overwrite {output}, {audit} or {trace}")
    environment = os.environ.copy()
    environment.update(PC_PHYSICS_RESET_SEED="771001", PC_PHYSICS_TRACE_PATH=str(trace))
    print("Record held-out chains and execute unchanged planning policy", flush=True)
    subprocess.run([
        sys.executable, "-m", "experiments.pc_eval_seeded_trace_20261105_codex",
        "policy=pusht/lewm", "seed=42", "solver.seed=42",
        "solver._target_=experiments.pc_prefix_heldout_physical_20261105_codex.NativePrefixHeldoutPhysical",
        "+solver.mode=conditional", f"+solver.audit_path={audit}",
        "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
        f"output.dir={output}",
    ], cwd=root, env=environment, check=True)
    if not (output / "eval_metrics.json").is_file():
        raise RuntimeError(f"missing metrics in {output}")


if __name__ == "__main__":
    main()
