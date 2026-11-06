"""Run one predeclared development audit without overwriting prior outputs."""
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "results" / "pc_prefix_candidate_set_audit_seed42_20261105_codex"
    audit = output.with_suffix(".jsonl")
    if output.exists() or audit.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {audit}")
    environment = os.environ.copy()
    environment["PC_PHYSICS_RESET_SEED"] = "771001"
    subprocess.run([
        sys.executable, "-m", "experiments.pc_eval_seeded_reset_20261105_codex",
        "policy=pusht/lewm", "seed=42",
        "solver._target_=experiments.pc_prefix_candidate_set_audit_20261105_codex.NativePrefixCandidateSetAudit",
        "+solver.mode=conditional", f"+solver.audit_path={audit}",
        "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
        f"output.dir={output}",
    ], cwd=root, env=environment, check=True)
    if not (output / "eval_metrics.json").is_file():
        raise RuntimeError(f"missing metrics: {output}")


if __name__ == "__main__":
    main()
