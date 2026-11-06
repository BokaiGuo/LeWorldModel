"""Run controlled seeded-reset development arms with exclusive output paths."""
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PC_PHYSICS_RESET_SEED"] = "771001"
    arms = ("native", "native_repeat", "response", "random", "mean", "sample", "joint")
    for arm in arms:
        stem = f"pc_seeded_reset_{arm}_seed42_20261105_codex"
        output = root / "results" / stem
        audit = output.with_suffix(".jsonl")
        if output.exists() or audit.exists():
            raise FileExistsError(f"refusing to overwrite {output} or {audit}")
        command = [
            sys.executable, "-m", "experiments.pc_eval_seeded_reset_20261105_codex",
            "policy=pusht/lewm", "seed=42", "solver.num_samples=300",
            "solver.n_steps=10", "solver.topk=30", f"output.dir={output}",
        ]
        if arm in ("response", "random"):
            command.extend([
                "solver._target_=experiments.pc_native_internal_response_20261105_codex.NativeInternalResponse",
                f"+solver.mode={arm}", f"+solver.audit_path={audit}",
            ])
        elif arm in ("mean", "sample", "joint"):
            command.extend([
                "solver._target_=experiments.pc_native_mean_selection_20261105_codex.NativeMeanSelection",
                f"+solver.mode={arm}", f"+solver.audit_path={audit}",
            ])
        print(f"Controlled physics reset seed 771001; arm={arm}", flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)
        if not (output / "eval_metrics.json").is_file():
            raise RuntimeError(f"missing metrics: {output}")


if __name__ == "__main__":
    main()
