"""Run predeclared additional fixed-reset development seeds."""
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PC_PHYSICS_RESET_SEED"] = "771001"
    for seed in (43, 44):
        for mode in ("native", "conditional", "shared"):
            output = root / "results" / f"pc_seeded_prefix_{mode}_seed{seed}_20261105_codex"
            audit = output.with_suffix(".jsonl")
            if output.exists() or audit.exists():
                raise FileExistsError(f"refusing to overwrite {output} or {audit}")
            command = [
                sys.executable, "-m", "experiments.pc_eval_seeded_reset_20261105_codex",
                "policy=pusht/lewm", f"seed={seed}", "solver.num_samples=300",
                "solver.n_steps=10", "solver.topk=30", f"output.dir={output}",
            ]
            if mode != "native":
                command.extend([
                    "solver._target_=experiments.pc_native_prefix_conditional_20261105_codex.NativePrefixConditional",
                    f"+solver.mode={mode}", f"+solver.audit_path={audit}",
                ])
            print(f"fixed reset prefix-tail development seed={seed} mode={mode}", flush=True)
            subprocess.run(command, cwd=root, env=environment, check=True)
            if not (output / "eval_metrics.json").is_file():
                raise RuntimeError(f"missing metrics: {output}")


if __name__ == "__main__":
    main()
