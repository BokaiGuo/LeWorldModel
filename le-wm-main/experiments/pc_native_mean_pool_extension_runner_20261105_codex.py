"""Run the remaining predeclared development arms without overwriting runs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for seed, mode in ((43, "sample"), (43, "joint"), (44, "mean"),
                       (44, "sample"), (44, "joint")):
        stem = f"pc_native_mean_pool_{mode}_seed{seed}_20261105_codex"
        output = root / "results" / stem
        audit = output.with_suffix(".jsonl")
        if output.exists() or audit.exists():
            raise FileExistsError(f"refusing to overwrite {output} or {audit}")
        command = [
            sys.executable, "eval.py", "policy=pusht/lewm", f"seed={seed}",
            "solver._target_=experiments.pc_native_mean_selection_20261105_codex.NativeMeanSelection",
            f"+solver.mode={mode}", f"+solver.audit_path={audit}",
            "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
            f"output.dir={output}",
        ]
        print(f"Running seed={seed} mode={mode}; evaluating selection, not response.", flush=True)
        subprocess.run(command, cwd=root, check=True)
        if not (output / "eval_metrics.json").is_file():
            raise RuntimeError(f"missing metrics: {output}")
    print("All remaining extension arms finished.", flush=True)


if __name__ == "__main__":
    main()
