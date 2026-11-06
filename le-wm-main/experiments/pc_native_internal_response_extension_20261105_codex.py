"""Execute the predeclared internal-response seed extensions sequentially."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    for seed in (43, 44):
        for mode in ("native", "response", "random"):
            stem = f"pc_native_internal_{mode}_seed{seed}_20261105_codex"
            output = root / "results" / stem
            audit = output.with_suffix(".jsonl")
            if output.exists() or audit.exists():
                raise FileExistsError(f"refusing to overwrite {output} or {audit}")
            command = [
                sys.executable, "eval.py", "policy=pusht/lewm", f"seed={seed}",
                "solver.num_samples=300", "solver.n_steps=10", "solver.topk=30",
                f"output.dir={output}",
            ]
            if mode != "native":
                command.extend([
                    "solver._target_=experiments.pc_native_internal_response_20261105_codex.NativeInternalResponse",
                    f"+solver.mode={mode}", f"+solver.audit_path={audit}",
                ])
            print(f"Running seed {seed}, {mode}; development only", flush=True)
            subprocess.run(command, cwd=root, check=True)
            if not (output / "eval_metrics.json").is_file():
                raise RuntimeError(f"missing metrics: {output}")


if __name__ == "__main__":
    main()
