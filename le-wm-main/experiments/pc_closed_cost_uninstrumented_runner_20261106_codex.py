"""Repeat fixed-panel timings without trace or GPU timing instrumentation."""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from experiments.pc_closed_cost_repeat_runner_20261106_codex import CONFIGS, MEAN, SCALE, ROOT


def main():
    for repeat in (1, 2, 3):
        for name, overrides in CONFIGS.items():
            stem = f"pc_cost_uninstrumented_{name}_r{repeat}"
            output = ROOT / f"results/{stem}.run"
            log = ROOT / f"results/{stem}.log"
            receipt = ROOT / f"results/{stem}.json"
            if any(p.exists() for p in (output, log, receipt)):
                raise FileExistsError(stem)
            env = os.environ.copy()
            env["PC_PHYSICS_RESET_SEED"] = "871001"
            command = [sys.executable, "-m", "experiments.pc_eval_seeded_reset_20261105_codex",
                       "policy=pusht/lewm", "seed=42", "solver.seed=42", "eval.num_eval=10",
                       *overrides, MEAN, SCALE, f"output.dir={output}"]
            print(f"Running {stem}", flush=True)
            start = time.perf_counter()
            with log.open("x") as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            elapsed = time.perf_counter() - start
            text = (output / "pusht_results.txt").read_text()
            matches = re.findall(r"evaluation_time: ([0-9.eE+-]+) seconds", text)
            if len(matches) != 1:
                raise ValueError("expected a single evaluator timer")
            metrics = json.loads((output / "eval_metrics.json").read_text())
            with receipt.open("x") as stream:
                json.dump({"process_wall_s": elapsed, "evaluation_wall_s": float(matches[0]),
                           "episode_successes": metrics["episode_successes"],
                           "limitations": "video output enabled; no per-step trace or explicit GPU synchronization"},
                          stream, indent=2, allow_nan=False)
            print(f"Completed {stem}", flush=True)


if __name__ == "__main__":
    main()
