"""Run pinned/local pairs on fixed seeds with a three-packet replan interval."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for seed in (1, 2):
        for arm in ("local", "pinned"):
            stem = f"pc_jepa_multi_unroll_{arm}_seed{seed}_r3_step3_20261106_codex"
            log = ROOT / f"results/{stem}.log"
            if log.exists():
                raise FileExistsError(log)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["PYTHONPATH"] = os.pathsep.join((
                str(ROOT), str(ROOT / "external/jepa-wms"), str(ROOT / ".jepa-physics-site"),
                "/tmp/lewm-jepa-audit-smsyc1"))
            command = [sys.executable, "-m", "experiments.pc_jepa_multitask_unroll_parity_20261106_codex",
                       "--arm", arm, "--seed", str(seed), "--execution-packets", "3"]
            print(f"Running {stem}", flush=True)
            with log.open("x") as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            print(f"Completed {stem}", flush=True)


if __name__ == "__main__":
    main()
