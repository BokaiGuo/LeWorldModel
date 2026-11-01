"""Execute every unattempted cross-goal slot, retaining all failure exit codes."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ROSTER = [(0, "random"), (0, "immediate_progress"),
          (1, "structured"), (1, "native"), (1, "random"), (1, "immediate_progress")]


if __name__ == "__main__":
    if sys.argv[1:] != ["--authorize-host-cuda"]:
        raise ValueError("explicit host CUDA authorization")
    output = ROOT / "results/p1_cross_goal_remaining_20261101_main_MAIN01.json"
    if output.exists():
        raise ValueError("exclusive batch output")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    rows = []
    for root, arm in ROSTER:
        result = subprocess.run([sys.executable, "-m",
                                 "experiments.spectral_reachability.p1_cross_goal_shard_20261101_main",
                                 "--root", str(root), "--arm", arm, "--tag", "MAIN01",
                                 "--authorize-host-cuda"], cwd=ROOT, env=env, check=False)
        rows.append(dict(root=root, arm=arm, returncode=result.returncode))
        print(json.dumps(rows[-1]), flush=True)
    with output.open("x") as stream:
        json.dump(dict(rows=rows, P1_admission=False, failures_retained=True), stream, indent=2)
    raise SystemExit(int(any(r["returncode"] for r in rows)))
