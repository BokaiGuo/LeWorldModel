"""Fixed eight-slot push-and-return exploration, failures retained."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    if sys.argv[1:] != ["--authorize-host-cuda"]:
        raise ValueError("explicit host CUDA authorization")
    target = ROOT / "results/p1_return_target_cuda_20261101_main_MAIN06.json"
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    targets = json.loads(target.read_bytes())["rows"]
    output = ROOT / "results/p1_return_goal_batch_20261101_main_MAIN01.json"
    if output.exists():
        raise ValueError("exclusive batch output")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    rows = []
    for root in (0, 1):
        for arm in ("structured", "native", "random", "immediate_progress"):
            if targets[root]["native_initial"]["initial_success"]:
                rows.append(dict(root=root, arm=arm, returncode=0, executed=False,
                                 status="ineligible_already_successful_at_initial_state"))
                print(json.dumps(rows[-1]), flush=True)
                continue
            result = subprocess.run([sys.executable, "-m",
                                     "experiments.spectral_reachability.p1_return_goal_shard_20261101_main",
                                     "--root", str(root), "--arm", arm, "--target", str(target),
                                     "--target-sha", digest, "--tag", "MAIN01", "--authorize-host-cuda"],
                                    cwd=ROOT, env=env, check=False)
            rows.append(dict(root=root, arm=arm, returncode=result.returncode, executed=True))
            print(json.dumps(rows[-1]), flush=True)
    with output.open("x") as stream:
        json.dump(dict(rows=rows, target_sha256=digest, P1_admission=False), stream, indent=2)
    raise SystemExit(int(any(r["returncode"] for r in rows)))
