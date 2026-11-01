"""Run the fixed two-root, three-prefix development roster serially on CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MODULE = "experiments.spectral_reachability.p1_three_prefix_own_host_20261101_main"
NAME = "p1_three_prefix_own_host_20261101_main"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true")
    args = parser.parse_args()
    if not args.authorize_host_cuda or not args.tag.isalnum():
        raise ValueError("authorization and simple exclusive tag required")
    roster = [(root, role) for root in (0, 1) for role in ("selected", "zero", "greedy")]
    paths = []
    for root, role in roster:
        prefix = ROOT / "results" / f"{NAME}_root{root}_{role}"
        prepared = Path(str(prefix) + f"_PREP_{args.tag}.json")
        output = Path(str(prefix) + f"_RUN_{args.tag}")
        if prepared.exists() or prepared.is_symlink() or output.exists() or output.is_symlink():
            raise ValueError("exclusive roster output already exists")
        paths.append((root, role, prepared, output))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for root, role, prepared, output in paths:
        base = [sys.executable, "-m", MODULE, "--root", str(root), "--role", role]
        print(json.dumps({"root": root, "role": role, "state": "preparing"}), flush=True)
        subprocess.run(base + ["prepare", "--output", str(prepared)],
                       cwd=ROOT, env=env, check=True)
        digest = hashlib.sha256(prepared.read_bytes()).hexdigest()
        subprocess.run(base + ["run", "--prepared", str(prepared), "--sha256", digest,
                              "--output", str(output), "--authorization",
                              "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS"],
                       cwd=ROOT, env=env, check=True)
        print(json.dumps({"root": root, "role": role, "state": "recorded"}), flush=True)


if __name__ == "__main__":
    main()
