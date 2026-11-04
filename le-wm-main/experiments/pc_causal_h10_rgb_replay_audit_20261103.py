"""Replay every fixed H10 branch and compare each recorded RGB observation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.pc_pointmaze_obstacle_rgb_spectral_dev_20261103 import ROOTS
from experiments.pointmaze_native_rgb_bridge_20261102 import NativeRGBBridge
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    array_path = args.report.with_suffix(".npz")
    if sha256_file(array_path) != report["arrays_sha256"]:
        raise ValueError("arrays changed")
    print("Re-execute 36 H10 branches and verify all 648 RGB frames.", flush=True)
    begun, checks = time.monotonic(), 0
    with np.load(array_path, allow_pickle=False) as arrays:
        starts, actions = arrays["starts"], arrays["sequences"]
        if starts.shape != (36, 4) or actions.shape != (36, 17, 2):
            raise ValueError("unexpected actions")
        for k, root in enumerate(ROOTS):
            for i in range(9):
                idx = 9*k+i
                history = arrays[f"r{root}_p{i}_history"]
                if history.shape != (18, 224, 224, 3):
                    raise ValueError("unexpected history")
                with NativeRGBBridge(starts[idx]) as env:
                    if not np.array_equal(env.initial_rgb, history[0]):
                        raise ValueError(f"reset mismatch {root}/{i}")
                    checks += 1
                    for step, action in enumerate(actions[idx], 1):
                        if not np.array_equal(env.step(action[None]), history[step]):
                            raise ValueError(f"frame mismatch {root}/{i}/{step}")
                        checks += 1
    if checks != 648:
        raise RuntimeError("incomplete replay")
    result = {"schema": "pc_causal_h10_rgb_replay_audit.v1", "trajectories": 36,
              "physical_steps": 612, "rgb_frames_verified": checks,
              "report_sha256": sha256_file(args.report), "arrays_sha256": sha256_file(array_path),
              "script_sha256": sha256_file(Path(__file__)),
              "scope": "same-backend deterministic RGB replay, not independent root",
              "wall_seconds": time.monotonic() - begun}
    with args.output.open("x") as out:
        json.dump(result, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
