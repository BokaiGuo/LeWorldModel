"""Replay every saved action and require all RGB observations to match."""
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
        raise ValueError("saved arrays changed")
    print("Independently replay 36 fixed programs and verify every saved RGB frame.", flush=True)
    started, checks = time.monotonic(), 0
    with np.load(array_path, allow_pickle=False) as arrays:
        starts, actions = arrays["starts"], arrays["sequences"]
        if starts.shape != (36, 4) or actions.shape != (36, 12, 2):
            raise ValueError("invalid rollout array shapes")
        for k, root in enumerate(ROOTS):
            for i in range(9):
                history = arrays[f"r{root}_p{i}_history"]
                if history.shape != (13, 224, 224, 3):
                    raise ValueError("invalid RGB history")
                idx = k * 9 + i
                with NativeRGBBridge(starts[idx]) as env:
                    if not np.array_equal(env.initial_rgb, history[0]):
                        raise ValueError(f"reset RGB mismatch {root}/{i}")
                    checks += 1
                    for step, action in enumerate(actions[idx], 1):
                        if not np.array_equal(env.step(action[None]), history[step]):
                            raise ValueError(f"RGB mismatch {root}/{i}/{step}")
                        checks += 1
        if checks != 36 * 13:
            raise RuntimeError("incomplete frame audit")
    result = {"schema": "pc_causal_history_rgb_replay_audit.v1", "trajectories": 36,
              "physical_steps": 432, "rgb_frames_verified": checks,
              "report_sha256": sha256_file(args.report), "arrays_sha256": sha256_file(array_path),
              "script_sha256": sha256_file(Path(__file__)),
              "scope": "same-backend deterministic RGB replay, not independent root or causal proof",
              "wall_seconds": time.monotonic() - started}
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
