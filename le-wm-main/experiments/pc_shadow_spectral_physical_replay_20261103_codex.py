"""Verify paired RGB replay and evaluate fixed actions without planner access."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as prior
from experiments.pc_pointmaze_shadow_rgb_bridge_20261103_codex import (
    LEGACY, RUNTIME, ShadowRGBBridge, _scene,
)

ROOT = prior.ROOT
RUN = ROOT / "results/pc_shadow_spectral_paired_dev_20261103_CODEX"
AUDIT = ROOT / "results/pc_shadow_spectral_paired_audit_20261103_CODEX.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_SHADOW_SPECTRAL_PHYSICAL_REPLAY_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_shadow_spectral_physical_replay_20261103_CODEX.json"
WORKER = Path(__file__).with_name("pc_shadow_spectral_legacy_position_worker_20261103_codex.py")
ROSTER = ((0, "r8343_a0_p2", "baseline"), (0, "r8343_a0_p2", "spectral"),
          (8, "r8345_a0_p2", "baseline"), (8, "r8345_a0_p2", "spectral"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = json.loads(AUDIT.read_text())
    if sha(RUN / "events.jsonl") != audit["journal_sha256"]:
        raise ValueError("journal hash mismatch")
    _, _, names = prior.files()
    prior.load_sealed(prior.CONTEXT)
    completed = {(e["root"], e["arm"]): e for e in audit["branch_results"]}
    starts, actions, rows = [], [], []
    with np.load(prior.SOURCE.with_suffix(".npz"), allow_pickle=False) as source:
        for index, name, arm in ROSTER:
            if names[index] != name:
                raise ValueError("root roster mismatch")
            trajectory = ROOT / completed[(name, arm)]["trajectory_path"]
            if sha(trajectory) != completed[(name, arm)]["trajectory_sha256"]:
                raise ValueError("trajectory hash mismatch")
            with np.load(trajectory, allow_pickle=False) as data:
                sequence, frames = data["actions"].copy(), data["frames"].copy()
            if sequence.shape != (17, 2) or frames.shape != (18, 224, 224, 3):
                raise ValueError("bad trajectory")
            start = source["starts"][index].copy()
            with ShadowRGBBridge(start) as bridge:
                np.testing.assert_array_equal(bridge.initial_rgb, frames[0])
                for tick, action in enumerate(sequence, 1):
                    np.testing.assert_array_equal(bridge.step(action[None]), frames[tick])
            starts.append(start)
            actions.append(sequence)
            rows.append({"root": name, "arm": arm, "rgb_frames_equal": 18,
                         "trajectory_sha256": sha(trajectory),
                         "final_rgb_score": completed[(name, arm)]["final_rgb_score"]})
            print(f"Replayed {name} {arm}: 18/18 RGB frames matched", flush=True)
    starts, actions = np.asarray(starts), np.asarray(actions)
    with tempfile.TemporaryDirectory(prefix="pc_spectral_eval_") as directory:
        folder = Path(directory)
        request, xml, response = folder / "request.npz", folder / "scene.xml", folder / "response.npz"
        np.savez_compressed(request, starts=starts, actions=actions)
        xml.write_bytes(_scene())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                            "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(WORKER), str(request), str(xml), str(response)],
                       env=env, check=True, timeout=60)
        with np.load(response, allow_pickle=False) as replay:
            if str(replay["request_sha256"].item()) != sha(request):
                raise ValueError("replay input mismatch")
            states = replay["states"].copy()
    endpoints22 = prior.h.physical.replay_position_endpoints(starts, actions)
    goal = np.asarray(prior.h.GOAL[:2], dtype=float)
    for index, row in enumerate(rows):
        row["physical_states_21_eval_only"] = states[index].tolist()
        row["goal_position_eval_only"] = goal.tolist()
        row["distances_21"] = {str(t): float(np.linalg.norm(states[index, t, :2] - goal))
                               for t in (0, 12, 17)}
        row["endpoint_22_eval_only"] = endpoints22[index].tolist()
        row["endpoint_21_vs_22_position_max_abs"] = float(
            np.max(np.abs(states[index, -1, :2] - endpoints22[index, :2])))
    receipt = {"schema": "pc_shadow_spectral_physical_replay.v1", "rows": rows,
               "audit_sha256": sha(AUDIT), "journal_sha256": sha(RUN / "events.jsonl"),
               "protocol_sha256": sha(PROTOCOL), "worker_sha256": sha(WORKER),
               "runner_sha256": sha(Path(__file__)), "independent_confirmation": False}
    with OUTPUT.open("x") as out:
        json.dump(receipt, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
