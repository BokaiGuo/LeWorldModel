"""Reconstruct V16 obstacle traces with their actual one-block transition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from experiments.decision_benchmark.generate_pointmaze_obstacle_challenge_extension import _direct_segment_blocked
from experiments.run_pointmaze_v15_reachable_rgb_geometry import _step
from experiments.run_pointmaze_v16_position_anchored_dynamics import _legacy_pointmaze_xml
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import ROOT, sha256_file

DATA = ROOT / "data/point_maze_v16_obstacle_challenge_extension_20261021"
PANEL = ROOT / "results/pointmaze_v16_obstacle_challenge_stratified_replication24_20261021/panel.json"
SOURCE = ROOT / "results/pointmaze_v16_obstacle_challenge_stratified_replication24_20261021_cuda/report.json"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_OBSTACLE_V16_REPLAY_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_pointmaze_obstacle_v16_replay_20261103_DEV01.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError("refusing overwrite")
    manifest = json.loads((DATA / "challenge_manifest.json").read_text())
    panel, source = (json.loads(p.read_text()) for p in (PANEL, SOURCE))
    if sha256_file(DATA / "states.pth") != manifest["states_sha256"]:
        raise RuntimeError("challenge states mismatch")
    roots = panel["selected_roots"]
    if len(roots) != 24 or len(set(roots)) != 24 or source["roots"] != roots:
        raise ValueError("panel roots mismatch")
    states = torch.load(DATA / "states.pth", weights_only=True, map_location="cpu").numpy().astype(np.float64)
    goal = np.asarray(source["selected_goal_qpos"], np.float64)
    if not np.array_equal(goal, np.asarray(manifest["challenge_goal_qpos"])):
        raise ValueError("goal mismatch")
    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    rows = []
    started = time.monotonic()
    print("Replay all 96 exposed obstacle controller traces with their recorded transition.", flush=True)
    for old in source["rows"]:
        root, trace = old["episode_id"], old["trace"]
        if root not in roots or len(trace) != old["blocks_executed"]:
            raise ValueError("invalid trace identity")
        start = states[root, 40].copy()
        state, positions = start.copy(), [start.copy()]
        for t, step in enumerate(trace):
            action = np.asarray(step["selected_action"], np.float64)
            if step["block"] != t or action.shape != (2,) or not np.isfinite(action).all() or np.abs(action).max() > 1:
                raise ValueError("invalid saved action")
            state = _step(model, state, action)
            positions.append(state.copy())
        distances = np.linalg.norm(np.stack(positions)[:, :2] - goal, axis=1)
        if not np.isclose(distances[0], old["initial_distance"], atol=1e-12, rtol=0):
            raise RuntimeError("initial distance mismatch")
        if not np.isclose(distances[-1], old["final_distance"], atol=1e-10, rtol=0):
            raise RuntimeError(f"source endpoint mismatch: {root} {old['method']}")
        rows.append({"root": root, "method": old["method"], "blocks": len(trace),
                     "blocked_at_frame0": _direct_segment_blocked(states[root, 0, :2], goal, clearance=.08),
                     "blocked_at_frame40": _direct_segment_blocked(start[:2], goal, clearance=.08),
                     "initial_distance": float(distances[0]), "after_one_distance": float(distances[1]),
                     "after_five_distance": float(distances[min(5, len(trace))]),
                     "final_distance": float(distances[-1]), "reported_distance": old["final_distance"],
                     "reported_success": old["success"], "positions": np.stack(positions).tolist()})
    if len(rows) != 96 or len({(x["root"], x["method"]) for x in rows}) != 96:
        raise RuntimeError("incomplete panel")
    payload = {"schema": "pc_pointmaze_obstacle_v16_replay.v1", "rows": rows,
               "source_sha256": sha256_file(SOURCE), "panel_sha256": sha256_file(PANEL),
               "manifest_sha256": sha256_file(DATA / "challenge_manifest.json"),
               "states_sha256": sha256_file(DATA / "states.pth"),
               "protocol_sha256": sha256_file(PROTOCOL), "script_sha256": sha256_file(Path(__file__)),
               "transition": "five mujoco integrator steps per saved high-level block",
               "replayed_trajectories": len(rows), "new_model_inference": False,
               "wall_seconds": time.monotonic()-started,
               "scope": "exposed fixed traces, not P1 or independent confirmation"}
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
