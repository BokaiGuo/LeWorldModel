"""Privileged fixed-geometry feasibility diagnostic, not an RGB planner."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from experiments.decision_benchmark.generate_pointmaze_obstacle_challenge_extension import (
    _is_free,
    obstacle_intersection_starts,
)

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
XML = ROOT / "results/sra_pointmaze_replay_mujoco220_receipt_contract_20261024_cuda/legacy_pointmaze.xml"
PROTOCOL = PROJECT / "docs/PC_POINTMAZE_GEOMETRY_PERSISTENT_FEASIBILITY_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_geometry_persistent_feasibility_20261103_codex.json"
GOAL = np.array([1.8, 2.8])
CELLS = np.array([-0.2, 0.8, 1.8, 2.8, 3.8])


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bank() -> np.ndarray:
    return np.asarray([[m * np.cos(k * np.pi / 4), m * np.sin(k * np.pi / 4)]
                       for k in range(8) for m in (0.25, 0.5, 0.75, 1.0)])


def waypoint(position: np.ndarray) -> np.ndarray:
    positions = {(i, j): np.array([x, y]) for i, x in enumerate(CELLS)
                 for j, y in enumerate(CELLS) if _is_free(np.array([x, y]), clearance=0.08)}
    source = min(positions, key=lambda p: np.linalg.norm(positions[p] - position))
    destination = min(positions, key=lambda p: np.linalg.norm(positions[p] - GOAL))
    queue = deque([source])
    parents = {source: None}
    while queue:
        node = queue.popleft()
        if node == destination:
            break
        for neighbor in ((node[0] - 1, node[1]), (node[0] + 1, node[1]),
                         (node[0], node[1] - 1), (node[0], node[1] + 1)):
            if neighbor in positions and neighbor not in parents:
                parents[neighbor] = node
                queue.append(neighbor)
    if destination not in parents:
        raise ValueError("fixed lattice has no route")
    route = []
    node = destination
    while node is not None:
        route.append(node)
        node = parents[node]
    route.reverse()
    return positions[route[1]] if len(route) > 1 else GOAL.copy()


def step(model: mujoco.MjModel, data: mujoco.MjData, action: np.ndarray) -> None:
    data.qvel[:] = np.clip(data.qvel, -5.0, 5.0)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = np.clip(action, -1.0, 1.0)
    for _ in range(5):
        mujoco.mj_step(model, data)


def run(model: mujoco.MjModel, start: np.ndarray, policy: str) -> dict:
    data = mujoco.MjData(model)
    data.qpos[:] = start
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    trace = []
    initial = float(np.linalg.norm(start - GOAL))
    for tick in range(24):
        target = GOAL if policy == "direct" else waypoint(data.qpos[:2])
        scores = []
        for action in bank():
            candidate = mujoco.MjData(model)
            candidate.qpos[:] = data.qpos
            candidate.qvel[:] = data.qvel
            mujoco.mj_forward(model, candidate)
            step(model, candidate, action)
            scores.append(float(np.linalg.norm(candidate.qpos[:2] - target)))
        index = int(np.argmin(scores))
        action = bank()[index]
        step(model, data, action)
        distance = float(np.linalg.norm(data.qpos[:2] - GOAL))
        trace.append({"tick": tick + 1, "target": target.tolist(), "action_index": index,
                      "action": action.tolist(), "position": data.qpos[:2].tolist(),
                      "velocity": data.qvel[:2].tolist(), "distance": distance,
                      "contact_count": int(data.ncon), "candidate_scores": scores})
        if distance <= 0.25:
            break
    final = trace[-1]["distance"]
    return {"start": start.tolist(), "initial_distance": initial, "policy": policy,
            "success": final <= 0.25, "stop": "success" if final <= 0.25 else "horizon",
            "final_distance": final, "trace": trace}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if mujoco.__version__ != "3.1.6":
        raise RuntimeError("protocol requires MuJoCo 3.1.6")
    starts = obstacle_intersection_starts()
    if [s.tolist() for s in starts] != [[0.8, 0.8], [0.8, 1.8], [2.8, 0.8], [2.8, 1.8]]:
        raise RuntimeError("geometry-derived roots differ from sealed protocol")
    model = mujoco.MjModel.from_xml_string(XML.read_text())
    rows = []
    for start in starts:
        for policy in ("direct", "waypoint"):
            row = run(model, start, policy)
            rows.append(row)
            print(start.tolist(), policy, row["stop"], row["final_distance"], flush=True)
    report = {"schema": "pc_geometry_persistent_feasibility.v1", "mujoco": mujoco.__version__,
              "xml_sha256": digest(XML), "protocol_sha256": digest(PROTOCOL),
              "script_sha256": digest(Path(__file__)), "goal": GOAL.tolist(),
              "rows": rows, "claim": "privileged development feasibility only"}
    with OUTPUT.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
