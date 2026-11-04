"""Privileged fixed-position velocity intervention, never a planner input."""
import json
import sys
from pathlib import Path

import mujoco_py
import numpy as np


def main(source_path, xml_path, output_path):
    if mujoco_py.__version__ != "2.1.2.14":
        raise RuntimeError("wrong runtime")
    source = json.loads(Path(source_path).read_text())
    model = mujoco_py.load_model_from_xml(Path(xml_path).read_text())
    goal = np.array([1.8, 2.8])
    suffixes = np.asarray(source["common_suffixes"])
    if suffixes.shape != (6, 10, 2) or len(source["rows"]) != 48:
        raise ValueError("unexpected sealed panel")
    rows = []
    for row in source["rows"]:
        state = np.asarray(row["prefix_states"][-1])
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("invalid prefix state")
        conditions = {}
        for condition in ("recorded", "zero_velocity"):
            traces, distances = [], []
            for suffix in suffixes:
                sim = mujoco_py.MjSim(model)
                sim.data.qpos[:] = state[:2]
                sim.data.qvel[:] = state[2:] if condition == "recorded" else 0
                sim.forward()
                trace = [np.r_[sim.data.qpos[:2], sim.data.qvel[:2]].tolist()]
                for action in suffix:
                    sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
                    sim.forward()
                    sim.data.ctrl[:] = action
                    for _ in range(5):
                        sim.step()
                    trace.append(np.r_[sim.data.qpos[:2], sim.data.qvel[:2]].tolist())
                distances.append(float(np.linalg.norm(sim.data.qpos[:2] - goal)))
                traces.append(trace)
            conditions[condition] = {
                "distances": distances,
                "passive_change": distances[0] - row["prefix_distance"],
                "active_changes": [d - distances[0] for d in distances],
                "traces": traces,
            }
        original = np.asarray(row["common_distances"])
        error = np.asarray(conditions["recorded"]["distances"]) - original
        rows.append({"root_index": row["root_index"], "start": row["start"],
                     "prefix_name": row["prefix_name"], "prefix_state": state.tolist(),
                     "prefix_distance": row["prefix_distance"],
                     "non_improving": row["non_improving"], "conditions": conditions,
                     "reconstruction_difference": error.tolist()})
    with Path(output_path).open("x") as handle:
        json.dump({"binding": mujoco_py.__version__, "runtime": mujoco_py.get_version(),
                   "rows": rows}, handle, allow_nan=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
