"""CPU-only independent MuJoCo 2.2 raw-clock and state-reset diagnostic."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def setup(model, state, data=None):
    data = mujoco.MjData(model) if data is None else data
    data.qpos[:] = state[:2]
    data.qvel[:] = state[2:]
    mujoco.mj_forward(model, data)
    return data


def step(model, data, action):
    data.qvel[:] = np.clip(data.qvel, -5.0, 5.0)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = np.clip(action, -1, 1)
    for _ in range(5):
        mujoco.mj_step(model, data)


def state(data):
    return np.r_[data.qpos[:2], data.qvel[:2]].copy()


def run(model, source, actions):
    data = setup(model, source)
    for action in actions:
        step(model, data, action)
    return state(data)


def analyze(model, states, actions, roots, frames, lengths):
    rows = []
    for root in roots:
        for frame in frames:
            for horizon in (1, 2):
                if frame + horizon >= lengths[root]:
                    raise ValueError("frame crosses valid trajectory length")
                source = states[root, frame]
                sequence = actions[root, frame:frame+horizon]
                endpoint = run(model, source, sequence)
                target = states[root, frame+horizon]
                difference = endpoint - target
                rows.append({"root": int(root), "frame": int(frame), "actions": horizon,
                             "position_error": float(np.linalg.norm(difference[:2])),
                             "state_error": float(np.linalg.norm(difference))})
    # Closed-loop source restoration is probed with both recorded and adversarial
    # sequences. Reusing a data object is tested separately from fresh MjData.
    resets = []
    for root in roots:
        source = states[root, frames[0]]
        a, b = actions[root, frames[0]:frames[0]+2]
        for mode, first, second in (
            ("recorded", a, b), ("alternating_extreme", np.array([1., -1.]), np.array([-1., 1.])),
        ):
            direct_data = setup(model, source)
            step(model, direct_data, first)
            middle = state(direct_data)
            step(model, direct_data, second)
            direct = state(direct_data)
            fresh = run(model, middle, [second])
            dirty = setup(model, source)
            for _ in range(3):
                step(model, dirty, [-1., 1.])
            setup(model, middle, dirty)
            step(model, dirty, second)
            reused = state(dirty)
            resets.append({"root": int(root), "mode": mode,
                           "fresh_position_error": float(np.linalg.norm(direct[:2]-fresh[:2])),
                           "fresh_state_error": float(np.linalg.norm(direct-fresh)),
                           "reused_position_error": float(np.linalg.norm(direct[:2]-reused[:2])),
                           "reused_state_error": float(np.linalg.norm(direct-reused))})
    return {"backend": {"engine": "mujoco", "version": mujoco.__version__},
            "archive_rows": rows, "reset_rows": resets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.input, allow_pickle=False)
    model = mujoco.MjModel.from_xml_string(args.xml.read_text())
    report = analyze(model, data["states"], data["actions"], data["roots"],
                     data["frames"], data["lengths"])
    with args.output.open("x") as out:
        json.dump(report, out, allow_nan=False, sort_keys=True)


if __name__ == "__main__":
    main()
