"""Evaluator-only MuJoCo 2.1.2 replay; never imported by the planner."""
import hashlib
import sys
from pathlib import Path

import mujoco_py
import numpy as np


def main(request_path, xml_path, output_path):
    if output_path.exists():
        raise FileExistsError(output_path)
    with np.load(request_path, allow_pickle=False) as request:
        starts, actions = request["starts"], request["actions"]
    if starts.shape != (4, 4) or actions.shape != (4, 17, 2):
        raise ValueError("unexpected roster")
    if not np.isfinite(starts).all() or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
        raise ValueError("invalid action or state")
    xml = xml_path.read_bytes()
    if hashlib.sha256(xml).hexdigest() != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
        raise ValueError("unexpected XML")
    model = mujoco_py.load_model_from_xml(xml.decode())
    states = []
    for start, sequence in zip(starts, actions):
        sim = mujoco_py.MjSim(model)
        sim.reset()
        sim.data.qpos[:], sim.data.qvel[:] = start[:2], start[2:]
        sim.forward()
        row = [np.r_[sim.data.qpos.copy(), sim.data.qvel.copy()]]
        for action in sequence:
            sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
            sim.forward()
            sim.data.ctrl[:] = action
            for _ in range(5):
                sim.step()
            row.append(np.r_[sim.data.qpos.copy(), sim.data.qvel.copy()])
        states.append(row)
    with output_path.open("xb") as out:
        np.savez_compressed(out, states=np.asarray(states), request_sha256=np.asarray(
            hashlib.sha256(request_path.read_bytes()).hexdigest()))


if __name__ == "__main__":
    main(*(Path(x) for x in sys.argv[1:]))
