"""Legacy MuJoCo subprocess: fixed physical amplitudes, no model or selection access."""
from pathlib import Path
import sys

import mujoco_py
import numpy as np


def run(request, xml, response):
    states = np.load(request, allow_pickle=False)["states"]
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    values = [-.1, -.03, .03, .1, 0.]
    traces = np.empty((len(states), len(values), 6, 4), dtype=np.float64)
    for i, state in enumerate(states):
        for j, x in enumerate(values):
            sim.reset()
            sim.data.qpos[:] = state[:2]
            sim.data.qvel[:] = state[2:]
            sim.forward()
            for step in range(6):
                traces[i, j, step] = np.r_[sim.data.qpos, sim.data.qvel]
                if step < 5:
                    sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
                    sim.forward()
                    sim.data.ctrl[:] = [x, 0]
                    for _ in range(5):
                        sim.step()
    with open(response, "xb") as stream:
        np.savez_compressed(stream, traces=traces)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("expected request xml response")
    run(*sys.argv[1:])
