"""Isolate renderer temporal state from camera and physical state."""
from pathlib import Path
import sys

import mujoco_py
import numpy as np


def main(request, xml, output):
    with np.load(request, allow_pickle=False) as f:
        states = f["states"]
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    cam = sim.render_contexts[0].cam
    cam.azimuth, cam.elevation = 90, -90
    cam.lookat[0], cam.lookat[1] = 3., 3.
    frames = []
    for state in states:
        sim.reset()
        sim.data.qpos[:] = state[:2]
        sim.data.qvel[:] = state[2:]
        sim.forward()
        sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
        before = np.r_[sim.data.qpos, sim.data.qvel, sim.data.time].copy()
        frames.append([sim.render(224, 224).copy() for _ in range(6)])
        np.testing.assert_array_equal(before, np.r_[sim.data.qpos, sim.data.qvel, sim.data.time])
    with Path(output).open("xb") as f:
        np.savez_compressed(f, frames=frames)


if __name__ == "__main__":
    main(*sys.argv[1:])
