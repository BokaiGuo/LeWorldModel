"""Render preset camera grid for fixed initial states; no model decisions."""
from pathlib import Path
import sys

import mujoco_py
import numpy as np


def main(request, xml, output):
    with np.load(request, allow_pickle=False) as data:
        states, cameras = data["states"], data["cameras"]
    if states.shape != (8, 4) or cameras.shape != (25, 2):
        raise ValueError("unexpected fixed panel")
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    cam = sim.render_contexts[0].cam
    cam.azimuth, cam.elevation = 90, -90
    defaults = np.r_[cam.azimuth, cam.elevation, cam.distance, cam.lookat].copy()
    frames = []
    for state in states:
        sim.reset()
        sim.data.qpos[:] = state[:2]
        sim.data.qvel[:] = state[2:]
        sim.forward()
        sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
        before = np.r_[sim.data.qpos, sim.data.qvel].copy()
        images = []
        for x, y in cameras:
            cam.lookat[0], cam.lookat[1] = x, y
            images.append(sim.render(224, 224).copy())
            np.testing.assert_array_equal(before, np.r_[sim.data.qpos, sim.data.qvel])
        frames.append(images)
    with open(output, "xb") as f:
        np.savez_compressed(f, frames=np.asarray(frames), defaults=defaults)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("expected request xml output")
    main(*sys.argv[1:])
