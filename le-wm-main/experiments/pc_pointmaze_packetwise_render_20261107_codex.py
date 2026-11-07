"""Render a fixed state bank with the legacy MuJoCo runtime."""
from pathlib import Path
import sys

import mujoco_py
import numpy as np


def main(request, xml, output):
    with np.load(request, allow_pickle=False) as f:
        states = f["states"]
    if states.shape != (8, 6, 4) or not np.isfinite(states).all():
        raise ValueError("expected eight six-step state traces")
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    cam = sim.render_contexts[0].cam
    cam.azimuth, cam.elevation = 90, -90
    cam.lookat[0], cam.lookat[1] = 3., 2.5
    frames = []
    for chain in states:
        images = []
        for state in chain:
            sim.reset()
            sim.data.qpos[:] = state[:2]
            sim.data.qvel[:] = state[2:]
            sim.forward()
            before = np.r_[sim.data.qpos, sim.data.qvel].copy()
            sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
            images.append(sim.render(224, 224).copy())
            np.testing.assert_array_equal(before, np.r_[sim.data.qpos, sim.data.qvel])
        frames.append(images)
    with open(output, "xb") as f:
        np.savez_compressed(f, frames=np.asarray(frames))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("expected request xml output")
    main(*sys.argv[1:])
