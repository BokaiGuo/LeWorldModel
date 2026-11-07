"""Legacy physical re-render worker; no access to planning scores."""
from pathlib import Path
import sys

import mujoco_py
import numpy as np


def run(request, xml, response):
    states = np.load(request, allow_pickle=False)["states"]
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    cam = sim.render_contexts[0].cam
    cam.azimuth, cam.elevation = 90, -90
    actions = [-.1, -.03, .03, .1]
    cameras = ((3., 2.5), (3.5, 2.5))
    traces = np.empty((len(states), 4, 2, 6, 4), dtype=np.float64)
    frames = np.empty((len(states), 4, 2, 224, 224, 3), dtype=np.uint8)
    for i, state in enumerate(states):
        for j, x in enumerate(actions):
            for k, lookat in enumerate(cameras):
                sim.reset()
                sim.data.qpos[:] = state[:2]
                sim.data.qvel[:] = state[2:]
                sim.forward()
                cam.lookat[0], cam.lookat[1] = lookat
                for step in range(6):
                    traces[i, j, k, step] = np.r_[sim.data.qpos, sim.data.qvel]
                    sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
                    frame = sim.render(224, 224).copy()
                    if step == 5:
                        frames[i, j, k] = frame
                    else:
                        sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
                        sim.forward()
                        sim.data.ctrl[:] = [x, 0]
                        for _ in range(5):
                            sim.step()
    with open(response, "xb") as stream:
        np.savez_compressed(stream, traces=traces, frames=frames)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("expected request xml response")
    run(*sys.argv[1:])
