"""Legacy MuJoCo subprocess: render fixed physical states, no decision logic."""
from pathlib import Path
import sys

import mujoco_py
import numpy as np


def main(request, xml, response):
    with np.load(request, allow_pickle=False) as payload:
        states = payload["states"]
    if states.shape != (9, 4):
        raise ValueError("expected initial and eight final states")
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    cam = sim.render_contexts[0].cam
    cam.azimuth, cam.elevation = 90, -90
    cam.lookat[0], cam.lookat[1] = 3., 2.5
    frames = []
    for state in states:
        sim.reset()
        sim.data.qpos[:] = state[:2]
        sim.data.qvel[:] = state[2:]
        sim.forward()
        before = np.r_[sim.data.qpos, sim.data.qvel].copy()
        sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
        frames.append(sim.render(224, 224).copy())
        np.testing.assert_array_equal(before, np.r_[sim.data.qpos, sim.data.qvel])
    with open(response, "xb") as f:
        np.savez_compressed(f, frames=np.stack(frames), binding=np.asarray(mujoco_py.__version__),
                            runtime=np.asarray(mujoco_py.get_version()))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("expected request xml response")
    main(*sys.argv[1:])
