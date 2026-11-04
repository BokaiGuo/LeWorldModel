"""Replay sealed PointMaze plans in legacy MuJoCo and save all RGB steps."""
from pathlib import Path
import sys

import mujoco_py
import numpy as np


def main(request, xml, response):
    with np.load(request, allow_pickle=False) as data:
        starts, sequences = data["sources"], data["sequences"]
    if (starts.ndim != 2 or starts.shape[1] != 4 or sequences.shape != (len(starts), 10, 2)
            or not np.isfinite(starts).all() or not np.isfinite(sequences).all()
            or np.any(np.abs(sequences) > 1)):
        raise ValueError("invalid sealed physical inputs")
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    camera = sim.render_contexts[0].cam
    camera.azimuth, camera.elevation = 90, -90
    camera.lookat[0], camera.lookat[1] = 3., 3.
    target = model.site_name2id("target_site")
    histories = []
    for state, actions in zip(starts, sequences):
        sim.reset()
        sim.data.qpos[:], sim.data.qvel[:] = state[:2], state[2:]
        sim.forward()
        sim.data.site_xpos[target] = [11, 11, 0]
        frames = [np.ascontiguousarray(sim.render(224, 224), dtype=np.uint8)]
        for action in actions:
            sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
            sim.forward()
            sim.data.ctrl[:] = action
            for _ in range(5):
                sim.step()
            sim.data.site_xpos[target] = [11, 11, 0]
            frames.append(np.ascontiguousarray(sim.render(224, 224), dtype=np.uint8))
        histories.append(np.stack(frames))
    np.savez_compressed(response, histories=np.stack(histories))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("expected request, xml, response")
    main(*sys.argv[1:])
