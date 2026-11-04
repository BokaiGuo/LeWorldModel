"""Legacy renderer: identical sealed actions, with an intermediate RGB."""
from pathlib import Path
import sys
import mujoco_py
import numpy as np


def main(request, xml, response):
    with np.load(request, allow_pickle=False) as data:
        sources, sequences = data["sources"], data["sequences"]
    if sources.ndim != 2 or sources.shape[1] != 4 or sequences.shape != (len(sources), 10, 2):
        raise ValueError("unexpected physical batch")
    if not np.isfinite(sources).all() or not np.isfinite(sequences).all() or np.any(np.abs(sequences) > 1):
        raise ValueError("invalid physical inputs")
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    camera = sim.render_contexts[0].cam
    camera.azimuth, camera.elevation = 90, -90
    camera.lookat[0], camera.lookat[1] = 3., 3.
    target = model.site_name2id("target_site")
    mid, end, states = [], [], []
    for i, (state, actions) in enumerate(zip(sources, sequences)):
        sim.reset()
        sim.data.qpos[:], sim.data.qvel[:] = state[:2], state[2:]
        sim.forward()
        sim.data.site_xpos[target] = [11, 11, 0]
        for step, action in enumerate(actions):
            sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
            sim.forward()
            sim.data.ctrl[:] = action
            for _ in range(5):
                sim.step()
            sim.data.site_xpos[target] = [11, 11, 0]
            if step == 4:
                mid.append(np.ascontiguousarray(sim.render(224, 224), dtype=np.uint8))
        end.append(np.ascontiguousarray(sim.render(224, 224), dtype=np.uint8))
        states.append(np.r_[sim.data.qpos[:2], sim.data.qvel[:2]].copy())
        if (i + 1) % 48 == 0:
            print("rendered", i + 1, "of", len(sources), flush=True)
    np.savez_compressed(response, rgb5=np.stack(mid), rgb10=np.stack(end), endpoints=np.stack(states))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("expected request, xml, response")
    main(*sys.argv[1:])
