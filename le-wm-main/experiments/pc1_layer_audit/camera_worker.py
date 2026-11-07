"""Legacy MuJoCo worker; no model access and no candidate filtering."""
from pathlib import Path
import sys

import numpy as np

CAMERAS = ((3., 3.), (3.5, 3.))
STEPS = (0, 5, 10, 25)
ACTIONS = np.array(((0., 0.), (.1, 0.), (-.1, 0.)), dtype=np.float32)


def main(mode, request, xml, output):
    import mujoco_py
    if mode not in ("frames", "rollouts") or Path(output).exists():
        raise ValueError("invalid mode or output already exists")
    with np.load(request, allow_pickle=False) as data:
        states = data["states"]
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError("finite Nx4 states required")
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    cam = sim.render_contexts[0].cam
    cam.azimuth, cam.elevation = 90, -90
    camera_default = np.r_[cam.azimuth, cam.elevation, cam.distance, cam.lookat].copy()
    frames, traces, contacts, times = [], [], [], []
    actions = ACTIONS if mode == "rollouts" else ((0., 0.),)
    maximum = 25 if mode == "rollouts" else 0
    for index, state in enumerate(states):
        root_images, root_trace, root_contacts, root_times = [], [], [], []
        for action in actions:
            ai, at, ac, tm = [], [], [], []
            for view in CAMERAS:
                sim.reset()
                sim.data.qpos[:] = state[:2]
                sim.data.qvel[:] = state[2:]
                sim.forward()
                np.testing.assert_allclose(np.r_[sim.data.qpos, sim.data.qvel], state, atol=1e-10, rtol=0)
                cam.lookat[0], cam.lookat[1] = view
                images, trace, counts, clock = [], [], [], []
                for step in range(maximum + 1):
                    before = np.r_[sim.data.qpos, sim.data.qvel, sim.data.time].copy()
                    if step in STEPS:
                        sim.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
                        images.append(sim.render(224, 224).copy())
                        np.testing.assert_array_equal(before, np.r_[sim.data.qpos, sim.data.qvel, sim.data.time])
                    trace.append(before[:4])
                    counts.append(int(sim.data.ncon))
                    clock.append(float(sim.data.time))
                    if step < maximum:
                        sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
                        sim.forward()
                        sim.data.ctrl[:] = action
                        for _ in range(5):
                            sim.step()
                ai.append(images)
                at.append(trace)
                ac.append(counts)
                tm.append(clock)
            root_images.append(ai)
            root_trace.append(at)
            root_contacts.append(ac)
            root_times.append(tm)
        frames.append(root_images)
        traces.append(root_trace)
        contacts.append(root_contacts)
        times.append(root_times)
        if (index + 1) % 8 == 0:
            print(f"{mode}: {index + 1}/{len(states)} states", flush=True)
    with Path(output).open("xb") as stream:
        np.savez_compressed(stream, images=frames, traces=traces, contacts=contacts,
                            times=times, camera_default=camera_default,
                            binding=np.asarray(mujoco_py.__version__), runtime=np.asarray(mujoco_py.get_version()))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("expected frames|rollouts request xml output")
    main(*sys.argv[1:])
