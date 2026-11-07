"""Render synchronized states in a separate sim without mutating the integrator."""
from pathlib import Path
import sys

import numpy as np

CAMERAS = ((3., 3.), (3.5, 3.))
STEPS = (0, 5, 10, 25)
ACTIONS = np.array(((0., 0.), (.1, 0.), (-.1, 0.)), dtype=np.float32)


def main(mode, request, xml, output):
    import mujoco_py
    if mode not in ("frames", "rollouts") or Path(output).exists():
        raise ValueError("invalid mode or existing output")
    with np.load(request, allow_pickle=False) as data:
        states = data["states"]
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError("finite state matrix required")
    model = mujoco_py.load_model_from_xml(Path(xml).read_text())
    sim, visual = mujoco_py.MjSim(model), mujoco_py.MjSim(model)
    visual.render(224, 224)
    cam = visual.render_contexts[0].cam
    cam.azimuth, cam.elevation = 90, -90
    camera_default = np.r_[cam.azimuth, cam.elevation, cam.distance, cam.lookat].copy()
    from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
    graphics = [str(glGetString(k)) for k in (GL_VENDOR, GL_RENDERER, GL_VERSION)]
    site = model.site_name2id("particle_site")
    base = model.body_pos[model.body_name2id("particle"), :2].copy()
    frames, traces, contacts, times, lags = [], [], [], [], []
    actions = ACTIONS if mode == "rollouts" else ACTIONS[:1]
    maximum = 25 if mode == "rollouts" else 0
    for index, state in enumerate(states):
        ri, rt, rc, rm, rl = [], [], [], [], []
        for action in actions:
            ai, at, ac, am, al = [], [], [], [], []
            for view in CAMERAS:
                sim.reset()
                sim.data.qpos[:] = state[:2]
                sim.data.qvel[:] = state[2:]
                sim.forward()
                np.testing.assert_allclose(np.r_[sim.data.qpos, sim.data.qvel], state, atol=1e-10, rtol=0)
                images, trace, counts, clock, lag = [], [], [], [], []
                for step in range(maximum+1):
                    before = np.r_[sim.data.qpos, sim.data.qvel, sim.data.time].copy()
                    lag.append(np.abs(sim.data.site_xpos[site, :2] - base - sim.data.qpos).max())
                    if step in STEPS:
                        visual.reset()
                        visual.data.qpos[:] = sim.data.qpos
                        visual.data.qvel[:] = sim.data.qvel
                        visual.forward()
                        np.testing.assert_allclose(visual.data.site_xpos[site, :2], base+sim.data.qpos, atol=1e-10, rtol=0)
                        visual.data.site_xpos[model.site_name2id("target_site")] = [11, 11, 0]
                        cam.lookat[0], cam.lookat[1] = view
                        images.append(visual.render(224, 224).copy())
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
                ai.append(images); at.append(trace); ac.append(counts); am.append(clock); al.append(lag)
            ri.append(ai); rt.append(at); rc.append(ac); rm.append(am); rl.append(al)
        frames.append(ri); traces.append(rt); contacts.append(rc); times.append(rm); lags.append(rl)
        if (index+1) % 8 == 0:
            print(f"synchronized {mode}: {index+1}/{len(states)}", flush=True)
    with Path(output).open("xb") as f:
        np.savez_compressed(f, images=frames, traces=traces, contacts=contacts, times=times,
                            geometry_cache_lag=lags, graphics=np.asarray(graphics),
                            camera_default=camera_default, binding=np.asarray(mujoco_py.__version__),
                            runtime=np.asarray(mujoco_py.get_version()))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("expected mode request xml output")
    main(*sys.argv[1:])
