"""Observation-only shadow simulator for the frozen PointMaze model."""
import ctypes
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

EXTRA = ("ctrl", "mocap_pos", "mocap_quat", "qfrc_applied",
         "xfrc_applied", "qacc_warmstart", "userdata")


def snapshot(sim):
    state = sim.get_state()
    if state.udd_state not in (None, {}):
        raise ValueError("unsupported nonempty udd_state")
    return {"time": np.asarray(state.time), "qpos": state.qpos.copy(),
            "qvel": state.qvel.copy(),
            "act": np.asarray([]) if state.act is None else state.act.copy(),
            **{key: (np.asarray([]) if getattr(sim.data, key) is None
                     else getattr(sim.data, key).copy()) for key in EXTRA}}


def same(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def transfer(source, destination):
    state = source.get_state()
    before = snapshot(source)
    destination.set_state(state)
    for field in EXTRA:
        src, dst = getattr(source.data, field), getattr(destination.data, field)
        if (src is None) != (dst is None):
            raise ValueError(f"state field shape mismatch: {field}")
        if src is not None:
            dst[:] = src
    if not same(before, snapshot(destination)):
        raise ValueError("shadow state copy mismatch")


def camera(sim):
    sim.render(224, 224)
    cam = sim.render_contexts[0].cam
    cam.azimuth, cam.elevation = 90, -90
    cam.lookat[0], cam.lookat[1] = 3., 3.


def run(request, xml, output):
    import mujoco_py
    if output.exists():
        raise FileExistsError(output)
    with np.load(request, allow_pickle=False) as inp:
        starts, actions = inp["starts"], inp["actions"]
    if (starts.shape != (64, 4) or actions.shape != (64, 17, 2)
            or not np.isfinite(starts).all() or not np.isfinite(actions).all()
            or np.any(np.abs(actions) > 1)):
        raise ValueError("invalid frozen panel")
    models = [mujoco_py.load_model_from_xml(xml.read_text()) for _ in range(3)]
    model = models[0]
    dims = {k: int(getattr(model, k)) for k in
            ("nq", "nv", "na", "nu", "nmocap", "nuserdata", "neq", "nbody", "ngeom", "nsite", "ntendon")}
    if dims != {"nq": 2, "nv": 2, "na": 0, "nu": 2, "nmocap": 0,
                "nuserdata": 0, "neq": 0, "nbody": 2, "ngeom": 20, "nsite": 2, "ntendon": 0}:
        raise ValueError(f"unsupported model dimensions: {dims}")
    particle = [i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body_name2id("particle")]
    if len(particle) != 1:
        raise ValueError("particle geom not unique")
    target = model.site_name2id("target_site")
    states = np.empty((64, 86, 5))
    raw_hashes = np.empty((64, 18), dtype="U64")
    shadow_hashes = np.empty_like(raw_hashes)
    residuals = np.empty((64, 18, 3))  # all geom position, matrix, particle vs qpos
    invariants = np.zeros((64, 18, 3), dtype=bool)  # copy, main unchanged, shadow state after forward
    gl = None
    for k, (start, sequence) in enumerate(zip(starts, actions)):
        main, shadow, reference = [mujoco_py.MjSim(m) for m in models]
        camera(main)
        camera(shadow)
        main.reset()
        main.data.qpos[:], main.data.qvel[:] = start[:2], start[2:]
        main.forward()
        main.data.site_xpos[target] = [11, 11, 0]
        offset = main.data.geom_xpos[particle[0], :2].copy() - main.data.qpos[:2]

        def state(t):
            states[k, t] = np.r_[main.data.qpos[:2], main.data.qvel[:2], main.data.time]

        def observe(t):
            before = snapshot(main)
            raw = np.ascontiguousarray(main.render(224, 224), dtype=np.uint8)
            raw_hashes[k, t] = hashlib.sha256(raw.tobytes()).hexdigest()
            transfer(main, shadow)
            transfer(main, reference)
            invariants[k, t, 0] = same(snapshot(main), snapshot(shadow)) and same(snapshot(main), snapshot(reference))
            shadow.forward()
            reference.forward()
            shadow.data.site_xpos[target] = [11, 11, 0]
            reference.data.site_xpos[target] = [11, 11, 0]
            after_forward = snapshot(shadow)
            invariants[k, t, 2] = all(np.array_equal(before[x], after_forward[x])
                                      for x in ("time", "qpos", "qvel", "act"))
            residuals[k, t] = [
                np.max(np.abs(shadow.data.geom_xpos-reference.data.geom_xpos)),
                np.max(np.abs(shadow.data.geom_xmat-reference.data.geom_xmat)),
                np.linalg.norm(shadow.data.geom_xpos[particle[0], :2]-main.data.qpos[:2]-offset),
            ]
            image = np.ascontiguousarray(shadow.render(224, 224), dtype=np.uint8)
            shadow_hashes[k, t] = hashlib.sha256(image.tobytes()).hexdigest()
            invariants[k, t, 1] = same(before, snapshot(main))

        state(0)
        observe(0)
        if gl is None:
            lib = ctypes.CDLL("libGL.so.1")
            lib.glGetString.argtypes, lib.glGetString.restype = [ctypes.c_uint], ctypes.c_char_p
            gl = {key: lib.glGetString(val).decode() for key, val in
                  (("vendor", 0x1f00), ("renderer", 0x1f01), ("version", 0x1f02))}
            if gl["vendor"] != "NVIDIA Corporation":
                raise RuntimeError("NVIDIA host renderer required")
        for t, action in enumerate(sequence):
            main.data.qvel[:] = np.clip(main.data.qvel, -5, 5)
            main.forward()  # Original executor's pre-command call, not a new render-time call.
            main.data.ctrl[:] = action
            for j in range(5):
                main.step()
                state(5*t+j+1)
            main.data.site_xpos[target] = [11, 11, 0]
            observe(t+1)
        print(f"trajectory {k+1}/64", flush=True)
    with output.open("xb") as f:
        np.savez_compressed(f, states=states, raw_hashes=raw_hashes, shadow_hashes=shadow_hashes,
                            residuals=residuals, invariants=invariants,
                            metadata=np.asarray(json.dumps({"dimensions": dims, "gl": gl,
                                "mujoco_py": mujoco_py.__version__, "timestep": model.opt.timestep}, sort_keys=True)),
                            request_sha256=np.asarray(hashlib.sha256(request.read_bytes()).hexdigest()),
                            xml_sha256=np.asarray(hashlib.sha256(xml.read_bytes()).hexdigest()))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("request xml output")
    run(*(Path(x) for x in sys.argv[1:]))
