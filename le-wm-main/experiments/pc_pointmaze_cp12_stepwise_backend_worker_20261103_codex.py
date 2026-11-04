"""Isolated post-hoc state replay worker; never imported by a planner."""
import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def gl_details():
    lib = ctypes.CDLL("libGL.so.1")
    lib.glGetString.argtypes = [ctypes.c_uint]
    lib.glGetString.restype = ctypes.c_char_p
    def read(value):
        out = lib.glGetString(value)
        if not out:
            raise RuntimeError("missing active OpenGL context")
        return out.decode()
    return {"vendor": read(0x1f00), "renderer": read(0x1f01), "version": read(0x1f02)}


def validate(sources, actions):
    if (sources.shape != (64,4) or actions.shape != (64,17,2)
            or not np.isfinite(sources).all() or not np.isfinite(actions).all()
            or np.any(np.abs(actions)>1)):
        raise ValueError("invalid frozen input")


def native(sources, actions, xml):
    import mujoco_py
    model = mujoco_py.load_model_from_xml(xml)
    states, times, hashes = [], [], []
    gl = None
    for i, (initial, sequence) in enumerate(zip(sources, actions, strict=True)):
        sim = mujoco_py.MjSim(model)
        sim.render(224,224)
        camera = sim.render_contexts[0].cam
        camera.azimuth, camera.elevation = 90, -90
        camera.lookat[0], camera.lookat[1] = 3.0, 3.0
        target = model.site_name2id("target_site")
        sim.reset()
        sim.data.qpos[:], sim.data.qvel[:] = initial[:2], initial[2:]
        sim.forward()
        sim.data.site_xpos[target] = [11,11,0]
        path, clock, images = [], [], []
        def capture():
            image = np.ascontiguousarray(sim.render(224,224),dtype=np.uint8)
            images.append(hashlib.sha256(image.tobytes()).hexdigest())
            path.append(np.r_[sim.data.qpos[:2],sim.data.qvel[:2]].copy())
            clock.append(float(sim.data.time))
        capture()
        if gl is None:
            gl = gl_details()
            if gl["vendor"] != "NVIDIA Corporation" or "NVIDIA" not in gl["renderer"]:
                raise RuntimeError("native renderer is not NVIDIA")
        for action in sequence:
            sim.data.qvel[:] = np.clip(sim.data.qvel,-5,5)
            sim.forward()
            sim.data.ctrl[:] = action
            for _ in range(5):
                sim.step()
            sim.data.site_xpos[target] = [11,11,0]
            capture()
        states.append(path)
        times.append(clock)
        hashes.append(images)
        print(f"native {i+1}/64",flush=True)
    meta = {"engine":"mujoco_py", "binding":mujoco_py.__version__,
            "reported_runtime":mujoco_py.get_version(), "cymj":mujoco_py.cymj.__file__, "gl":gl,
            "timestep":float(model.opt.timestep)}
    return np.asarray(states),np.asarray(times),np.asarray(hashes),meta


def position(sources, actions, xml):
    import mujoco
    sys.path.insert(0,str(Path(__file__).resolve().parent / "spectral_reachability"))
    from audit_sra_pointmaze_replay_backend import replay
    model = mujoco.MjModel.from_xml_string(xml)
    states, times = [], []
    for i, (initial, sequence) in enumerate(zip(sources, actions, strict=True)):
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = initial[:2], initial[2:]
        mujoco.mj_forward(model,data)
        path = [np.r_[data.qpos[:2],data.qvel[:2]].copy()]
        clock = [float(data.time)]
        for action in sequence:
            data.qvel[:] = np.clip(data.qvel,-5,5)
            mujoco.mj_forward(model,data)
            data.ctrl[:] = np.clip(action,-1,1)
            for _ in range(5):
                mujoco.mj_step(model,data)
            path.append(np.r_[data.qpos[:2],data.qvel[:2]].copy())
            clock.append(float(data.time))
        for t in range(18):
            independently = replay(model,initial,sequence[:t])
            if not np.array_equal(independently,path[t]):
                raise RuntimeError(f"original position backend mismatch {i} t={t}")
        states.append(path)
        times.append(clock)
        print(f"position {i+1}/64",flush=True)
    return np.asarray(states),np.asarray(times),np.empty((0,),dtype="U64"),{
        "engine":"mujoco", "binding":mujoco.__version__, "timestep":float(model.opt.timestep)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine",choices=("native","position"),required=True)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--xml",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing overwrite")
    with np.load(args.request,allow_pickle=False) as src:
        sources,actions = src["sources"],src["actions"]
    validate(sources,actions)
    states,times,hashes,meta = (native if args.engine=="native" else position)(
        sources,actions,args.xml.read_text())
    with args.output.open("xb") as f:
        np.savez_compressed(f,states=states,times=times,rgb_sha256=hashes,
                            metadata=np.asarray(json.dumps(meta,sort_keys=True)),
                            request_sha256=np.asarray(hashlib.sha256(args.request.read_bytes()).hexdigest()),
                            xml_sha256=np.asarray(hashlib.sha256(args.xml.read_bytes()).hexdigest()))


if __name__ == "__main__":
    main()
