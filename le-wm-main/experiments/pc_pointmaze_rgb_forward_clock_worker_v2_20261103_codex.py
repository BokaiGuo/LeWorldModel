"""Isolated RGB-clock worker with serial independent rendering contexts."""
import ctypes
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def run(request, xml, output):
    import mujoco_py
    if output.exists():
        raise FileExistsError(output)
    with np.load(request,allow_pickle=False) as data:
        starts,actions=data["starts"],data["actions"]
    if starts.shape!=(64,4) or actions.shape!=(64,17,2) or not np.isfinite(starts).all() or not np.isfinite(actions).all() or np.any(np.abs(actions)>1):
        raise ValueError("invalid panel")
    model=mujoco_py.load_model_from_xml(xml.read_text())
    geom=[i for i in range(model.ngeom) if model.geom_bodyid[i]==model.body_name2id("particle")]
    if len(geom)!=1:
        raise ValueError("particle geom not unique")
    geom,target=geom[0],model.site_name2id("target_site")
    states=np.empty((2,64,86,5),np.float64)
    geoms=np.empty((2,64,86,3),np.float64)
    boundary=np.empty((2,64,18,3),np.float64)
    post_forward_state=np.empty((2,64,18,5),np.float64)
    hashes=np.empty((2,64,18),dtype="U64")
    pixels=np.zeros((64,18,3),np.float64)
    offsets=np.empty((64,2),np.float64)
    gl=None
    for k,(start,sequence) in enumerate(zip(starts,actions)):
        raw_images=[]
        for arm in range(2):
            sim=mujoco_py.MjSim(model)
            sim.render(224,224)
            cam=sim.render_contexts[0].cam
            cam.azimuth,cam.elevation=90,-90
            cam.lookat[0],cam.lookat[1]=3.,3.
            sim.reset()
            sim.data.qpos[:],sim.data.qvel[:]=start[:2],start[2:]
            sim.forward()
            sim.data.site_xpos[target]=[11,11,0]
            def record(tick):
                d=sim.data
                states[arm,k,tick]=np.r_[d.qpos[:2],d.qvel[:2],d.time]
                geoms[arm,k,tick]=d.geom_xpos[geom].copy()
            def capture(t):
                rgb=np.ascontiguousarray(sim.render(224,224),dtype=np.uint8)
                hashes[arm,k,t]=hashlib.sha256(rgb.tobytes()).hexdigest()
                if arm==0:
                    raw_images.append(rgb.copy())
                else:
                    diff=np.abs(raw_images[t].astype(np.int16)-rgb.astype(np.int16))
                    pixels[k,t]=[np.count_nonzero(diff),diff.max(),diff.mean()]
            record(0)
            boundary[arm,k,0]=sim.data.geom_xpos[geom].copy()
            if arm==0:
                offsets[k]=geoms[0,k,0,:2]-states[0,k,0,:2]
            capture(0)
            post_forward_state[arm,k,0]=np.r_[sim.data.qpos[:2],sim.data.qvel[:2],sim.data.time]
            if gl is None:
                lib=ctypes.CDLL("libGL.so.1")
                lib.glGetString.argtypes=[ctypes.c_uint]
                lib.glGetString.restype=ctypes.c_char_p
                gl={key:lib.glGetString(value).decode() for key,value in
                    (("vendor",0x1f00),("renderer",0x1f01),("version",0x1f02))}
                if gl["vendor"]!="NVIDIA Corporation":
                    raise RuntimeError("host NVIDIA renderer required")
            for t,action in enumerate(sequence):
                sim.data.qvel[:]=np.clip(sim.data.qvel,-5,5)
                sim.forward()
                sim.data.ctrl[:]=action
                for j in range(5):
                    sim.step()
                    record(t*5+j+1)
                boundary[arm,k,t+1]=sim.data.geom_xpos[geom].copy()
                if arm==1:
                    sim.forward()
                post_forward_state[arm,k,t+1]=np.r_[sim.data.qpos[:2],sim.data.qvel[:2],sim.data.time]
                sim.data.site_xpos[target]=[11,11,0]
                capture(t+1)
                geoms[arm,k,(t+1)*5]=sim.data.geom_xpos[geom].copy()
            del sim
        print(f"trajectory {k+1}/64",flush=True)
    with output.open("xb") as f:
        np.savez_compressed(f,states=states,geoms=geoms,boundary_pre_forward=boundary,post_forward_state=post_forward_state,
                            image_hashes=hashes,pixel_differences=pixels,geometry_offsets=offsets,
                            metadata=np.asarray(json.dumps({"mujoco_py":mujoco_py.__version__,
                                "reported_runtime":mujoco_py.get_version(),"gl":gl,
                                "timestep":model.opt.timestep},sort_keys=True)),
                            request_sha256=np.asarray(hashlib.sha256(request.read_bytes()).hexdigest()),
                            xml_sha256=np.asarray(hashlib.sha256(xml.read_bytes()).hexdigest()))


if __name__=="__main__":
    if len(sys.argv)!=4:
        raise SystemExit("request xml output")
    run(*(Path(x) for x in sys.argv[1:]))
