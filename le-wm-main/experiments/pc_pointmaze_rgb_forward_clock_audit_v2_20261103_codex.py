"""Audit frozen cp12 actions under raw and synchronized RGB observation."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from experiments import pc_pointmaze_cp12_native_position_stepwise_20261103_codex as panel
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY,RUNTIME

ROOT=panel.ROOT
PROTOCOL=ROOT.parent/"docs/PC_POINTMAZE_RGB_FORWARD_CLOCK_PROTOCOL_20261103_CODEX.md"
WORKER=ROOT/"experiments/pc_pointmaze_rgb_forward_clock_worker_v2_20261103_codex.py"
OUT=ROOT/"results/pc_pointmaze_rgb_forward_clock_audit_v2_20261103_CODEX.json"
NPZ=OUT.with_suffix(".npz")
JOURNAL=OUT.with_name(OUT.stem+"_events.jsonl")
EXPECTED={
    PROTOCOL:"eb67e26a7388c11ce95c18e317375dcfbfaf0d9fd59219644485d3d52f27f388",
    WORKER:"28d4fa88ec88ae92c380fb4ecd8f173376ced5b9cb1b2c5cec76ceae896da72a",
    panel.OUTPUT:"699300a1d2a3566e58d448411d5de857ade7ced506f51fd555147c9d85b75477",
}


def analyze(states,geoms,boundary,offset):
    if states.shape!=(2,64,86,5) or geoms.shape!=(2,64,86,3) or boundary.shape!=(2,64,18,3) or offset.shape!=(64,2):
        raise ValueError("invalid state geometry shapes")
    if not all(np.isfinite(x).all() for x in (states,geoms,boundary,offset)):
        raise ValueError("nonfinite state geometry")
    if not np.array_equal(states[0],states[1]):
        raise ValueError("forward changed trajectory")
    ticks=np.arange(1,18)*5
    raw=boundary[0,:,1:,:2]
    current=states[0][:,ticks,:2] + offset[:,None,:]
    previous=states[0][:,ticks-1,:2] + offset[:,None,:]
    synced=geoms[1][:,ticks,:2]
    return {"raw_minus_previous":raw-previous,"raw_minus_current":raw-current,
            "sync_minus_current":synced-current}


def main():
    if any(x.exists() for x in (OUT,NPZ,JOURNAL)):
        raise FileExistsError("refuse overwrite or automatic retry")
    for path,digest in {**panel.EXPECTED,**EXPECTED}.items():
        if panel.prior.sha(path)!=digest:
            raise ValueError(f"source changed {path}")
    source=json.loads(panel.OUTPUT.read_text())
    if panel.prior.sha(panel.ARRAYS)!=source["arrays_sha256"]:
        raise ValueError("state audit arrays changed")
    _,_,names=panel.prior.files()
    for p in (panel.prior.CONTEXT,panel.prior.DECISIONS,panel.prior.EXECUTION,panel.prior.EVALUATION):
        panel.prior.load_sealed(p)
    starts,actions,expected=panel.build_inputs(names)
    xml=official_xml().encode()
    if panel.digest_bytes(xml)!="64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
        raise ValueError("XML changed")
    t0=time.monotonic()
    with JOURNAL.open("x") as journal, tempfile.TemporaryDirectory(prefix="rgb_forward_clock_") as temp:
        temp=Path(temp)
        request,scene,result=temp/"request.npz",temp/"scene.xml",temp/"out.npz"
        np.savez_compressed(request,starts=starts,actions=actions)
        scene.write_bytes(xml)
        env=os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"]=str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"]=":".join((str(RUNTIME/"lib"),str(RUNTIME/"bin"),"/usr/lib/nvidia",env.get("LD_LIBRARY_PATH","")))
        journal.write(json.dumps({"event":"worker_attempt","time_ns":time.time_ns()})+"\n");journal.flush()
        run=subprocess.run([str(LEGACY/"bin/python"),str(WORKER),str(request),str(scene),str(result)],
                           env=env,text=True,capture_output=True,timeout=300)
        journal.write(json.dumps({"event":"worker_returned","returncode":run.returncode,
                                  "stdout_tail":run.stdout[-500:],"stderr_tail":run.stderr[-1500:]})+"\n");journal.flush()
        if run.returncode or not result.exists():
            raise RuntimeError(f"worker failed {run.stderr[-1500:]}")
        with np.load(result,allow_pickle=False) as src:
            if str(src["request_sha256"].item())!=panel.prior.sha(request) or str(src["xml_sha256"].item())!=panel.prior.sha(scene):
                raise ValueError("request provenance mismatch")
            data={k:src[k].copy() for k in ("states","geoms","boundary_pre_forward","post_forward_state","image_hashes",
                                             "pixel_differences","geometry_offsets","metadata")}
        if not np.array_equal(data["image_hashes"][0],expected):
            mismatch=np.argwhere(data["image_hashes"][0]!=expected)
            journal.write(json.dumps({"event":"sealed_frame_mismatch","count":len(mismatch),
                                      "first_indices":mismatch[:20].tolist()})+"\n");journal.flush()
            raise ValueError("legacy frames do not match sealed execution")
        if not np.array_equal(data["post_forward_state"],data["states"][:,:,::5,:]):
            raise ValueError("forward changed boundary qpos/qvel/time")
        comparison=analyze(data["states"],data["geoms"],data["boundary_pre_forward"],data["geometry_offsets"])
        ticks=data["states"][0,:,:,4]
        if not np.allclose(ticks,np.arange(86)[None,:]*.01,atol=1e-12,rtol=0):
            raise ValueError("tick clock mismatch")
        with np.load(panel.ARRAYS,allow_pickle=False) as previous:
            np.testing.assert_array_equal(data["states"][0,:,::5,:4],previous["native_states"])
        data.update(comparison)
        with NPZ.open("xb") as f:
            np.savez_compressed(f,**data)
        journal.write(json.dumps({"event":"completed","time_ns":time.time_ns()})+"\n")
    pixels=data["pixel_differences"]
    raw_res=np.linalg.norm(comparison["raw_minus_previous"],axis=2)
    sync_res=np.linalg.norm(comparison["sync_minus_current"],axis=2)
    stale=np.linalg.norm(comparison["raw_minus_current"],axis=2)
    report={"schema":"pointmaze_rgb_forward_clock_audit.v1",
            "scope":"instrumented deterministic replay; no model, planning or inferred qpos from pixels",
            "raw_saved_exact":int((data["image_hashes"][0]==expected).sum()),"frames":64*18,
            "full_tick_state_exact":bool(np.array_equal(data["states"][0],data["states"][1])),
            "post_forward_boundary_state_exact":True,
            "max_raw_geom_to_previous_qpos_l2":float(raw_res.max()),
            "max_synced_geom_to_current_qpos_l2":float(sync_res.max()),
            "max_raw_geom_to_current_qpos_l2":float(stale.max()),
            "nonzero_raw_lag_boundaries":int(np.count_nonzero(stale>1e-12)),
            "sync_vs_raw_different_frames":int(np.count_nonzero(pixels[:,:,0])),
            "max_different_channels":int(pixels[:,:,0].max()),"max_pixel_value_difference":int(pixels[:,:,1].max()),
            "metadata":json.loads(str(data["metadata"].item())),"arrays_sha256":panel.prior.sha(NPZ),
            "journal_sha256":panel.prior.sha(JOURNAL),"dependencies":{str(p):panel.prior.sha(p) for p in
                list({**panel.EXPECTED,**EXPECTED})+[Path(__file__),panel.ARRAYS]},
            "counts":{"raw_commands":1088,"synchronized_commands":1088,"model_calls":0,"selection_calls":0},
            "elapsed_seconds":time.monotonic()-t0}
    with OUT.open("x") as f:
        json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    print(f"Saved {OUT}, sha256={panel.prior.sha(OUT)}",flush=True)


if __name__=="__main__":
    main()
