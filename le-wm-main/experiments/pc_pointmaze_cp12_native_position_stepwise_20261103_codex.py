"""Sealed post-outcome stepwise renderer/position backend comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from experiments import pc_pointmaze_cp12_rgb_history_planning_20261103_codex as prior
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME

ROOT = prior.ROOT
WORKER = ROOT / "experiments/pc_pointmaze_cp12_stepwise_backend_worker_20261103_codex.py"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CP12_NATIVE_POSITION_STEPWISE_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc_pointmaze_cp12_native_position_stepwise_20261103_CODEX.json"
ARRAYS = OUTPUT.with_suffix(".npz")
JOURNAL = OUTPUT.with_name(OUTPUT.stem + "_events.jsonl")
EXPECTED = {
    PROTOCOL: "8d815b8415c4e1199bede7ebb5ca398bff6db139cee072ddbbfe15f6ddcc5004",
    WORKER: "70abafa44e82c2bfa9d03a0a290b018119a69e9c3076d391b9d3d7fbf8644c00",
    prior.CONTEXT: "00e7daa21352c53f168471f790591265cfc1342192191f033731ad8b7fa0880e",
    prior.DECISIONS: "fedb164fdbc7d1f80421eea09510b9d27bd6aa2f601ab575d54547de28003e18",
    prior.EXECUTION: "b1477f951ea7d65339b7a5648e85000a81a6070ec5c32382a99953bcf7e45fa4",
    prior.EVALUATION: "697b0a41c98a469e5960bdb829ff072c28f1f1752fa8809bcf0191cd3e83a186",
}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def build_inputs(names):
    """Bind executed actions, sealed decisions, source resets, and all RGB."""
    sources, actions, expected_rgb = [], [], []
    with np.load(prior.SOURCE.with_suffix(".npz"),allow_pickle=False) as source, \
         np.load(prior.CONTEXT.with_suffix(".npz"),allow_pickle=False) as context, \
         np.load(prior.DECISIONS.with_suffix(".npz"),allow_pickle=False) as decision, \
         np.load(prior.EXECUTION.with_suffix(".npz"),allow_pickle=False) as execution:
        for i,name in enumerate(names):
            if not np.array_equal(source[name+"_actions"][:12],context[name+"_actions"]):
                raise ValueError("source prefix mismatch")
            for arm in ("one","three"):
                sequence = execution[f"{name}_{arm}_actions"]
                if (sequence.shape != (17,2) or not np.array_equal(sequence[:12],context[name+"_actions"])
                        or not np.array_equal(sequence[12:],decision[f"{name}_{arm}_raw"])):
                    raise ValueError("sealed action mismatch")
                frames = np.concatenate((context[name+"_frames"],execution[f"{name}_{arm}_tail_frames"]))
                if frames.shape != (18,224,224,3) or frames.dtype != np.uint8:
                    raise ValueError("invalid sealed RGB")
                sources.append(source["starts"][i])
                actions.append(sequence)
                expected_rgb.append([digest_bytes(np.ascontiguousarray(frame).tobytes()) for frame in frames])
    sources,actions = np.asarray(sources,dtype=np.float64),np.asarray(actions,dtype=np.float64)
    if sources.shape != (64,4) or actions.shape != (64,17,2) or not np.isfinite(sources).all() or not np.isfinite(actions).all() or np.any(np.abs(actions)>1):
        raise ValueError("invalid request")
    return sources,actions,np.asarray(expected_rgb)


def worker(engine, request, xml, output, journal):
    env = os.environ.copy()
    env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
    env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME/"lib"),str(RUNTIME/"bin"),
                                      "/usr/lib/nvidia",env.get("LD_LIBRARY_PATH","")))
    command = [str(LEGACY/"bin/python"),str(WORKER),"--engine",engine,"--request",str(request),
               "--xml",str(xml),"--output",str(output)]
    journal.write(json.dumps({"event":"worker_attempt","engine":engine,"time_ns":time.time_ns()})+"\n")
    journal.flush()
    started = time.monotonic()
    result = subprocess.run(command,env=env,capture_output=True,text=True,timeout=300)
    journal.write(json.dumps({"event":"worker_returned","engine":engine,"returncode":result.returncode,
                              "elapsed_seconds":time.monotonic()-started,
                              "stdout_tail":result.stdout[-500:],"stderr_tail":result.stderr[-1000:]})+"\n")
    journal.flush()
    if result.returncode or not output.exists():
        raise RuntimeError(f"{engine} worker failed: {result.stderr[-1500:]}")
    with np.load(output,allow_pickle=False) as data:
        if str(data["request_sha256"].item()) != prior.sha(request) or str(data["xml_sha256"].item()) != prior.sha(xml):
            raise ValueError("worker request binding mismatch")
        return {k:data[k].copy() for k in ("states","times","rgb_sha256","metadata")}


def main():
    if any(p.exists() for p in (OUTPUT,ARRAYS,JOURNAL)):
        raise FileExistsError("refusing overwrite or retry of partial audit")
    for path,digest in EXPECTED.items():
        if prior.sha(path) != digest:
            raise ValueError(f"input changed {path}")
    _,_,names = prior.files()
    context,decisions,execution,evaluation = (prior.load_sealed(x) for x in
                                              (prior.CONTEXT,prior.DECISIONS,prior.EXECUTION,prior.EVALUATION))
    if (decisions["context_sha256"] != prior.sha(prior.CONTEXT)
            or execution["decisions_sha256"] != prior.sha(prior.DECISIONS)
            or execution["context_sha256"] != prior.sha(prior.CONTEXT)
            or evaluation["execution_sha256"] != prior.sha(prior.EXECUTION)):
        raise ValueError("stage receipt mismatch")
    sources,actions,expected_rgb = build_inputs(names)
    native_xml, position_xml = official_xml().encode(),_legacy_pointmaze_xml().encode()
    if (digest_bytes(native_xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d"
            or digest_bytes(position_xml) != "a41452e9b6bd6d55e8e1f746e8e29e4eb1356484ee7161550e51aabbc2ca51c8"):
        raise ValueError("XML source changed")
    started = time.monotonic()
    print("Post-outcome 64 sealed actions: native RGB replay versus position backend, all 18 ticks.",flush=True)
    with JOURNAL.open("x") as journal:
        with tempfile.TemporaryDirectory(prefix="pc_cp12_stepwise_") as temp:
            temp = Path(temp)
            request, nx, px = temp/"request.npz",temp/"native.xml",temp/"position.xml"
            np.savez_compressed(request,sources=sources,actions=actions)
            nx.write_bytes(native_xml)
            px.write_bytes(position_xml)
            native = worker("native",request,nx,temp/"native.npz",journal)
            pos = worker("position",request,px,temp/"position.npz",journal)
        nstates,pstates = native["states"],pos["states"]
        ntimes,ptimes = native["times"],pos["times"]
        if (nstates.shape != (64,18,4) or pstates.shape != nstates.shape
                or ntimes.shape != (64,18) or ptimes.shape != ntimes.shape
                or native["rgb_sha256"].shape != (64,18)
                or not all(np.isfinite(x).all() for x in (nstates,pstates,ntimes,ptimes))):
            raise ValueError("worker output shape or finiteness")
        if not np.array_equal(native["rgb_sha256"],expected_rgb):
            raise ValueError("native RGB replay does not match saved execution")
        if not np.allclose(ntimes,ptimes,rtol=0,atol=1e-12) or not np.allclose(ntimes,np.arange(18)[None,:]*.05,rtol=0,atol=1e-12):
            raise ValueError("backend clock mismatch")
        with np.load(prior.EVALUATION.with_suffix(".npz"),allow_pickle=False) as old:
            if not np.array_equal(pstates[:,-1,:],old["endpoints"]):
                raise ValueError("position final does not match old receipt")
        delta = nstates-pstates
        pos_norm = np.linalg.norm(delta[:,:,:2],axis=2)
        vel_norm = np.linalg.norm(delta[:,:,2:],axis=2)
        goal = np.array([1.8,2.8])
        nd = np.linalg.norm(nstates[:,-1,:2]-goal,axis=1)
        pd = np.linalg.norm(pstates[:,-1,:2]-goal,axis=1)
        rows = []
        for i,name in enumerate(names):
            for j,arm in enumerate(("one","three")):
                k=2*i+j
                rows.append({"name":name,"arm":arm,"max_position_l2":float(pos_norm[k].max()),
                             "max_velocity_l2":float(vel_norm[k].max()),
                             "final_position_l2":float(pos_norm[k,-1]),"final_velocity_l2":float(vel_norm[k,-1]),
                             "native_distance":float(nd[k]),"position_distance":float(pd[k]),
                             "native_success":bool(nd[k]<.5),"position_success":bool(pd[k]<.5),
                             "final_commands":17,"native_final_time":float(ntimes[k,-1]),
                             "position_final_time":float(ptimes[k,-1]),
                             "exact_states":int(np.all(delta[k]==0,axis=1).sum())})
        with ARRAYS.open("xb") as f:
            np.savez_compressed(f,native_states=nstates,position_states=pstates,delta=delta,
                                native_times=ntimes,position_times=ptimes,
                                native_rgb_sha256=native["rgb_sha256"],expected_rgb_sha256=expected_rgb,
                                position_l2=pos_norm,velocity_l2=vel_norm)
        journal.write(json.dumps({"event":"completed","time_ns":time.time_ns()})+"\n")
    meta_native,meta_pos = json.loads(str(native["metadata"].item())),json.loads(str(pos["metadata"].item()))
    if meta_pos["binding"] != "2.2.0" or meta_native["binding"] != "2.1.2.14":
        raise ValueError("unexpected simulator versions")
    result = {"schema":"cp12_native_position_stepwise.v1","scope":"post-outcome deterministic instrumented replay, never planner input",
              "rows":rows,"backend_native":meta_native,"backend_position":meta_pos,
              "native_xml_sha256":digest_bytes(native_xml),"position_xml_sha256":digest_bytes(position_xml),
              "max_position_l2":float(pos_norm.max()),"max_velocity_l2":float(vel_norm.max()),
              "max_abs_component":float(np.abs(delta).max()),
              "exact_state_count":int(np.all(delta==0,axis=2).sum()),"state_count":64*18,
              "rgb_exact_count":int((native["rgb_sha256"]==expected_rgb).sum()),"rgb_count":64*18,
              "success_disagreements":sum(x["native_success"] != x["position_success"] for x in rows),
              "max_clock_difference":float(np.abs(ntimes-ptimes).max()),
              "counts":{"native_commands":1088,"position_progressive_commands":1088,
                        "position_independent_prefix_commands":9792,"model_calls":0,"selection_calls":0},
              "arrays_sha256":prior.sha(ARRAYS),"journal_sha256":prior.sha(JOURNAL),
              "dependencies":{str(p):prior.sha(p) for p in list(EXPECTED)+[Path(__file__),
                  prior.SOURCE.with_suffix(".npz"),prior.CONTEXT.with_suffix(".npz"),
                  prior.DECISIONS.with_suffix(".npz"),prior.EXECUTION.with_suffix(".npz"),
                  prior.EVALUATION.with_suffix(".npz"),
                  ROOT/"experiments/spectral_reachability/audit_sra_pointmaze_replay_backend.py",
                  RUNTIME/"lib/libmujoco.so.2.1.2"]},
              "elapsed_seconds":time.monotonic()-started}
    with OUTPUT.open("x") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(f"Saved {OUTPUT}, sha256={prior.sha(OUTPUT)}",flush=True)


if __name__ == "__main__":
    main()
