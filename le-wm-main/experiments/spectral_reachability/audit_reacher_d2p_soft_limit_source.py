"""Causal simulator/source audit of hard-range admission of soft-limit roots.

No neural inference, no new future outcomes. Run on the CUDA host for EGL and
an explicit CUDA allocation receipt; never clip roots, relax goal bounds or
skip any source identity. Historical v1 remains available and unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def summarize(rows):
    if len(rows)!=16 or len({r["seed"] for r in rows})!=16:
        raise ValueError("all16 distinct fixed source roots required")
    invalid = [r for r in rows if not r["v1_admitted"]]
    valid = [r for r in rows if r["v1_admitted"]]
    if (len(invalid)!=1 or invalid[0]["seed"]!=202651005
            or any(r["v2_source_qpos_exact"] is not True or r["full_offset_target_inside_nominal_limits"] is not True for r in rows)
            or any(r["v1_v2_context_exact"] is not True for r in valid)
            or invalid[0]["source_limit_violation_radians"][1]<=0):
        raise ValueError("strict source preservation/goal invariance/one reproduced rejection required")
    return {"roots":16,"v1_accepted_and_exactly_unchanged":len(valid),"v1_rejected_v2_preserved_roots":[r["seed"] for r in invalid],
        "neural_predict_samples":0,"physical_future_branches":0,"P1_admission":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-protocol",type=Path,required=True)
    parser.add_argument("--original-reference",type=Path,required=True)
    parser.add_argument("--adapter-artifact",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("new immutable source audit required")
    failed = json.loads(args.failed_protocol.read_text())
    old = json.loads(args.original_reference.read_text())
    if old["status"]!="completed" or not all(v is True for v in old["gates"].values()):
        raise ValueError("completed original comparison required")
    roots = old["sealed_protocol"]["source_roots"]+failed["source_roots"]
    if len(roots)!=16 or failed["goal_contract"]!="joint-feasible-v1": raise ValueError("exact failed protocol and all16 roots required")
    dataset,source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha!=failed["source_sha256_receipt"] or digest(args.adapter_artifact)!=failed["adapter_sha256"]:
        raise ValueError("unchanged official source and normalization receipt required")
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    bindings = {str(p.resolve()):digest(p) for p in (args.failed_protocol,args.original_reference,args.adapter_artifact)}
    implementations = {str(p.resolve()):digest(p) for p in (Path(__file__),Path(goals.__file__),Path(goals.feasible_goal_delta.__code__.co_filename),Path(shared._source_state_context.__code__.co_filename))}
    envs = {version:goals.goal_wrapper(version)(gym.make("swm/ReacherDMControl-v0",render_mode="rgb_array"),target_delta=(.65,-.65))
        for version in ("joint-feasible-v1","joint-feasible-v2-soft-source")}
    rows = []
    try:
        with h5py.File(dataset,"r",swmr=True) as data:
            for root in roots:
                seed = root["seed"]
                source = np.asarray(data["qpos"][root["source_start_row"]+15],np.float64)
                new = shared._source_state_context(envs["joint-feasible-v2-soft-source"],dataset=dataset,seed=seed)
                if any(new[k]!=root[k] for k in ("source_episode","source_start_row")):
                    raise RuntimeError("source identity changed")
                physics = envs["joint-feasible-v2-soft-source"].unwrapped.env.physics
                receipt = goals.capture_goal_admission(envs["joint-feasible-v2-soft-source"],"joint-feasible-v2-soft-source")
                exact = np.array_equal(source,np.asarray(new["execution_snapshot"]["qpos"])) and np.array_equal(source,np.asarray(physics.data.qpos))
                target = np.asarray(receipt["feasible_target_qpos"])
                limited = np.asarray(receipt["limited_joints"],bool)
                ranges = np.asarray(receipt["ranges_radians"])
                legal = bool(np.all(target[limited]>=ranges[limited,0]) and np.all(target[limited]<=ranges[limited,1])
                    and np.array_equal(np.abs(receipt["effective_delta"]),[.65,.65]))
                admitted,error,unchanged = True,None,None
                try:
                    previous = shared._source_state_context(envs["joint-feasible-v1"],dataset=dataset,seed=seed)
                except ValueError as exc:
                    if str(exc)!="bounded source joint must start inside its physical range": raise
                    admitted,error = False,str(exc)
                    if not np.array_equal(source,np.asarray(envs["joint-feasible-v1"].unwrapped.env.physics.data.qpos)):
                        raise RuntimeError("v1 failed for a state different from official source")
                else:
                    keys = ("source_episode","source_start_row","root_rgb_sha256","goal_rgb_sha256","history_rgb_sha256","warmup_actions_sha256","execution_snapshot")
                    unchanged = all(previous[k]==new[k] for k in keys)
                rows.append({**root,"source_qpos":source.tolist(),"source_limit_violation_radians":receipt["source_limit_violation_radians"],
                    "v1_admitted":admitted,"v1_error":error,"v1_v2_context_exact":unchanged,"v2_source_qpos_exact":bool(exact),
                    "full_offset_target_inside_nominal_limits":legal,"v2_goal_admission":receipt,
                    "root_rgb_sha256":new["root_rgb_sha256"],"goal_rgb_sha256":new["goal_rgb_sha256"],
                    "joint_limit_solver_ref":np.asarray(physics.model.jnt_solref).tolist(),
                    "joint_limit_solver_imp":np.asarray(physics.model.jnt_solimp).tolist()})
                print(json.dumps({"seed":seed,"v1_admitted":admitted,"v2_source_exact":bool(exact),"v1_v2_context_exact":unchanged}),flush=True)
    finally:
        for env in envs.values(): env.close()
    result = summarize(rows)
    gates = {"source_and_receipt_unchanged":all(digest(Path(p))==s for p,s in bindings.items()),
        "implementation_unchanged":all(digest(Path(p))==s for p,s in implementations.items()),
        "all16_source_roots_retained":True,"all15_valid_v1_contexts_exact":True,"failed_source_reproduced_and_preserved":True,
        "all_targets_inside_unrelaxed_nominal_limits":True,"no_neural_or_future_outcomes":True}
    write_new(args.output,{"schema":"reacher_d2p_soft_limit_source.v1","status":"completed" if all(gates.values()) else "fail",
        "cuda_host":host,"gates":gates,"bindings":bindings,"implementation_sha256":implementations,"rows":rows,"summary":result})
    print(json.dumps(result),flush=True)


if __name__=="__main__": main()
