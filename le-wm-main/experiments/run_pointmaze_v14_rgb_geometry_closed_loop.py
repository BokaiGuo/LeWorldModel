"""Short-horizon RGB-only closed-loop validation for the frozen V14 policy.

Policy input is only rendered RGB geometry history and a fixed in-support
candidate bank.  MuJoCo state is used to advance/evaluate the simulator and
is never passed to the action selector.  V14's visual calibration roots are
fixed before this new-root evaluation begins.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from experiments.decision_benchmark import run_pointmaze_temporal_ood_v12_decoupled_belief as v12
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as v6
from experiments.gpu_preflight import require_cuda_recurrent
from experiments.run_live_pointmaze_planning import DatasetMatchedRenderer, _make_context, _make_gdd_source_contract_context, _pool_visual, fit_visual_calibration, green_center, load_episode
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _legacy_pointmaze_xml


SCHEMA = "pointmaze_v14_rgb_geometry_closed_loop.v1"


def _bootstrap(values: np.ndarray, *, seed: int = 202610371) -> dict[str, float]:
    rng=np.random.default_rng(seed); values=np.asarray(values,dtype=np.float64)
    means=np.asarray([rng.choice(values,len(values),replace=True).mean() for _ in range(4000)])
    return {"mean_gain":float(values.mean()),"lower":float(np.quantile(means,.025)),"upper":float(np.quantile(means,.975))}


def _candidate_bank(context: Path) -> np.ndarray:
    _meta, rows=v6._read_panel(context,v6.CONTEXT_SCHEMA)
    root=next(row["instance_id"] for row in rows if str(row["split"])=="outcome")
    values=sorted([row for row in rows if row["instance_id"]==root],key=lambda row:int(row["candidate_id"]))
    bank=np.asarray([row["action_sequence"] for row in values],dtype=np.float64)
    if bank.shape != (16,20,2): raise ValueError("V14 requires its frozen 16x20 candidate bank")
    return bank


def _history_geometry(renderer: DatasetMatchedRenderer, calibration: Any, states: list[np.ndarray]) -> np.ndarray:
    if len(states)!=9: raise ValueError("V14 policy needs a predecessor plus eight RGB observations")
    values=[]
    for index,state in enumerate(states[1:],start=1):
        image=renderer.render(state); center=green_center(image)
        old=green_center(renderer.render(states[index-1]))
        values.append(np.r_[calibration.position(center),calibration.velocity(np.asarray(center)-np.asarray(old))])
    return np.asarray(values,dtype=np.float32)


def _roll(model: mujoco.MjModel, state: np.ndarray, actions: np.ndarray, *, samples: bool) -> tuple[np.ndarray,list[np.ndarray]]:
    data=mujoco.MjData(model); data.qpos[:]=np.asarray(state,dtype=np.float64)[:2]; data.qvel[:]=np.asarray(state,dtype=np.float64)[2:]; mujoco.mj_forward(model,data)
    trace=[]
    for index, action in enumerate(np.asarray(actions,dtype=np.float64)):
        data.ctrl[:]=np.clip(action,-1,1);mujoco.mj_step(model,data)
        if samples and (index+1)%5==0: trace.append(np.r_[data.qpos[:2],data.qvel[:2]].copy())
    return np.r_[data.qpos[:2],data.qvel[:2]].copy(),trace


def _costs(estimator: Any, ranker: Any, action: Any, fit: dict[str,np.ndarray], geometry: np.ndarray, bank: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    device=next(estimator.parameters()).device
    # JSON/context candidate banks are float64 by default, whereas the frozen
    # V14 modules are trained in float32.  Match the estimator's parameter
    # dtype at the policy boundary so RGB deployment cannot silently depend on
    # NumPy's default precision.
    dtype=next(estimator.parameters()).dtype
    with torch.inference_mode():
        h=torch.as_tensor((geometry[None]-fit["history_mean"])/fit["history_scale"],device=device,dtype=dtype)
        c=torch.as_tensor((bank-fit["candidate_mean"])/fit["candidate_scale"],device=device,dtype=dtype)
        belief=estimator(h).repeat(len(bank),1)
        scale,offset=fit["target_scale"],fit["target_mean"]
        return (ranker(belief,c).cpu().numpy()*scale+offset,action(c).cpu().numpy()*scale+offset)


def _run_arm(method: str, start_history: list[np.ndarray], model: mujoco.MjModel, renderer: DatasetMatchedRenderer, calibration: Any, bank: np.ndarray, estimator: Any, ranker: Any, action: Any, fit: dict[str,np.ndarray], goal: np.ndarray, blocks: int) -> dict[str,Any]:
    history=[value.copy() for value in start_history]; trace=[]; current=history[-1].copy()
    initial_distance=float(np.linalg.norm(current[:2]-goal))
    for block in range(blocks):
        geometry=_history_geometry(renderer,calibration,history)
        belief_cost, action_cost=_costs(estimator,ranker,action,fit,geometry,bank)
        scores=belief_cost if method=="rgb_geometry_v14" else action_cost
        selected=int(np.argmin(scores)); current,observed=_roll(model,current,bank[selected],samples=True)
        history=(history+observed)[-9:]
        trace.append({"block":block,"selected_candidate_id":selected,"predicted_cost":float(scores[selected]),"rgb_geometry_history":geometry.tolist(),"simulator_state_in_policy_input":False})
    final_distance=float(np.linalg.norm(current[:2]-goal))
    return {"method":method,"initial_distance":initial_distance,"final_distance":final_distance,"progress":initial_distance-final_distance,"success":bool(final_distance<=.25),"blocks_executed":blocks,"trace":trace,"policy_contract":{"runtime_input":"RGB geometry history plus fixed candidate actions","simulator_state_in_policy_input":False,"candidate_bank_count":int(len(bank)),"replanning":"20-step pulse, reobserve, replan"}}


def run(dataset_root:Path, frozen:Path, context:Path, output:Path, *, roots:list[int], calibration_roots:list[int], blocks:int=3, device:str="cuda", goal:np.ndarray|None=None)->dict[str,Any]:
    if output.exists(): raise FileExistsError("refusing to overwrite closed-loop artifact")
    if set(roots)&set(calibration_roots): raise ValueError("closed-loop roots overlap RGB calibration roots")
    if any(root<7160 for root in roots): raise ValueError("V14 closed loop requires new extension roots")
    runtime=require_cuda_recurrent(device); frozen_payload=json.loads(frozen.read_text())
    if frozen_payload.get("observation_chart")!="rgb_geometry" or not frozen_payload.get("authorization",{}).get("permitted"): raise ValueError("requires V14 authorized RGB-geometry frozen policy")
    estimator,ranker,action,fit=v12._load(frozen_payload,frozen,device=device); bank=_candidate_bank(context); target=np.asarray([.8,.8] if goal is None else goal,dtype=np.float64)
    states=torch.load(dataset_root/"states.pth",map_location="cpu",weights_only=True).numpy().astype(np.float64)
    calibration=fit_visual_calibration(dataset_root,calibration_roots,max_frames=1600); images,_=load_episode(dataset_root,calibration_roots[0]); renderer=DatasetMatchedRenderer.from_episode(images,calibration); model=mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    rows=[]
    for root in roots:
        history=[states[root,frame].copy() for frame in range(0,41,5)]
        for method in ("action_only","rgb_geometry_v14"):
            rows.append({"episode_id":int(root),**_run_arm(method,history,model,renderer,calibration,bank,estimator,ranker,action,fit,target,blocks)})
    paired=[]
    for root in roots:
        values={row["method"]:row for row in rows if row["episode_id"]==root}; paired.append(values["action_only"]["final_distance"]-values["rgb_geometry_v14"]["final_distance"])
    result={"schema":SCHEMA,"cuda_preflight":runtime,"roots":roots,"calibration_roots":calibration_roots,"root_disjoint":not bool(set(roots)&set(calibration_roots)),"candidate_bank_source":str(context.resolve()),"methods":["action_only","rgb_geometry_v14"],"blocks":int(blocks),"goal_qpos":target.tolist(),"rows":rows,"paired_rgb_geometry_over_action_only":_bootstrap(np.asarray(paired)),"mean_final_distance":{method:float(np.mean([r["final_distance"] for r in rows if r["method"]==method])) for method in ("action_only","rgb_geometry_v14")},"success_rate":{method:float(np.mean([r["success"] for r in rows if r["method"]==method])) for method in ("action_only","rgb_geometry_v14")},"claim_boundary":"RGB-only renderer-backed short-horizon closed-loop pointmaze evidence; not frozen-latent-only LeWM planning or cross-environment evidence."}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result


def main()->int:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset-root",type=Path,required=True);p.add_argument("--frozen",type=Path,required=True);p.add_argument("--context",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--roots",type=int,nargs="+",required=True);p.add_argument("--calibration-roots",type=int,nargs="+",default=list(range(1800,1840)));p.add_argument("--blocks",type=int,default=3);p.add_argument("--device",choices=("cuda",),default="cuda");args=vars(p.parse_args());result=run(**args);print(json.dumps({"artifact":str(args["output"]),"root_count":len(result["roots"]),"paired_rgb_geometry_over_action_only":result["paired_rgb_geometry_over_action_only"],"mean_final_distance":result["mean_final_distance"],"success_rate":result["success_rate"]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
