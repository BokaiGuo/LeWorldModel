"""Exact all-root goal render and selected/control pixel/latent/task replay.

Frozen encoder on host CUDA only. No predictor inference, fitted readout,
objective retuning or outcome-selected roots/tasks/actions. Pixel costs are
evaluator-only diagnostics, never a new success definition or planner input.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability import reacher_d2p_fixed_warm_render as render
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import analyze_reacher_d2p_root_metric_confirmation as reference_analysis
from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest, delta, TASKS
from experiments.spectral_reachability.audit_reacher_d2p_capability_rejections import indexed
from experiments.spectral_reachability.audit_reacher_d2p_root_metric_score import OFFICIAL
from experiments.spectral_reachability.run_reacher_d2p_masked_program_pilot import check_raw


def pixel_cost(image, target):
    a,b = np.asarray(image), np.asarray(target)
    if a.shape != (224,224,3) or b.shape != a.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError("unaltered complete224 uint8 RGB image pair required")
    return float(np.mean(np.square((a.astype(np.float64)-b.astype(np.float64))/255.)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--adapter-artifact",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    paths = {k:args.output.with_suffix(s) for k,s in (("protocol",".protocol.json"),("implementation",".implementation"),("images",".images"))}
    if any(p.exists() for p in (args.output,*paths.values())): raise FileExistsError("new immutable pixel/goal audit paths required")
    reference_analysis.verify_files(args.reference)
    source = json.loads(args.reference.read_text())
    prediction_path = args.reference.with_suffix(".predictions.json")
    predictions = json.loads(prediction_path.read_text())
    plans, actual = indexed(predictions["rows"]), indexed(source["rows"])
    prior = source["sealed_protocol"]
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if (digest(args.policy/"weights.pt") != OFFICIAL or prior["weights_sha256"] != OFFICIAL
            or digest(args.adapter_artifact) != prior["adapter_sha256"] or source_sha != prior["source_sha256_receipt"]):
        raise ValueError("unchanged official model/source/action statistics required")
    seeds = prior["root_seeds"]
    if len(seeds)!=8 or set(seeds)!=set(plans) or set(plans)!=set(actual): raise ValueError("complete8 original roots required")
    roles = {str(s):{t:{"metric":plans[s]["tasks"][t]["scores"]["decision"]["native_index"],
        "greedy":plans[s]["tasks"][t]["scores"]["decision"]["greedy_index"]} for t in TASKS} for s in seeds}
    modules = {"auditor":Path(__file__), **{k:Path(m.__file__) for k,m in (("shared",shared),("execution",execution),
        ("goals",goals),("renderer",render),("p0",p0),("reference_analysis",reference_analysis))},
        "source_context":Path(shared._source_state_context.__code__.co_filename),
        "loader":Path(shared.load_compatible_pretrained.__code__.co_filename)}
    # Bind the actual transaction implementation, not just its feasible wrapper.
    modules["goal_transaction"] = Path(goals.ReacherFeasibleGoalImageWrapper.__mro__[1]._render_target_and_restore.__code__.co_filename)
    modules["goal_wrapper"] = Path(goals.feasible_goal_delta.__code__.co_filename)
    bindings = {str(p.resolve()):digest(p) for p in (args.reference,prediction_path)}
    protocol = {"seeds":seeds,"roles":roles,"goal_contract":prior["goal_contract"],"weights_sha256":OFFICIAL,
        "source_sha256_receipt":source_sha,"adapter_sha256":prior["adapter_sha256"],"input_bindings":bindings,
        "render_policy":render.POLICY,"tasks":list(TASKS),"expected_goal_identity_checks":8,
        "expected_unique_action_replays":sum(len(set(r.values())) for ts in roles.values() for r in ts.values()),
        "pixel_cost":"full224 uint8 RGB MSE after division by255; no masks/crops/fitted features",
        "all_roles_model_sealed":True,"P1_admission":False,
        "implementation_sha256":{k:digest(p) for k,p in modules.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with paths["protocol"].open("x") as f: json.dump(protocol,f,indent=2,allow_nan=False)
    paths["implementation"].mkdir()
    paths["images"].mkdir()
    for k,p in modules.items(): shutil.copyfile(p,paths["implementation"]/(k+".py"))
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean,std = shared._load_action_stats(args.adapter_artifact,"swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean,std))
    model = shared.load_compatible_pretrained(args.policy,device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameters,buffers = shared._parameter_hash(model),{k:v.clone() for k,v in model.named_buffers()}
    predict = model.predict
    def forbidden_predict(*a,**kw): raise RuntimeError("this audit forbids predictor calls; encoder CUDA only")
    model.predict = forbidden_predict
    def prepare(c,goal=None):
        return p0._prepare(processor,c["frames"],c["root_frame"],c["goal"] if goal is None else goal,
            np.asarray(c["history_action_blocks"],np.float32),torch.device("cuda"))
    rows,receipts,replays = [],[],0
    try:
        for seed in seeds:
            old = plans[seed]
            env = goals.goal_wrapper(protocol["goal_contract"])(gym.make("swm/ReacherDMControl-v0",render_mode="rgb_array"),target_delta=(.65,-.65))
            renderer = render.FixedWarmRenderer(env)
            try:
                ctx = shared._source_state_context(env,dataset=dataset,seed=seed)
                ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
                if (any(ctx[k]!=old[k] for k in ("source_episode","source_start_row","root_rgb_sha256","goal_rgb_sha256"))
                        or ctx["execution_snapshot"]!=old["initial_snapshot"]): raise RuntimeError("original full root/goal state changed")
                _,goal,_ = prepare(ctx)
                target = p0._goal_latent(model,goal)
                if target.tolist()!=old["goal_vector"]: raise RuntimeError("original goal encoding changed")
                physics = env.unwrapped.env.physics
                target_qpos = old["goal_admission"]["feasible_target_qpos"]
                physics.data.qpos[:] = target_qpos
                physics.data.qvel[:] = 0.
                physics.forward()
                direct = np.asarray(env.unwrapped.render(width=224,height=224)).copy()
                exact = np.array_equal(direct,ctx["goal"])
                if not exact: raise RuntimeError("goal image does not reproduce direct admitted physical target rendering")
                _,direct_goal,_ = prepare(ctx,direct)
                if not torch.equal(p0._goal_latent(model,direct_goal),target): raise RuntimeError("direct target RGB/latent identity differs")
                goal_path = paths["images"]/f"root_{seed}_goal.png"
                Image.fromarray(direct).save(goal_path)
                tasks = {}
                for task in TASKS:
                    mapping = roles[str(seed)][task]
                    endpoints = {}
                    for index in sorted(set(mapping.values())):
                        raw = np.asarray(old["tasks"][task]["bank"][index]["raw_actions"],np.float32)
                        if not check_raw(raw,task): raise RuntimeError("full masked literal raw action changed")
                        frame,goal = execution.restore_execution_snapshot(env,seed=seed,snapshot=ctx["execution_snapshot"],root_identity="sealed_observation_exact_state")
                        p0._verify_execution_identity(frame,goal,ctx["root_frame"],ctx["goal"])
                        current = ctx
                        for j in range(3):
                            current = execution.advance_executed_context(env,context=current,raw_action_block=raw[j*5:(j+1)*5],action_block=5,verified_root_frame=current["root_frame"])
                            pix,_,hist = prepare(current)
                            with torch.no_grad(): value = p0._latent_context(model,pix,hist)[0,-1]
                            qpos = np.asarray(physics.data.qpos,np.float64)[:2].tolist()
                            packet = {"packet":j+1,"observed_vector":value.tolist(),"qpos":qpos,
                                "root_rgb_sha256":current["root_rgb_sha256"],"action_sha256":current["advanced_action_sha256"]}
                            if packet!=actual[seed]["tasks"][task]["bank_"+str(index)]["packets"][j]:
                                raise RuntimeError("same-action complete physical/encoder packet replay changed")
                        image_path = paths["images"]/f"root_{seed}_{task}_bank_{index}.png"
                        Image.fromarray(current["root_frame"]).save(image_path)
                        endpoints[str(index)] = {"pixel_mse":pixel_cost(current["root_frame"],direct),
                            "latent_mse":float(np.mean((np.asarray(value.tolist())-np.asarray(old["goal_vector"]))**2)),
                            "joint_distance":float(np.linalg.norm(delta(target_qpos,qpos,old["goal_admission"]["limited_joints"]))),
                            "qpos":qpos,"image_sha256":digest(image_path),"image_path":str(image_path.resolve())}
                        replays += 1
                    methods = {name:endpoints[str(i)] for name,i in mapping.items()}
                    tasks[task] = {"roles":mapping,"methods":methods,
                        "metric_minus_greedy":{key:methods["metric"][key]-methods["greedy"][key] for key in ("pixel_mse","latent_mse","joint_distance")}}
                rows.append({"seed":seed,"direct_goal_rgb_exact":exact,"goal_image_sha256":digest(goal_path),"tasks":tasks})
                receipts.append({"seed":seed,"stats":dict(renderer.stats)})
                print(json.dumps({"seed":seed,"goal_direct_render_exact":exact,"unique_replays":replays}),flush=True)
            finally:
                renderer.restore()
                env.close()
    finally: model.predict = predict
    gates = {"cuda_encoder_only":True,"all8_goal_identity_checks":len(rows)==8 and all(r["direct_goal_rgb_exact"] for r in rows),
        "all_action_replays":replays==protocol["expected_unique_action_replays"],"full_root_order":[r["seed"] for r in rows]==seeds,
        "parameters_unchanged":shared._parameter_hash(model)==parameters,"buffers_unchanged":all(torch.equal(v,buffers[k]) for k,v in model.named_buffers()),
        "all_modules_eval":not any(m.training for m in model.modules()),"references_unchanged":all(digest(Path(p))==sha for p,sha in bindings.items()),
        "weights_unchanged":digest(args.policy/"weights.pt")==OFFICIAL,"render_receipts_verified":render.valid_receipts(receipts,seeds),
        "implementation_unchanged":all(digest(p)==protocol["implementation_sha256"][k] for k,p in modules.items())}
    result = {"schema":"reacher_d2p_goal_metric_pixel_audit.v1","status":"completed" if all(gates.values()) else "fail",
        "sealed_protocol":protocol,"protocol_sha256":digest(paths["protocol"]),"cuda_host":host,"gates":gates,
        "rows":rows,"render_receipts":receipts,"unique_action_replays":replays,"neural_predict_samples":0,"P1_admission":False,
        "scope":"Known-root evaluator-only full goal/pixel/latent/task replay, no new planner or independent efficacy."}
    with args.output.open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({"gates":gates,"unique_replays":replays}),flush=True)
    if not all(gates.values()): raise RuntimeError("goal/pixel identity integrity failed")


if __name__ == "__main__":
    main()
