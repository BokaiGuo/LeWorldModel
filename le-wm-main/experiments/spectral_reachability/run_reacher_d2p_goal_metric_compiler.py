"""Prospective full fixed-Q compiler versus original frozen-model pipeline.

Every root/task/bank is retained and presealed before any physical future.
This diagnostic does not admit P1, claim matched compute or train any module.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import run_reacher_d2p_root_metric_confirmation as baseline
from experiments.spectral_reachability import reacher_d2p_goal_metric_compiler as compiler
from experiments.spectral_reachability import reacher_d2p_goal_metric_runtime as runtime
from experiments.spectral_reachability import reacher_d2p_goal_encoder_metric as metric
from experiments.spectral_reachability import reacher_d2p_proposal_native_selection as native

shared, execution, goals, chart = baseline.shared, baseline.execution, baseline.goals, baseline.chart
p0, render, digest, write_new = baseline.p0, baseline.render, baseline.digest, baseline.write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--adapter-artifact",type=Path,required=True)
    parser.add_argument("--disjoint-reference",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=202655000)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    paths = {k:args.output.with_suffix(s) for k,s in (("protocol",".protocol.json"),
        ("predictions",".predictions.json"),("roots",".roots"),("implementation",".implementation"))}
    if any(p.exists() for p in (args.output,*paths.values())):
        raise FileExistsError("new immutable full compiler output required")
    seeds = list(range(args.seed,args.seed+8))
    excluded, bindings = baseline.bind_exclusions([args.disjoint_reference],seeds)
    prior = json.loads(args.disjoint_reference.read_text())["sealed_protocol"]
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if (len(excluded) < 240 or args.seed < 202655000 or prior["weights_sha256"] != baseline.OFFICIAL
            or digest(args.policy/"weights.pt") != baseline.OFFICIAL
            or digest(args.adapter_artifact) != prior["adapter_sha256"] or source_sha != prior["source_sha256_receipt"]):
        raise ValueError("official frozen model/source/stats,240 excluded episodes and prospective seed block required")
    roots = baseline.planned_source_roots(dataset,seeds,excluded)
    modules = {"runner":Path(__file__),"baseline_runner":Path(baseline.__file__),
        **{k:Path(m.__file__) for k,m in (("compiler",compiler),("goal_runtime",runtime),
            ("metric",metric),("native",native),("shared",shared),("execution",execution),
            ("goals",goals),("chart",chart),("p0",p0),("render",render),
            ("old_compiler",baseline.compiler),("search",baseline.search),
            ("old_scoring",baseline.scoring),("qp",baseline.qp))},
        "model_encode":runtime.declared_encoder_source(args.policy)}
    for key,fn in (("source_context",shared._source_state_context),("loader",shared.load_compatible_pretrained),
            ("feasible_goal",goals.feasible_goal_delta),("source_planner",baseline.planned_source_roots),
            ("exclusions",baseline.bind_exclusions),("physical_writer",baseline.physical_row)):
        modules[key] = Path(fn.__code__.co_filename)
    protocol = {"root_seeds":seeds,"source_roots":roots,"excluded_source_episodes":sorted(excluded),
        "exclusion_bindings":bindings,"weights_sha256":baseline.OFFICIAL,"adapter_sha256":digest(args.adapter_artifact),
        "source_sha256_receipt":source_sha,"checkpoint_config_sha256":digest(args.policy/"config.json"),
        "goal_contract":"joint-feasible-v2-soft-source","target_delta":[.65,-.65],
        "tasks":{k:list(v) for k,v in chart.TASKS.items()},"policy":"full-goal-input-cometric-compiler-v1",
        "metric_policy":metric.POLICY,"ridge":.01,"own_state_selector":native.CONTRACT,
        "render_policy":render.POLICY,"same_metric_in_spectrum_stage_reach_and_search":True,
        "all17_programs_retained_per_method":True,"methods":["original","goal_metric"],
        "all_forecasts_sealed_before_future_execution":True,"physical_branches":912,
        "expected_predict_samples_per_task":{"original":16830,"goal_metric":16734},
        "expected_encoder_forwards":104,"expected_input_vjp_rows":1536,
        "implementation_sha256":{k:digest(p) for k,p in modules.items()},
        "P1_admission":False,"compute_matched":False,"physical_recompilation":False,
        "scope":"Prospective fixed 15-action witness execution; all3 tasks, same frozen official checkpoint, no posttraining, no outcome-picked roots or parameters. Not yet closed-loop replanning or true capability proof."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_new(paths["protocol"],protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for k,p in modules.items(): shutil.copyfile(p,paths["implementation"]/(k+".py"))
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean,std = shared._load_action_stats(args.adapter_artifact,"swm/ReacherDMControl-v0")
    normalizer = shared.ActionNormalizer(mean,std)
    processor = shared._Preprocessor(normalizer)
    mean_t,std_t = (torch.tensor(v,device="cuda") for v in (normalizer.block_mean,normalizer.block_std))
    model = shared.load_compatible_pretrained(args.policy,device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if digest(Path(model.encode.__func__.__code__.co_filename)) != protocol["implementation_sha256"]["model_encode"]:
        raise RuntimeError("actual model encoder source mismatch")
    parameters,buffers = shared._parameter_hash(model),{k:v.clone() for k,v in model.named_buffers()}
    original_predict,samples = model.predict,0
    def counted(emb,act):
        nonlocal samples
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("neural CPU fallback forbidden")
        samples += int(emb.shape[0])
        return original_predict(emb,act)
    model.predict = counted
    env = goals.goal_wrapper(protocol["goal_contract"])(gym.make("swm/ReacherDMControl-v0",render_mode="rgb_array"),target_delta=(.65,-.65))
    renderer = render.FixedWarmRenderer(env)
    def prepare(ctx):
        return p0._prepare(processor,ctx["frames"],ctx["root_frame"],ctx["goal"],
            np.asarray(ctx["history_action_blocks"],np.float32),torch.device("cuda"))
    pending,predictions,rows = [],[],[]
    try:
        for root in roots:
            seed = root["seed"]
            ctx = shared._source_state_context(env,dataset=dataset,seed=seed)
            if any(ctx[k] != root[k] for k in ("source_episode","source_start_row")):
                raise RuntimeError("admitted root source changed")
            ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
            pixels,goal,history = prepare(ctx)
            emb,target = p0._latent_context(model,pixels,history),p0._goal_latent(model,goal)
            operator,receipt = runtime.measure_goal(model,goal,target.tolist(),seed=seed)
            tasks = {}
            for task in chart.TASKS:
                methods = {}
                for method,fn,extra in (("original",baseline.upgraded_bank,{}),("goal_metric",compiler.compile_programs,{"metric":operator})):
                    before = samples
                    methods[method] = fn(model,emb,history,target,task=task,mean=mean_t,std=std_t,seed=seed,**extra)
                    methods[method]["actual_predict_samples"] = samples-before
                    if samples-before != protocol["expected_predict_samples_per_task"][method]:
                        raise RuntimeError("fixed compiler model-call accounting mismatch: "+str((method,samples-before)))
                    print(json.dumps({"seed":seed,"task":task,"method":method,"phase":"predictions",
                        "predict_samples":samples-before}),flush=True)
                tasks[task] = methods
            record = {**root,"goal_vector":target.tolist(),"root_rgb_sha256":ctx["root_rgb_sha256"],
                "goal_rgb_sha256":ctx["goal_rgb_sha256"],"initial_snapshot":ctx["execution_snapshot"],
                "goal_admission":goals.capture_goal_admission(env,protocol["goal_contract"]),"goal_metric":receipt,"tasks":tasks}
            predictions.append(record)
            pending.append((ctx,record))
        write_new(paths["predictions"],{"protocol_sha256":digest(paths["protocol"]),"rows":predictions})
        prediction_sha = digest(paths["predictions"])
        print(json.dumps({"all_predictions_presealed":True,"predictions_sha256":prediction_sha}),flush=True)
        for ctx,record in pending:
            tasks = {}
            for task,methods in record["tasks"].items():
                tasks[task] = {}
                for method,plan in methods.items():
                    measured = {}
                    for name,branch in [("bank_"+str(b["index"]),b) for b in plan["bank"]]+list(plan["controls"].items()):
                        raw = np.asarray(branch["raw_actions"],np.float32)
                        if not baseline.check_raw(raw,task): raise RuntimeError("literal raw mask/bounds changed")
                        frame,goal = execution.restore_execution_snapshot(env,seed=record["seed"],snapshot=ctx["execution_snapshot"],root_identity="sealed_observation_exact_state")
                        p0._verify_execution_identity(frame,goal,ctx["root_frame"],ctx["goal"])
                        current,packets = ctx,[]
                        for j in range(3):
                            current = execution.advance_executed_context(env,context=current,raw_action_block=raw[j*5:(j+1)*5],action_block=5,verified_root_frame=current["root_frame"])
                            pixels,_,history = prepare(current)
                            with torch.no_grad(): observed = p0._latent_context(model,pixels,history)[0,-1]
                            packets.append({"packet":j+1,"observed_vector":observed.tolist(),
                                "qpos":np.asarray(env.unwrapped.env.physics.data.qpos,np.float64)[:2].tolist(),
                                "root_rgb_sha256":current["root_rgb_sha256"],"action_sha256":current["advanced_action_sha256"]})
                        measured[name] = {"raw_actions":raw.tolist(),"packets":packets}
                    tasks[task][method] = measured
                    print(json.dumps({"seed":record["seed"],"task":task,"method":method,"phase":"physical","branches":len(measured)}),flush=True)
            row = baseline.physical_row(record,tasks)
            write_new(paths["roots"]/f"root_{record['seed']}.json",row)
            rows.append(row)
    finally:
        model.predict = original_predict
        renderer.restore()
        env.close()
    stats = renderer.stats
    gates = {"cuda_only":True,"all8_roots":[r["seed"] for r in rows]==seeds,
        "source_disjoint":len({r["source_episode"] for r in rows})==8 and not ({r["source_episode"] for r in rows}&excluded),
        "all912_physical_branches":len(rows)==8 and all(set(r["tasks"])==set(chart.TASKS)
            and all(set(t)=={"original","goal_metric"} and all(len(m)==19 for m in t.values()) for t in r["tasks"].values()) for r in rows),
        "all1536_input_vjps":sum(r["goal_metric"]["input_vjp_rows"] for r in predictions)==1536,
        "all104_encoder_forwards":sum(r["goal_metric"]["encoder_forwards"] for r in predictions)==104,
        "all_full_split_exact":all(b["full_split_exact"] for r in predictions for t in r["tasks"].values() for m in t.values() for b in m["bank"]),
        "parameters_unchanged":shared._parameter_hash(model)==parameters,
        "buffers_unchanged":all(torch.equal(v,buffers[k]) for k,v in model.named_buffers()),
        "all_modules_eval":not any(m.training for m in model.modules()),
        "predict_sample_accounting":samples==sum(m["actual_predict_samples"] for r in predictions for t in r["tasks"].values() for m in t.values()),
        "predictions_unchanged":digest(paths["predictions"])==prediction_sha,
        "weights_unchanged":digest(args.policy/"weights.pt")==baseline.OFFICIAL,
        "config_unchanged":digest(args.policy/"config.json")==protocol["checkpoint_config_sha256"],
        "adapter_unchanged":digest(args.adapter_artifact)==protocol["adapter_sha256"],
        "reference_unchanged":all(digest(Path(b["path"]))==b["sha256"] for b in bindings),
        "implementation_unchanged":all(digest(p)==protocol["implementation_sha256"][k] for k,p in modules.items()),
        "fixed_render_receipts":stats["requests"]>0 and stats["raw_render_passes"]==stats["exact_state_checks"]==2*stats["requests"] and stats["dither_state_changes"]==0}
    write_new(args.output,{"schema":"reacher_d2p_full_goal_metric_compiler.v1","status":"completed" if all(gates.values()) else "fail",
        "sealed_protocol":protocol,"protocol_sha256":digest(paths["protocol"]),"predictions_sha256":prediction_sha,
        "cuda_host":host,"gates":gates,"rows":rows,"render_stats":stats,"actual_predict_samples":samples,"P1_admission":False})
    print(json.dumps({"gates":gates,"actual_predict_samples":samples}),flush=True)
    if not all(gates.values()): raise RuntimeError("full metric compiler integrity failed")


if __name__ == "__main__":
    main()
