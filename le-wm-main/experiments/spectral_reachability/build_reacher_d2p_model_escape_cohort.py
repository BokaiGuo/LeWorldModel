"""Host-CUDA model-only plateau/escape cohort, without physical futures.

Keeps all32 source-admitted roots and all3 actuator tasks. This is task-state
admission for a later P1 experiment, not a retrospective change to old gates.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.spectral_reachability import run_reacher_d2p_goal_metric_compiler as foundation
from experiments.spectral_reachability import reacher_d2p_model_escape_admission as admission

shared,execution,goals,chart = foundation.shared,foundation.execution,foundation.goals,foundation.chart
p0,render,runtime = foundation.p0,foundation.render,foundation.runtime
compiler,weighted = foundation.baseline.compiler,foundation.compiler.weighted_terminal
digest,write_new = foundation.digest,foundation.write_new


def measure_task(model, emb, history, target, *, metric, task, mean, std, seed):
    if emb.device.type != "cuda" or metric.device != emb.device or target.device != emb.device:
        raise RuntimeError("host CUDA model admission required; neural CPU fallback forbidden")
    width = len(chart.active_columns(task))
    def terminal(action):
        return chart.terminal(model,emb,history,action.reshape(1,3,width),task=task,mean=mean,std=std)[0]
    def batch(action):
        n = len(action)
        return chart.terminal(model,emb.expand(n,-1,-1),history.expand(n,-1,-1),
            action.reshape(n,3,width),task=task,mean=mean,std=std)
    single,batch_weighted = weighted(terminal,metric),weighted(batch,metric)
    expected = metric.apply(target.double())
    current = torch.zeros(3*width,device=emb.device,dtype=emb.dtype)
    def record(action):
        with torch.no_grad(): vector = single(action)
        return {"weighted_vector":vector.tolist(),"cost":float((vector-expected).square().mean()),
            "raw_actions":chart.expand_commands(action.reshape(1,3,width),task)[0].reshape(15,2).tolist()},vector
    values = {}
    values["zero"],_ = record(current)
    receipts = {}
    for total,iterations in ((4,4),(8,4),(16,8)):
        current,receipt = compiler.fit(single,current,expected,iterations=iterations)
        key = "gn"+str(total)
        values[key],vector = record(current)
        receipts[key] = receipt
    cem,cem_vector,cem_receipt = compiler.flat_cem_control(batch_weighted,current,vector,expected,seed=seed)
    values["cem"] = {"weighted_vector":cem_vector.tolist(),"cost":float((cem_vector-expected).square().mean()),
        "raw_actions":chart.expand_commands(cem.reshape(1,3,width),task)[0].reshape(15,2).tolist()}
    action = current.detach().clone().requires_grad_(True)
    cost = (single(action)-expected).square().mean()
    gradient = torch.autograd.grad(cost,action)[0]
    if not torch.isfinite(gradient).all(): raise RuntimeError("nonfinite model task gradient")
    mapping = action.detach()-(action.detach()-gradient.detach()).clamp(-1,1)
    values.update({"gn_receipts":receipts,"cem_receipt":cem_receipt,
        "gn16_gradient":gradient.tolist(),"gn16_projected_gradient_inf":float(mapping.abs().max()),
        "logical_action_vjp_rows":16*192+1,"physical_future_labels_read":False})
    admission.task_evidence(values)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--adapter-artifact",type=Path,required=True)
    parser.add_argument("--disjoint-reference",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=202656000)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    paths = {key:args.output.with_suffix(suffix) for key,suffix in (("protocol",".protocol.json"),
        ("roots",".roots"),("implementation",".implementation"))}
    if any(p.exists() for p in (args.output,*paths.values())):
        raise FileExistsError("new immutable model-cohort output required")
    prior_report = json.loads(args.disjoint_reference.read_text())
    prior = prior_report["sealed_protocol"]
    seeds = list(range(args.seed,args.seed+admission.CONTRACT["metadata_scan_count"]))
    excluded,bindings = foundation.baseline.bind_exclusions([args.disjoint_reference],seeds)
    dataset,source_sha = shared.receipt_source(args.adapter_artifact)
    if (args.seed < 202656000 or len(excluded) < 248 or prior["weights_sha256"] != foundation.baseline.OFFICIAL
            or digest(args.policy/"weights.pt") != foundation.baseline.OFFICIAL
            or digest(args.adapter_artifact) != prior["adapter_sha256"] or source_sha != prior["source_sha256_receipt"]):
        raise ValueError("official frozen checkpoint/stats/receipt and248 prior episode exclusions required")
    with h5py.File(dataset,"r",swmr=True) as handle:
        pool = admission.metadata_pool(np.asarray(handle["ep_len"],np.int64),np.asarray(handle["ep_offset"],np.int64),excluded,seed=args.seed)
    modules = {"runner":Path(__file__),"admission":Path(admission.__file__),
        "foundation":Path(foundation.__file__),"shared":Path(shared.__file__),"context":Path(execution.__file__),
        "goals":Path(goals.__file__),"chart":Path(chart.__file__),"p0":Path(p0.__file__),
        "goal_runtime":Path(runtime.__file__),"metric":Path(foundation.metric.__file__),
        "weighted_compiler":Path(foundation.compiler.__file__),"raw_compiler":Path(compiler.__file__),
        "qp":Path(foundation.baseline.qp.__file__),"render":Path(render.__file__),
        "model_encode":runtime.declared_encoder_source(args.policy)}
    for key,fn in (("source_context",shared._source_state_context),("loader",shared.load_compatible_pretrained),
            ("exclusions",foundation.baseline.bind_exclusions),("goal_task",goals.feasible_goal_delta)):
        modules[key] = Path(fn.__code__.co_filename)
    protocol = {"contract":admission.CONTRACT,"source_pool":pool,"excluded_source_episodes":sorted(excluded),
        "exclusion_bindings":bindings,"weights_sha256":foundation.baseline.OFFICIAL,
        "checkpoint_config_sha256":digest(args.policy/"config.json"),"adapter_sha256":digest(args.adapter_artifact),
        "source_sha256_receipt":source_sha,"goal_contract":"joint-feasible-v2-soft-source","target_delta":[.65,-.65],
        "tasks":{k:list(v) for k,v in chart.TASKS.items()},"metric_policy":foundation.metric.POLICY,
        "expected_predict_samples_per_task":1554,"expected_predict_samples":149184,
        "expected_encoder_forwards":416,"expected_input_vjp_rows":6144,"expected_action_vjp_rows":295008,
        "physical_future_branches":0,"selection_uses_only_model_predictions":True,"P1_admission":False,
        "implementation_sha256":{key:digest(path) for key,path in modules.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_new(paths["protocol"],protocol)
    paths["implementation"].mkdir()
    paths["roots"].mkdir()
    for key,path in modules.items(): shutil.copyfile(path,paths["implementation"]/(key+".py"))
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
    original_predict,count = model.predict,0
    def counted(emb,action):
        nonlocal count
        if emb.device.type != "cuda" or action.device != emb.device:
            raise RuntimeError("neural CPU fallback forbidden")
        count += len(emb)
        return original_predict(emb,action)
    model.predict = counted
    env = goals.goal_wrapper(protocol["goal_contract"])(gym.make("swm/ReacherDMControl-v0",render_mode="rgb_array"),target_delta=(.65,-.65))
    renderer = render.FixedWarmRenderer(env)
    rows = []
    try:
        for root in pool["roots"]:
            ctx = shared._source_state_context(env,dataset=dataset,seed=root["seed"])
            if any(ctx[k] != root[k] for k in ("source_episode","source_start_row")):
                raise RuntimeError("metadata-admitted source history changed; no replacement allowed")
            pixels,goal,history = p0._prepare(processor,ctx["frames"],ctx["root_frame"],ctx["goal"],
                np.asarray(ctx["history_action_blocks"],np.float32),torch.device("cuda"))
            emb,target = p0._latent_context(model,pixels,history),p0._goal_latent(model,goal)
            metric,receipt = runtime.measure_goal(model,goal,target.tolist(),seed=root["seed"])
            tasks = {}
            for task in chart.TASKS:
                before = count
                tasks[task] = measure_task(model,emb,history,target,metric=metric,task=task,mean=mean_t,std=std_t,seed=root["seed"])
                tasks[task]["actual_predict_samples"] = count-before
                if count-before != protocol["expected_predict_samples_per_task"]:
                    raise RuntimeError("fixed model admission prediction accounting mismatch: "+str(count-before))
                print(json.dumps({"seed":root["seed"],"task":task,"phase":"model_only_admission",
                    "predict_samples":count-before,"evidence":admission.task_evidence(tasks[task])}),flush=True)
            row = {**root,"root_rgb_sha256":ctx["root_rgb_sha256"],"goal_rgb_sha256":ctx["goal_rgb_sha256"],
                "goal_vector":target.tolist(),"goal_metric":receipt,"tasks":tasks}
            write_new(paths["roots"]/f"root_{root['seed']}.json",row)
            rows.append(row)
    finally:
        model.predict = original_predict
        renderer.restore()
        env.close()
    selection = admission.select(rows)
    gates = {"cuda_only":True,"all32_roots":[r["seed"] for r in rows] == [r["seed"] for r in pool["roots"]],
        "all3_tasks":all(set(r["tasks"]) == set(chart.TASKS) for r in rows),
        "source_disjoint":len({r["source_episode"] for r in rows}) == 32 and not ({r["source_episode"] for r in rows}&excluded),
        "predict_accounting":count == protocol["expected_predict_samples"],
        "encoder_accounting":sum(r["goal_metric"]["encoder_forwards"] for r in rows) == 416 and sum(r["goal_metric"]["input_vjp_rows"] for r in rows) == 6144,
        "action_vjp_accounting":sum(t["logical_action_vjp_rows"] for r in rows for t in r["tasks"].values()) == 295008,
        "parameters_unchanged":shared._parameter_hash(model) == parameters,
        "buffers_unchanged":all(torch.equal(value,buffers[key]) for key,value in model.named_buffers()),
        "all_modules_eval":not any(m.training for m in model.modules()),
        "weights_unchanged":digest(args.policy/"weights.pt") == protocol["weights_sha256"],
        "config_unchanged":digest(args.policy/"config.json") == protocol["checkpoint_config_sha256"],
        "adapter_unchanged":digest(args.adapter_artifact) == protocol["adapter_sha256"],
        "reference_unchanged":all(digest(Path(b["path"])) == b["sha256"] for b in bindings),
        "implementation_unchanged":all(digest(path) == protocol["implementation_sha256"][key] for key,path in modules.items())}
    write_new(args.output,{"schema":"reacher_d2p_model_escape_cohort.v1","status":"completed" if all(gates.values()) else "fail",
        "sealed_protocol":protocol,"protocol_sha256":digest(paths["protocol"]),"cuda_host":host,
        "rows":rows,"selection":selection,"gates":gates,"actual_predict_samples":count,
        "physical_future_branches":0,"P1_admission":False})
    print(json.dumps({"gates":gates,"selection_status":selection["status"],
        "eligible_count":selection["eligible_count"],"selected_seeds":selection["selected_seeds"]}),flush=True)
    if not all(gates.values()): raise RuntimeError("model-only cohort integrity failed")


if __name__ == "__main__":
    main()
