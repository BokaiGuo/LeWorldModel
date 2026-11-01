"""Host-CUDA regression replay of singleton incumbent/finiteness hardening.

All16 roots/all3 masks, both flat controls plus zero-prefix suffix CEM.
Historical future measurements are not rerun or counted as new outcomes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability import reacher_d2p_masked_programs as compiler
from experiments.spectral_reachability import reacher_d2p_masked_suffix_search as suffix
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,action="append",required=True)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--adapter-artifact",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("new immutable guard replay report required")
    if len(args.reference)!=2: raise ValueError("both complete known/fresh suffix panels required")
    dataset,source_sha = shared.receipt_source(args.adapter_artifact)
    bindings,panels,all_seeds,finite_receipts = {},[],set(),0
    for path in args.reference:
        report = json.loads(path.read_text())
        if report["status"]!="completed" or not all(v is True for v in report["gates"].values()):
            raise ValueError("complete integrity-passing suffix panel required")
        protocol = report["sealed_protocol"]
        pp = path.with_suffix(".predictions.json")
        predictions = json.loads(pp.read_text())
        bp = Path(protocol["reference_path"])
        baseline = json.loads(bp.read_text())
        bpp = bp.with_suffix(".predictions.json")
        old = json.loads(bpp.read_text())
        if (digest(pp)!=report["predictions_sha256"] or digest(bp)!=protocol["reference_sha256"]
                or digest(bpp)!=protocol["reference_prediction_sha256"]
                or digest(args.policy/"weights.pt")!=protocol["weights_sha256"]
                or digest(args.adapter_artifact)!=protocol["adapter_sha256"] or source_sha!=protocol["source_sha256_receipt"]):
            raise ValueError("unchanged model/source/baseline/prediction seals required")
        old_index = {r["seed"]:r for r in old["rows"]}
        seeds = {r["seed"] for r in predictions["rows"]}
        if len(seeds)!=8 or all_seeds&seeds or seeds!=set(old_index): raise ValueError("all16 distinct original roots required")
        all_seeds |= seeds
        for p in (path,pp,bp,bpp): bindings[str(p.resolve())] = digest(p)
        for r in predictions["rows"]:
            for plan in r["tasks"].values():
                for b in plan["bank"]:
                    costs = [b["cem_receipt"]["initial_cost"],b["cem_receipt"]["best_cost"],b["variants"]["old_gn4"]["terminal_cost"]]
                    if not all(math.isfinite(c) for c in costs) or costs[1]>costs[0] or not np.isclose(costs[0],costs[2],rtol=0,atol=1e-7):
                        raise ValueError("historical singleton initial/final receipts must be finite and model-incumbent consistent")
                    finite_receipts += 1
        panels.append((protocol,predictions["rows"],old_index,baseline))
    if finite_receipts!=816: raise ValueError("all816 historical suffix receipts required")
    modules = {str(p.resolve()):digest(p) for p in (Path(__file__),Path(compiler.__file__),Path(suffix.__file__),Path(chart.__file__),Path(goals.__file__),Path(p0.__file__))}
    print("HOST CUDA: replay all16 roots/all3 masks, exact old flat GN/CEM controls and zero-prefix suffix after numerical guard hardening. No new physical outcomes or efficacy claim.",flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean,std = shared._load_action_stats(args.adapter_artifact,"swm/ReacherDMControl-v0")
    normalizer = shared.ActionNormalizer(mean,std)
    processor = shared._Preprocessor(normalizer)
    mean_t,std_t = (torch.tensor(v,device="cuda") for v in (normalizer.block_mean,normalizer.block_std))
    model = shared.load_compatible_pretrained(args.policy,device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    params,buffers = shared._parameter_hash(model),{k:v.clone() for k,v in model.named_buffers()}
    predict,count = model.predict,0
    def counted(emb,act):
        nonlocal count
        if emb.device.type!="cuda" or act.device.type!="cuda": raise RuntimeError("CPU neural fallback forbidden")
        count += len(emb)
        return predict(emb,act)
    model.predict = counted
    rows = []
    try:
        for protocol,forecasts,old_index,baseline in panels:
            contract = baseline["sealed_protocol"]["goal_contract"]
            env = goals.goal_wrapper(contract)(gym.make("swm/ReacherDMControl-v0",render_mode="rgb_array"),target_delta=(.65,-.65))
            try:
                for forecast in forecasts:
                    seed = forecast["seed"]
                    old = old_index[seed]
                    ctx = shared._source_state_context(env,dataset=dataset,seed=seed)
                    if any(ctx[k]!=old[k] for k in ("source_episode","source_start_row","root_rgb_sha256","goal_rgb_sha256")):
                        raise RuntimeError("original source/goal/RGB root changed")
                    pixels,goal,history = p0._prepare(processor,ctx["frames"],ctx["root_frame"],ctx["goal"],np.asarray(ctx["history_action_blocks"],np.float32),torch.device("cuda"))
                    emb,target = p0._latent_context(model,pixels,history),p0._goal_latent(model,goal)
                    if target.tolist()!=old["goal_vector"]: raise RuntimeError("target encoding changed")
                    tasks = {}
                    for task in chart.TASKS:
                        before = count
                        width = len(chart.active_columns(task))
                        def root_batch(x):
                            n = len(x)
                            return chart.terminal(model,emb.expand(n,-1,-1),history.expand(n,-1,-1),x.reshape(n,3,width),task=task,mean=mean_t,std=std_t)
                        flat,gn_receipt = compiler.fit(lambda x:root_batch(x[None])[0],torch.zeros(3*width,device="cuda"),target)
                        with torch.no_grad(): value = root_batch(flat[None])[0]
                        controls = old["tasks"][task]["controls"]
                        def matches(commands,prediction,reference,horizon=3):
                            raw = chart.expand_commands(commands.reshape(1,horizon,width),task)[0].reshape(5*horizon,2).tolist()
                            return raw==reference["raw_actions"] and prediction.tolist()==reference["terminal_vector"]
                        gn_exact = matches(flat,value,controls["flat_gn"])
                        chosen,prediction,receipt = compiler.flat_cem_control(root_batch,flat,value,target,seed=seed)
                        cem_exact = matches(chosen,prediction,controls["flat_cem_gn_incumbent"])
                        b = forecast["tasks"][task]["bank"][0]
                        raw = np.asarray(b["variants"]["old_gn4"]["raw_actions"],np.float32)
                        prefix = torch.tensor(raw[:5].ravel()[chart.active_columns(task)],device="cuda")
                        initial = torch.tensor(raw[5:].ravel()[chart.active_columns(task,2)],device="cuda")
                        after,actions,_ = compiler.after_prefix(model,emb,history,prefix.reshape(1,1,width),task=task,mean=mean_t,std=std_t)
                        def suffix_batch(x):
                            n = len(x)
                            return chart.terminal(model,after.expand(n,-1,-1),actions.expand(n,-1,-1),x.reshape(n,2,width),task=task,mean=mean_t,std=std_t)
                        command,suffix_receipt = suffix.search(suffix_batch,initial,target,seed=seed+100000)
                        with torch.no_grad(): terminal = suffix_batch(command[None])[0]
                        expected = b["variants"]["cem_incumbent"]
                        suffix_raw = chart.expand_commands(command.reshape(1,2,width),task)[0].reshape(10,2).tolist()
                        suffix_exact = suffix_raw==expected["raw_actions"][5:] and terminal.tolist()==expected["terminal_vector"]
                        if not (gn_exact and cem_exact and suffix_exact and count-before==2029):
                            raise RuntimeError("valid original model actions/forecasts or accounting changed under guard hardening")
                        tasks[task] = {"flat_gn_exact":gn_exact,"flat_cem_exact":cem_exact,"zero_prefix_suffix_cem_exact":suffix_exact,
                            "singleton_fallback":receipt["singleton_fallback"],"actual_predict_samples":count-before,
                            "suffix_initial_cost":suffix_receipt["initial_cost"],"suffix_final_cost":suffix_receipt["best_cost"]}
                    rows.append({"seed":seed,"tasks":tasks})
                    print(json.dumps({"seed":seed,"all3_tasks_exact":True}),flush=True)
            finally: env.close()
    finally: model.predict = predict
    gates = {"cuda_only":True,"all16_roots":len(rows)==16 and len({r["seed"] for r in rows})==16,
        "all48_flat_controls_and_suffixes_exact":all(set(r["tasks"])==set(chart.TASKS) and all(c["flat_gn_exact"] is True and c["flat_cem_exact"] is True and c["zero_prefix_suffix_cem_exact"] is True for c in r["tasks"].values()) for r in rows),
        "all816_historical_suffix_receipts_finite":finite_receipts==816,"predict_accounting":count==97392,
        "parameters_unchanged":shared._parameter_hash(model)==params,"buffers_unchanged":all(torch.equal(v,buffers[k]) for k,v in model.named_buffers()),
        "all_modules_eval":not any(m.training for m in model.modules()),"references_unchanged":all(digest(Path(p))==s for p,s in bindings.items()),
        "implementation_unchanged":all(digest(Path(p))==s for p,s in modules.items())}
    write_new(args.output,{"schema":"reacher_d2p_singleton_guards_cuda.v1","status":"completed" if all(gates.values()) else "fail",
        "cuda_host":host,"gates":gates,"references":bindings,"implementation_sha256":modules,"actual_predict_samples":count,
        "physical_future_branches":0,"rows":rows,"P1_admission":False,
        "scope":"Host CUDA valid-input regression of latent guard defects, not evidence that either defect caused historical ranking reversals or new physical efficacy."})
    print(json.dumps({"gates":gates,"actual_predict_samples":count}),flush=True)


if __name__=="__main__": main()
