"""Same-prefix causal suffix-search intervention with both residual contracts.

All 8 known roots, all 3 tasks, all 17 prefixes, 3 suffixes retained. Frozen
official model, all forecasts sealed before outcomes, host CUDA only. Not P1.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability import reacher_d2p_masked_programs as compiler
from experiments.spectral_reachability import reacher_d2p_masked_suffix_search as search
from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability import reacher_d2p_proposal_native_selection as proposal
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.run_reacher_d2p_masked_program_pilot import check_raw, physical_row
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

VARIANTS = ("old_gn4", "cem_incumbent", "cem_then_gn4")


def prepare_variant_rows(bank,variant):
    return [{"index":b["index"],"prefix_cost":b["prefix_cost"],"terminal_cost":b["variants"][variant]["terminal_cost"],
        "common_response_gain":b["common_response_gain"]} for b in bank]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--adapter-artifact",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--proposal-native-selector",action="store_true")
    parser.add_argument("--source-disjoint-confirmation",action="store_true")
    args = parser.parse_args()
    paths = {k:args.output.with_suffix(s) for k,s in (("protocol",".protocol.json"),("predictions",".predictions.json"),("roots",".roots"),("implementation",".implementation"))}
    if any(p.exists() for p in (args.output,*paths.values())):
        raise FileExistsError("new immutable suffix-search report and sidecars required")
    ref = json.loads(args.reference.read_text())
    if (ref.get("schema") != "reacher_d2p_masked_program_pilot.v1" or ref.get("status") != "completed"
            or not ref.get("gates") or not all(v is True for v in ref["gates"].values())):
        raise ValueError("strict completed all8 original program report required")
    ref_prediction_path = args.reference.with_suffix(".predictions.json")
    old_predictions = json.loads(ref_prediction_path.read_text())
    if digest(ref_prediction_path) != ref["predictions_sha256"] or old_predictions["protocol_sha256"] != ref["protocol_sha256"]:
        raise ValueError("original pre-execution model seal changed")
    prior = ref["sealed_protocol"]
    goal_contract = prior.get("goal_contract","joint-feasible-v1")
    dataset,source_sha = shared.receipt_source(args.adapter_artifact)
    if (digest(args.policy/"weights.pt") != prior["weights_sha256"] or digest(args.adapter_artifact) != prior["adapter_sha256"]
            or source_sha != prior["source_sha256_receipt"]):
        raise ValueError("original official model/source/stats binding required")
    seeds = prior["root_seeds"]
    if args.source_disjoint_confirmation:
        excluded = set(prior.get("excluded_source_episodes",[]))
        episodes = {r["source_episode"] for r in prior["source_roots"]}
        if len(excluded) < 208 or len(episodes) != 8 or episodes & excluded:
            raise ValueError("confirmation requires eight fresh episodes disjoint from at least208 prior episodes")
    def indexed(rows):
        values = {r["seed"]:r for r in rows}
        if len(values) != len(rows) or set(values) != set(seeds) or len(values) != 8:
            raise ValueError("complete duplicate-free original root identities required")
        return values
    forecast,physical = indexed(old_predictions["rows"]),indexed(ref["rows"])
    modules = {"auditor":Path(__file__),"search":Path(search.__file__),"compiler":Path(compiler.__file__),
        "chart":Path(chart.__file__),"qp":Path(qp.__file__),"context":Path(execution.__file__),"goal_contract":Path(goals.__file__),
        "p0":Path(p0.__file__),"writer":Path(physical_row.__code__.co_filename),
        "source_context":Path(shared._source_state_context.__code__.co_filename),"loader":Path(shared.load_compatible_pretrained.__code__.co_filename)}
    modules["feasible_goal"] = Path(goals.feasible_goal_delta.__code__.co_filename)
    protocol = {"root_seeds":seeds,"source_roots":prior["source_roots"],"reference_path":str(args.reference.resolve()),
        "reference_sha256":digest(args.reference),"reference_prediction_sha256":digest(ref_prediction_path),
        "weights_sha256":prior["weights_sha256"],"adapter_sha256":prior["adapter_sha256"],"source_sha256_receipt":source_sha,
        "tasks":prior["tasks"],"variants":list(VARIANTS),"prefix_packets":1,"future_packets":2,"bank_count":17,
        "cem":{"samples":128,"steps":3,"elite":16,"seed_law":"root seed + 100000 + prefix index","initial":"same fixed old GN4 suffix"},
        "gn_after_cem_iterations":4,"damping":.01,"prefixes_goals_thresholds_unchanged":True,
        "selector_contract":"unchanged strict common-residual qualification, separately applied to each complete suffix bank",
        "own_residual":"goal minus predicted two-packet zero-suffix endpoint from the particular predicted post-prefix state",
        "common_residual":"unchanged goal minus predicted original-root three-packet zero-action endpoint",
        "response_saved":"complete 192-by-active-coordinate raw-action Jacobian at the literal-zero suffix",
        "all_predictions_sealed_before_any_future_execution":True,"no_post_training":True,"compute_matched":False,
        "physical_branches":1224,"expected_predict_samples":349320,
        "implementation_sha256":{k:digest(v) for k,v in modules.items()},
        "scope":"Known-root causal solver/metric-contract audit, not fresh confirmation, full efficacy, P1 admission or theory rejection."}
    if args.proposal_native_selector:
        modules["proposal_selector"] = Path(proposal.__file__)
        protocol["proposal_native_contract"] = proposal.CONTRACT
    if args.source_disjoint_confirmation:
        protocol["source_disjoint_confirmation"] = True
        protocol["excluded_source_episodes"] = prior["excluded_source_episodes"]
        protocol["scope"] = "Fresh source-disjoint confirmation of fixed suffix-search and explicit proposal-native objective; original baseline is retained as paired control. Not calibrated P1, compute-matched efficacy or theory rejection."
    protocol["implementation_sha256"] = {k:digest(v) for k,v in modules.items()}
    protocol["goal_contract"] = goal_contract
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_new(paths["protocol"],protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for k,v in modules.items(): shutil.copyfile(v,paths["implementation"]/(k+".py"))
    print("HOST CUDA: keep every original prefix/goal, add incumbent-preserving suffix CEM and CEM+GN; save full response matrices and both residual contracts. Seal all forecasts before all1224 physical branches. No P1 claim.",flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean,std = shared._load_action_stats(args.adapter_artifact,"swm/ReacherDMControl-v0")
    normalizer = shared.ActionNormalizer(mean,std)
    processor = shared._Preprocessor(normalizer)
    mean_t,std_t = (torch.tensor(v,device="cuda") for v in (normalizer.block_mean,normalizer.block_std))
    model = shared.load_compatible_pretrained(args.policy,device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameters,buffers = shared._parameter_hash(model),{k:v.clone() for k,v in model.named_buffers()}
    original_predict,samples = model.predict,0
    def counted(emb,act):
        nonlocal samples
        if emb.device.type != "cuda" or act.device.type != "cuda": raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return original_predict(emb,act)
    model.predict = counted
    env = goals.goal_wrapper(goal_contract)(gym.make("swm/ReacherDMControl-v0",render_mode="rgb_array"),target_delta=(.65,-.65))
    def prepare(ctx):
        return p0._prepare(processor,ctx["frames"],ctx["root_frame"],ctx["goal"],np.asarray(ctx["history_action_blocks"],np.float32),torch.device("cuda"))
    pending,predictions,rows = [],[],[]
    try:
        for seed in seeds:
            old = forecast[seed]
            ctx = shared._source_state_context(env,dataset=dataset,seed=seed)
            ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
            if (any(ctx[k] != old[k] for k in ("source_episode","source_start_row","root_rgb_sha256","goal_rgb_sha256"))
                    or ctx["execution_snapshot"] != old["initial_snapshot"]):
                raise RuntimeError("original root/goal/full snapshot not exactly reproduced")
            pixels,goal,history = prepare(ctx)
            emb,target = p0._latent_context(model,pixels,history),p0._goal_latent(model,goal)
            if target.tolist() != old["goal_vector"]: raise RuntimeError("full original goal coordinate changed")
            tasks = {}
            for task in chart.TASKS:
                width = len(chart.active_columns(task))
                started,before = time.monotonic(),samples
                def root_terminal(flat):
                    return chart.terminal(model,emb,history,flat.reshape(1,3,width),task=task,mean=mean_t,std=std_t)[0]
                with torch.no_grad(): common = target-root_terminal(torch.zeros(3*width,device="cuda"))
                if common.tolist() != old["tasks"][task]["common_model_residual"]:
                    raise RuntimeError("fixed original common residual changed")
                bank = []
                for old_branch in old["tasks"][task]["bank"]:
                    i = old_branch["index"]
                    raw = np.asarray(old_branch["raw_actions"],np.float32)
                    prefix = torch.tensor(raw[:5].ravel()[chart.active_columns(task)],device="cuda")
                    initial = torch.tensor(raw[5:].ravel()[chart.active_columns(task,2)],device="cuda")
                    after,actions,prefix_prediction = compiler.after_prefix(model,emb,history,prefix.reshape(1,1,width),task=task,mean=mean_t,std=std_t)
                    if prefix_prediction[0].tolist() != old_branch["prefix_vector"]: raise RuntimeError("original prefix prediction changed")
                    def suffix_terminal(flat):
                        return chart.terminal(model,after,actions,flat.reshape(1,2,width),task=task,mean=mean_t,std=std_t)[0]
                    def batch_terminal(flat):
                        n = len(flat)
                        return chart.terminal(model,after.expand(n,-1,-1),actions.expand(n,-1,-1),flat.reshape(n,2,width),task=task,mean=mean_t,std=std_t)
                    zero = torch.zeros(2*width,device="cuda")
                    with torch.no_grad(): zero_value = suffix_terminal(zero)
                    response = torch.autograd.functional.jacobian(suffix_terminal,zero.requires_grad_(True),vectorize=True).detach()
                    own = (target-zero_value).detach()
                    gains = {name:qp.certified_response_gain(response.to(torch.float64),residual.to(torch.float64),damping=.01)
                        for name,residual in (("common",common),("own",own))}
                    if not np.isclose(gains["common"]["bounded_quadratic_gain"],old_branch["common_response_gain"]["bounded_quadratic_gain"],rtol=1e-10,atol=1e-10):
                        raise RuntimeError("original common-response gain changed")
                    def variant(commands):
                        full = torch.cat((prefix,commands)).reshape(1,3,width)
                        with torch.no_grad(): split,direct = suffix_terminal(commands),root_terminal(full.ravel())
                        if not torch.equal(split,direct): raise RuntimeError("same-action split/full recurrence changed")
                        return {"raw_actions":chart.expand_commands(full,task)[0].reshape(15,2).tolist(),"terminal_vector":direct.tolist(),
                            "terminal_cost":float((direct-target).square().mean()),"full_split_exact":True}
                    legacy = variant(initial)
                    if legacy["raw_actions"] != old_branch["raw_actions"] or legacy["terminal_vector"] != old_branch["terminal_vector"]:
                        raise RuntimeError("original complete command/prediction not exactly reproduced")
                    cem,cem_receipt = search.search(batch_terminal,initial,target,seed=seed+100000+i)
                    cem_branch = variant(cem)
                    refined,gn_receipt = compiler.fit(suffix_terminal,cem,target,iterations=4)
                    gn_branch = variant(refined)
                    variants = {"old_gn4":legacy,"cem_incumbent":cem_branch,"cem_then_gn4":gn_branch}
                    if cem_branch["terminal_cost"] > legacy["terminal_cost"] or gn_branch["terminal_cost"] > cem_branch["terminal_cost"]:
                        raise RuntimeError("intervention lost its model incumbent")
                    bank.append({"index":i,"prefix_vector":prefix_prediction[0].tolist(),"prefix_cost":old_branch["prefix_cost"],
                        "zero_suffix_vector":zero_value.tolist(),"response":response.tolist(),"own_residual":own.tolist(),
                        "common_response_gain":gains["common"],"own_response_gain":gains["own"],"variants":variants,
                        "cem_receipt":cem_receipt,"gn_receipt":gn_receipt})
                decisions = {v:compiler.select(prepare_variant_rows(bank,v)) for v in VARIANTS}
                if decisions["old_gn4"] != old["tasks"][task]["decision"]: raise RuntimeError("original selector decision changed")
                torch.cuda.synchronize()
                tasks[task] = {"bank":bank,"decisions":decisions,"common_residual":common.tolist(),"actual_predict_samples":samples-before,
                    "planning_seconds":time.monotonic()-started}
                if args.proposal_native_selector:
                    tasks[task]["proposal_native_decisions"] = {v:proposal.select(bank,v) for v in VARIANTS}
                print(json.dumps({"seed":seed,"task":task,"phase":"predictions","predict_samples":samples-before,
                    "witness_indices":{v:d["qualified_indices"] for v,d in decisions.items()}}),flush=True)
            record = {**{k:old[k] for k in ("seed","source_episode","source_start_row","goal_vector","root_rgb_sha256","goal_rgb_sha256","goal_admission","initial_snapshot")},"tasks":tasks}
            predictions.append(record)
            pending.append((ctx,record))
        write_new(paths["predictions"],{"protocol_sha256":digest(paths["protocol"]),"rows":predictions})
        prediction_sha = digest(paths["predictions"])
        print(json.dumps({"all_predictions_presealed":True,"predictions_sha256":prediction_sha}),flush=True)
        replays = []
        for ctx,record in pending:
            seed = record["seed"]
            tasks = {}
            for task,plan in record["tasks"].items():
                branches = {}
                for b in plan["bank"]:
                    results = {}
                    for variant_name in VARIANTS:
                        branch = b["variants"][variant_name]
                        raw = np.asarray(branch["raw_actions"],np.float32)
                        if not check_raw(raw,task): raise RuntimeError("literal mask/raw torque witness changed")
                        frame,goal = execution.restore_execution_snapshot(env,seed=seed,snapshot=ctx["execution_snapshot"],root_identity="sealed_observation_exact_state")
                        p0._verify_execution_identity(frame,goal,ctx["root_frame"],ctx["goal"])
                        current,packets = ctx,[]
                        for j in range(3):
                            current = execution.advance_executed_context(env,context=current,raw_action_block=raw[j*5:(j+1)*5],action_block=5,verified_root_frame=current["root_frame"])
                            pixels,_,hist = prepare(current)
                            with torch.no_grad(): observed = p0._latent_context(model,pixels,hist)[0,-1]
                            packets.append({"packet":j+1,"observed_vector":observed.tolist(),"qpos":np.asarray(env.unwrapped.env.physics.data.qpos,np.float64)[:2].tolist(),
                                "root_rgb_sha256":current["root_rgb_sha256"],"action_sha256":current["advanced_action_sha256"]})
                        measured = {"raw_actions":raw.tolist(),"packets":packets}
                        if variant_name == "old_gn4":
                            exact = measured == physical[seed]["tasks"][task]["bank_"+str(b["index"])]
                            replays.append(exact)
                            if not exact: raise RuntimeError("original physical branch no longer reproduces exactly")
                        results[variant_name] = measured
                    branches["bank_"+str(b["index"])] = results
                tasks[task] = branches
                print(json.dumps({"seed":seed,"task":task,"phase":"physical","all51_branches":True}),flush=True)
            row = physical_row(record,tasks)
            write_new(paths["roots"]/f"root_{seed}.json",row)
            rows.append(row)
    finally:
        model.predict = original_predict
        env.close()
    gates = {"cuda_only":True,"all8_roots":[r["seed"] for r in rows]==seeds,
        "all_tasks_and1224_branches":len(rows)==8 and all(set(r["tasks"])==set(chart.TASKS) and all(len(t)==17 and all(set(b)==set(VARIANTS) for b in t.values()) for t in r["tasks"].values()) for r in rows),
        "all408_original_physical_replays_exact":len(replays)==408 and all(replays),
        "all_model_incumbents_preserved":all(b["variants"]["cem_then_gn4"]["terminal_cost"]<=b["variants"]["cem_incumbent"]["terminal_cost"]<=b["variants"]["old_gn4"]["terminal_cost"] for r in predictions for t in r["tasks"].values() for b in t["bank"]),
        "all_full_split_exact":all(v["full_split_exact"] is True for r in predictions for t in r["tasks"].values() for b in t["bank"] for v in b["variants"].values()),
        "all_full_responses_saved":all(np.asarray(b["response"]).shape==(192,2*len(chart.active_columns(task))) for r in predictions for task,t in r["tasks"].items() for b in t["bank"]),
        "predict_sample_accounting":samples==349320,"all_task_compute_receipts":all(t["actual_predict_samples"]==14555 for r in predictions for t in r["tasks"].values()),
        "parameters_unchanged":shared._parameter_hash(model)==parameters,"buffers_unchanged":all(torch.equal(v,buffers[k]) for k,v in model.named_buffers()),
        "all_modules_eval":not any(m.training for m in model.modules()),"predictions_unchanged":digest(paths["predictions"])==prediction_sha,
        "reference_unchanged":digest(args.reference)==protocol["reference_sha256"] and digest(ref_prediction_path)==protocol["reference_prediction_sha256"],
        "weights_file_unchanged":digest(args.policy/"weights.pt")==protocol["weights_sha256"],
        "implementation_unchanged":all(digest(v)==protocol["implementation_sha256"][k] for k,v in modules.items())}
    write_new(args.output,{"schema":"reacher_d2p_masked_suffix_search.v1","status":"completed" if all(gates.values()) else "fail","cuda_host":host,
        "sealed_protocol":protocol,"protocol_sha256":digest(paths["protocol"]),"predictions_sha256":prediction_sha,"gates":gates,
        "physical_branches":1224,"actual_model_predict_samples":samples,"rows":rows})
    print(json.dumps({"gates":gates,"actual_model_predict_samples":samples}),flush=True)


if __name__ == "__main__": main()
