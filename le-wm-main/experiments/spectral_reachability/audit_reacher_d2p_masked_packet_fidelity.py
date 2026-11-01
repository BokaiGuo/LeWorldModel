"""Localize remaining fixed-action RGB ranking errors on both complete panels.

Keep all16 roots/3 masks and common-selector, same-bank greedy, literal-zero
prefix programs, with their sealed CEM+GN suffixes. Actual-history predictions
are evaluator-only; they do not alter actions or count as deployed fidelity.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_worldmodel.wm.lewm import lewm as released

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.analyze_reacher_d2p_masked_suffix_search import verify_files
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

FAMILIES = ("common_native","same_bank_greedy","zero_prefix")


def family_indices(plan):
    decision = plan["decisions"]["cem_then_gn4"]
    if (len(plan["bank"])!=17 or [b["index"] for b in plan["bank"]]!=list(range(17))
            or decision["physical_labels_used"] is not False
            or any(not isinstance(decision[k],int) or not 0<=decision[k]<17 for k in ("native_index","greedy_index"))):
        raise ValueError("complete original model-only bank/decision required")
    return dict(zip(FAMILIES,(decision["native_index"],decision["greedy_index"],0)))


def validate_packets(packets):
    if len(packets)!=3 or [p["packet"] for p in packets]!=[1,2,3]: raise ValueError("all3 ordered packet observations required")
    for p in packets:
        arrays = {k:np.asarray(v,np.float64) for k,v in p["vectors"].items()}
        if set(arrays)!={"free","actual_history","observed","goal"} or any(v.shape!=(192,) or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("all complete finite192-coordinate packet vectors required")
        for k in ("free","actual_history","observed"):
            if not np.isclose(np.mean((arrays[k]-arrays["goal"])**2),p["costs"][k],rtol=2e-6,atol=1e-7):
                raise ValueError("packet costs do not match full coordinates")
    if packets[0]["vectors"]["free"]!=packets[0]["vectors"]["actual_history"]:
        raise ValueError("first free/actual-history forecast must be bit-exact")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,action="append",required=True)
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--adapter-artifact",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    paths = {k:args.output.with_suffix(s) for k,s in (("protocol",".protocol.json"),("roots",".roots"),("implementation",".implementation"))}
    if len(args.reference)!=2 or any(p.exists() for p in (args.output,*paths.values())):
        raise ValueError("both immutable complete panels and new output required")
    panels,bindings,seeds = [],[],set()
    dataset,source_sha = shared.receipt_source(args.adapter_artifact)
    for path in args.reference:
        verify_files(path)
        report = json.loads(path.read_text())
        prior = report["sealed_protocol"]
        pp = path.with_suffix(".predictions.json")
        forecast = json.loads(pp.read_text())
        if (digest(args.policy/"weights.pt")!=prior["weights_sha256"] or digest(args.adapter_artifact)!=prior["adapter_sha256"]
                or source_sha!=prior["source_sha256_receipt"]):
            raise ValueError("unchanged official model/source/stats required")
        roots = {r["seed"]:r for r in report["rows"]}
        if len(roots)!=8 or seeds&set(roots): raise ValueError("all16 distinct original roots required")
        seeds |= set(roots)
        panels.append((prior,forecast["rows"],roots))
        bindings.append({"path":str(path.resolve()),"sha256":digest(path),"predictions_path":str(pp.resolve()),"predictions_sha256":digest(pp)})
    modules = {"auditor":Path(__file__),"chart":Path(chart.__file__),"context":Path(execution.__file__),"p0":Path(p0.__file__),
        "goal_contract":Path(goals.__file__),"source_context":Path(shared._source_state_context.__code__.co_filename),"released":Path(released.__file__),
        "feasible_goal":Path(goals.feasible_goal_delta.__code__.co_filename),"loader":Path(shared.load_compatible_pretrained.__code__.co_filename)}
    protocol = {"root_seeds":sorted(seeds),"references":bindings,"families":list(FAMILIES),"tasks":{k:list(v) for k,v in chart.TASKS.items()},
        "variant":"cem_then_gn4","packets":3,"physical_branches":144,"expected_predict_samples":1728,
        "weights_sha256":digest(args.policy/"weights.pt"),"adapter_sha256":digest(args.adapter_artifact),"source_sha256_receipt":source_sha,
        "actions_and_decisions_unchanged":True,"actual_history_use":"evaluator-only prediction immediately before the same sealed packet; no oracle selection or action change",
        "implementation_sha256":{k:digest(v) for k,v in modules.items()},
        "scope":"Both known panels, all16 roots/3 masks; fixed-action packetwise error localization, not new-source efficacy, deployed correction, P1 or theory rejection."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_new(paths["protocol"],protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for k,v in modules.items(): shutil.copyfile(v,paths["implementation"]/(k+".py"))
    print("HOST CUDA: replay all16 roots/3 masks/3 sealed programs. Original actions and physical packet identities unchanged. Separate free, released-native, and evaluator-only actual-history predictions; no P1 claim.",flush=True)
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
    def counted(emb,actions):
        nonlocal count
        if emb.device.type!="cuda" or actions.device.type!="cuda": raise RuntimeError("CPU neural fallback forbidden")
        count += len(emb)
        return predict(emb,actions)
    model.predict = counted
    def prepare(ctx):
        return p0._prepare(processor,ctx["frames"],ctx["root_frame"],ctx["goal"],np.asarray(ctx["history_action_blocks"],np.float32),torch.device("cuda"))
    rows = []
    try:
        for prior,forecasts,physical in panels:
            contract = prior.get("goal_contract","joint-feasible-v1")
            env = goals.goal_wrapper(contract)(gym.make("swm/ReacherDMControl-v0",render_mode="rgb_array"),target_delta=(.65,-.65))
            try:
                for forecast in forecasts:
                    seed = forecast["seed"]
                    ctx = shared._source_state_context(env,dataset=dataset,seed=seed)
                    ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
                    if any(ctx[k]!=forecast[k] for k in ("source_episode","source_start_row","root_rgb_sha256","goal_rgb_sha256")) or ctx["execution_snapshot"]!=forecast["initial_snapshot"]:
                        raise RuntimeError("original full source/root/goal snapshot changed")
                    pixels,goal,history = prepare(ctx)
                    emb,target = p0._latent_context(model,pixels,history),p0._goal_latent(model,goal)
                    if target.tolist()!=forecast["goal_vector"]: raise RuntimeError("original goal coordinate changed")
                    tasks = {}
                    for task,plan in forecast["tasks"].items():
                        branches = {}
                        for family,index in family_indices(plan).items():
                            b = plan["bank"][index]["variants"]["cem_then_gn4"]
                            measured = physical[seed]["tasks"][task]["bank_"+str(index)]["cem_then_gn4"]
                            raw = np.asarray(b["raw_actions"],np.float32)
                            commands = torch.tensor(raw.reshape(1,3,10)[:,:,chart.active_columns(task)],device="cuda")
                            future = chart.normalize_raw(chart.expand_commands(commands,task),mean_t,std_t)
                            frame,goal_rgb = execution.restore_execution_snapshot(env,seed=seed,snapshot=ctx["execution_snapshot"],root_identity="sealed_observation_exact_state")
                            p0._verify_execution_identity(frame,goal_rgb,ctx["root_frame"],ctx["goal"])
                            current,packets = ctx,[]
                            free_emb,free_actions = emb,torch.cat((history[:,1:],future[:,:1]),dim=1)
                            with torch.no_grad():
                                direct = chart.terminal_normalized(model,emb,history,future)[0]
                                native = model.rollout({"pixels":pixels[:,None]},torch.cat((history[:,1:],future),dim=1)[:,None],history_size=3)["predicted_emb"][0,0,-3:]
                                for j in range(3):
                                    free = model.predict(free_emb[:,-3:],model.action_encoder(free_actions[:,-3:]))[0,-1]
                                    observed_pixels,_,observed_history = prepare(current)
                                    observed_emb = p0._latent_context(model,observed_pixels,observed_history)
                                    forced = chart.terminal_normalized(model,observed_emb,observed_history,future[:,j:j+1])[0]
                                    current = execution.advance_executed_context(env,context=current,raw_action_block=raw[j*5:(j+1)*5],action_block=5,verified_root_frame=current["root_frame"])
                                    new_pixels,_,new_history = prepare(current)
                                    actual = p0._latent_context(model,new_pixels,new_history)[0,-1]
                                    physical_packet = {"packet":j+1,"observed_vector":actual.tolist(),"qpos":np.asarray(env.unwrapped.env.physics.data.qpos,np.float64)[:2].tolist(),
                                        "root_rgb_sha256":current["root_rgb_sha256"],"action_sha256":current["advanced_action_sha256"]}
                                    if physical_packet!=measured["packets"][j]: raise RuntimeError("original complete physical packet replay changed")
                                    if not torch.equal(free,native[j]): raise RuntimeError("released and raw-chart free rollout differ")
                                    vectors = {"free":free,"actual_history":forced,"observed":actual,"goal":target}
                                    packets.append({"packet":j+1,"vectors":{k:v.tolist() for k,v in vectors.items()},
                                        "costs":{k:float((v-target).square().mean()) for k,v in vectors.items() if k!="goal"},
                                        "physical":physical_packet,"released_vector":native[j].tolist()})
                                    free_emb = torch.cat((free_emb,free[None,None]),dim=1)
                                    if j<2: free_actions = torch.cat((free_actions,future[:,j+1:j+2]),dim=1)
                            validate_packets(packets)
                            if direct.tolist()!=b["terminal_vector"] or packets[-1]["vectors"]["free"]!=b["terminal_vector"]:
                                raise RuntimeError("original complete terminal forecast changed")
                            branches[family] = {"bank_index":index,"raw_actions":raw.tolist(),"packets":packets,
                                "all_physical_packets_exact":True,"released_all_packets_exact":True,"original_terminal_exact":True}
                        tasks[task] = branches
                    row = {"seed":seed,"source_episode":forecast["source_episode"],"source_start_row":forecast["source_start_row"],"tasks":tasks}
                    write_new(paths["roots"]/f"root_{seed}.json",row)
                    rows.append(row)
                    print(json.dumps({"seed":seed,"all9_programs_and27_packets_exact":True}),flush=True)
            finally: env.close()
    finally: model.predict = predict
    gates = {"cuda_only":True,"all16_roots":sorted(r["seed"] for r in rows)==sorted(seeds),
        "all144_fixed_programs_and432_packets_exact":len(rows)==16 and all(set(r["tasks"])==set(chart.TASKS) and all(set(t)==set(FAMILIES) and all(b["all_physical_packets_exact"] is True and b["released_all_packets_exact"] is True and b["original_terminal_exact"] is True for b in t.values()) for t in r["tasks"].values()) for r in rows),
        "predict_accounting":count==1728,"parameters_unchanged":shared._parameter_hash(model)==params,
        "buffers_unchanged":all(torch.equal(v,buffers[k]) for k,v in model.named_buffers()),"all_modules_eval":not any(m.training for m in model.modules()),
        "references_unchanged":all(digest(Path(b["path"]))==b["sha256"] and digest(Path(b["predictions_path"]))==b["predictions_sha256"] for b in bindings),
        "weights_file_unchanged":digest(args.policy/"weights.pt")==protocol["weights_sha256"],
        "implementation_unchanged":all(digest(v)==protocol["implementation_sha256"][k] for k,v in modules.items())}
    write_new(args.output,{"schema":"reacher_d2p_masked_packet_fidelity.v1","status":"completed" if all(gates.values()) else "fail","cuda_host":host,
        "gates":gates,"sealed_protocol":protocol,"protocol_sha256":digest(paths["protocol"]),"actual_predict_samples":count,"physical_branches":144,"rows":rows})
    print(json.dumps({"gates":gates,"actual_predict_samples":count}),flush=True)


if __name__=="__main__": main()
