"""Fresh source-disjoint paired score intervention, frozen official host CUDA.

Keep every root/task/program and old suffix-search contract. All actions and
both decisions are sealed before any future execution. Not a P1 replacement.
"""
from __future__ import annotations

import argparse
import json
import shutil
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
from experiments.spectral_reachability import reacher_d2p_root_metric_score as scoring
from experiments.spectral_reachability import reacher_d2p_fixed_warm_render as render
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability.audit_reacher_d2p_root_metric_score import OFFICIAL
from experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.run_reacher_d2p_masked_program_pilot import check_raw, physical_row
from experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation import bind_exclusions
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots
from experiments.spectral_reachability import reacher_d2p_goal_metric_runtime as goal_runtime
from experiments.spectral_reachability import reacher_d2p_goal_encoder_metric as goal_metric


def upgraded_bank(model, emb, history, target, *, task, mean, std, seed):
    """Original compiler, then identical fixed incumbent-preserving suffix CEM/GN."""
    original = compiler.compile_programs(model, emb, history, target, task=task, mean=mean, std=std, seed=seed)
    width = len(chart.active_columns(task))
    def full_terminal(flat):
        return chart.terminal(model, emb, history, flat.reshape(1, 3, width), task=task, mean=mean, std=std)[0]
    common = torch.tensor(original["common_model_residual"], device="cuda")
    bank = []
    for old in original["bank"]:
        raw = np.asarray(old["raw_actions"], np.float32)
        prefix = torch.tensor(raw[:5].ravel()[chart.active_columns(task)], device="cuda")
        initial = torch.tensor(raw[5:].ravel()[chart.active_columns(task, 2)], device="cuda")
        after, actions, prediction = compiler.after_prefix(model, emb, history, prefix.reshape(1, 1, width), task=task, mean=mean, std=std)
        if prediction[0].tolist() != old["prefix_vector"]:
            raise RuntimeError("original prefix recurrence changed")
        def terminal(flat):
            return chart.terminal(model, after, actions, flat.reshape(1, 2, width), task=task, mean=mean, std=std)[0]
        def batch(flat):
            n = len(flat)
            return chart.terminal(model, after.expand(n, -1, -1), actions.expand(n, -1, -1), flat.reshape(n, 2, width), task=task, mean=mean, std=std)
        response = torch.autograd.functional.jacobian(terminal, torch.zeros(2*width, device="cuda", requires_grad=True), vectorize=True).detach()
        gain = qp.certified_response_gain(response.double(), common.double(), damping=.01)
        if not np.isclose(gain["bounded_quadratic_gain"], old["common_response_gain"]["bounded_quadratic_gain"], rtol=1e-10, atol=1e-10):
            raise RuntimeError("original common response changed")
        cem, cem_receipt = search.search(batch, initial, target, seed=seed+100000+old["index"])
        refined, gn_receipt = compiler.fit(terminal, cem, target, iterations=4)
        full = torch.cat((prefix, refined)).reshape(1, 3, width)
        with torch.no_grad():
            split, direct = terminal(refined), full_terminal(full.ravel())
        if not torch.equal(split, direct):
            raise RuntimeError("same-action split/full recurrence mismatch")
        cost = float((direct-target).square().mean())
        if cost > old["terminal_cost"]:
            raise RuntimeError("fixed suffix search lost original incumbent")
        bank.append({"index": old["index"], "prefix_cost": old["prefix_cost"],
            "terminal_cost": cost, "common_response_gain": gain, "response": response.tolist(),
            "raw_actions": chart.expand_commands(full, task)[0].reshape(15, 2).tolist(),
            "prefix_vector": old["prefix_vector"], "terminal_vector": direct.tolist(),
            "full_split_exact": True, "cem_receipt": cem_receipt, "gn_receipt": gn_receipt})
    projected = [{k:b[k] for k in ("index", "prefix_cost", "terminal_cost", "common_response_gain", "response")} for b in bank]
    scores = scoring.rescore(projected, common.tolist(), device="cuda")
    return {"bank": bank, "common_residual": common.tolist(), "scores": scores,
        "controls": original["controls"], "original_compiler_decision": original["decision"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202653000)
    parser.add_argument("--goal-encoder-metric", action="store_true",
        help="Prospective additional goal-only metric; retains original root-metric baseline and original suffix optimization.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {k:args.output.with_suffix(s) for k,s in (("protocol", ".protocol.json"), ("predictions", ".predictions.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in (args.output, *paths.values())):
        raise FileExistsError("new immutable confirmation paths required")
    seeds = list(range(args.seed, args.seed+8))
    excluded, bindings = bind_exclusions([args.disjoint_reference], seeds)
    prior = json.loads(args.disjoint_reference.read_text())["sealed_protocol"]
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if (len(excluded) < 224 or prior["weights_sha256"] != OFFICIAL or digest(args.policy/"weights.pt") != OFFICIAL
            or digest(args.adapter_artifact) != prior["adapter_sha256"] or source_sha != prior["source_sha256_receipt"]):
        raise ValueError("official unchanged model/source/stats and224 excluded episodes required")
    roots = planned_source_roots(dataset, seeds, excluded)
    if args.goal_encoder_metric and (len(excluded) < 232 or args.seed < 202654000):
        raise ValueError("prospective goal metric requires232 excluded episodes and a new predeclared seed block")
    modules = {"runner":Path(__file__), **{k:Path(m.__file__) for k,m in
        (("shared",shared), ("execution",execution), ("goals",goals), ("chart",chart), ("compiler",compiler),
         ("search",search), ("scoring",scoring), ("render",render), ("p0",p0), ("qp",qp))}}
    if args.goal_encoder_metric:
        modules.update(goal_runtime=Path(goal_runtime.__file__), goal_metric=Path(goal_metric.__file__),
            model_encode=goal_runtime.declared_encoder_source(args.policy))
    for name, function in (("source_context",shared._source_state_context), ("loader",shared.load_compatible_pretrained),
            ("feasible_goal",goals.feasible_goal_delta), ("source_planner",planned_source_roots),
            ("exclusions",bind_exclusions), ("physical_writer",physical_row)):
        modules[name] = Path(function.__code__.co_filename)
    protocol = {"root_seeds": seeds, "source_roots": roots, "excluded_source_episodes":sorted(excluded),
        "exclusion_bindings": bindings, "weights_sha256": OFFICIAL, "adapter_sha256":digest(args.adapter_artifact),
        "source_sha256_receipt":source_sha, "goal_contract":"joint-feasible-v2-soft-source", "target_delta":[.65,-.65],
        "tasks":{k:list(v) for k,v in chart.TASKS.items()}, "policy":scoring.POLICY, "ridge":scoring.RIDGE,
        "render_policy":render.POLICY, "all17_programs_retained":True,
        "suffix_search":{"cem_samples":128,"cem_steps":3,"elite":16,"seed_law":"root+100000+index","gn_iterations":4},
        "controls":["unchanged_euclidean_selector","zero","greedy","flat_gn","flat_cem_gn_incumbent"],
        "stage_terminal_thresholds_unchanged":True, "all_forecasts_sealed_before_future_execution":True,
        "physical_branches":456, "expected_certified_metric_qps":408,
        "expected_predict_samples_per_task":2482+17*(1+2+386*2+4*8*2+2+3),
        "implementation_sha256":{k:digest(p) for k,p in modules.items()}, "P1_admission":False,
        "scope":"Fresh paired score intervention only, no posttraining, no root/task/parameter selection, no compute-matched efficacy or P1 admission."}
    if args.goal_encoder_metric:
        protocol["checkpoint_config_sha256"] = digest(args.policy/"config.json")
        protocol["goal_metric_intervention"] = {"policy": goal_metric.POLICY, "ridge": .01,
            "modes": list(goal_runtime.MODES), "expected_input_vjp_rows":1536,"expected_encoder_forwards":104,
            "gradient_checks":{"directions":3,"steps":[.001,.003],"maximum_relative_error":.15,"minimum_cosine":.98},
            "expected_extra_certified_qps":408,"same_goal_metric_all_tasks":True,
            "reoptimized_suffix":False,"physical_future_labels_read":False,"P1_admission":False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for k,p in modules.items(): shutil.copyfile(p, paths["implementation"]/(k+".py"))
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    normalizer = shared.ActionNormalizer(mean, std)
    processor = shared._Preprocessor(normalizer)
    mean_t, std_t = (torch.tensor(v, device="cuda") for v in (normalizer.block_mean, normalizer.block_std))
    model = shared.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if args.goal_encoder_metric:
        encode_source = Path(model.encode.__func__.__code__.co_filename)
        if digest(encode_source) != protocol["implementation_sha256"]["model_encode"]:
            raise RuntimeError("loaded encoder differs from presealed official implementation")
        # The runtime loader binds the loaded implementation; seal its actual
        # encoder source separately before producing any model decisions.
        write_new(args.output.with_suffix(".encoder-source.json"),
            {"source_path":str(encode_source),"sha256":digest(encode_source),"protocol_sha256":digest(paths["protocol"])})
    parameters = shared._parameter_hash(model)
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    original_predict, samples = model.predict, 0
    def counted(emb, act):
        nonlocal samples
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return original_predict(emb, act)
    model.predict = counted
    env = goals.goal_wrapper(protocol["goal_contract"])(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65,-.65))
    renderer = render.FixedWarmRenderer(env)
    def prepare(ctx):
        return p0._prepare(processor, ctx["frames"], ctx["root_frame"], ctx["goal"], np.asarray(ctx["history_action_blocks"], np.float32), torch.device("cuda"))
    pending, predictions, rows = [], [], []
    try:
        for root in roots:
            seed = root["seed"]
            ctx = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(ctx[k] != root[k] for k in ("source_episode", "source_start_row")):
                raise RuntimeError("admitted source identity changed")
            ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
            pixels, goal, history = prepare(ctx)
            emb, target = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            if args.goal_encoder_metric:
                operator, goal_receipt = goal_runtime.measure_goal(model, goal, target.tolist(), seed=seed)
                print(json.dumps({"seed":seed,"phase":"goal_metric","encoder_forwards":goal_receipt["encoder_forwards"],
                    "input_vjp_rows":goal_receipt["input_vjp_rows"]}),flush=True)
            tasks = {}
            for task in chart.TASKS:
                before = samples
                tasks[task] = upgraded_bank(model, emb, history, target, task=task, mean=mean_t, std=std_t, seed=seed)
                tasks[task]["actual_predict_samples"] = samples-before
                if args.goal_encoder_metric:
                    tasks[task]["goal_scores"] = goal_runtime.score_bank(operator,tasks[task],target)
                print(json.dumps({"seed":seed,"task":task,"phase":"predictions","samples":samples-before,
                    "metric_decision":tasks[task]["scores"]["decision"],"old_decision":tasks[task]["scores"]["original_decision"]}), flush=True)
            record = {**root,"goal_vector":target.tolist(),"root_rgb_sha256":ctx["root_rgb_sha256"],
                "goal_rgb_sha256":ctx["goal_rgb_sha256"],"initial_snapshot":ctx["execution_snapshot"],
                "goal_admission":goals.capture_goal_admission(env, protocol["goal_contract"]),"tasks":tasks}
            if args.goal_encoder_metric:
                record["goal_metric"] = goal_receipt
            predictions.append(record)
            pending.append((ctx,record))
        write_new(paths["predictions"], {"protocol_sha256":digest(paths["protocol"]),"rows":predictions})
        prediction_sha = digest(paths["predictions"])
        print(json.dumps({"all_predictions_presealed":True,"predictions_sha256":prediction_sha}), flush=True)
        for ctx, record in pending:
            tasks = {}
            for task,plan in record["tasks"].items():
                measured = {}
                branches = [("bank_"+str(b["index"]),b) for b in plan["bank"]]+list(plan["controls"].items())
                for name,branch in branches:
                    raw = np.asarray(branch["raw_actions"], np.float32)
                    if not check_raw(raw, task): raise RuntimeError("literal raw action mask/bounds changed")
                    frame,goal = execution.restore_execution_snapshot(env, seed=record["seed"], snapshot=ctx["execution_snapshot"], root_identity="sealed_observation_exact_state")
                    p0._verify_execution_identity(frame, goal, ctx["root_frame"], ctx["goal"])
                    current,packets = ctx,[]
                    for j in range(3):
                        current = execution.advance_executed_context(env, context=current, raw_action_block=raw[j*5:(j+1)*5], action_block=5, verified_root_frame=current["root_frame"])
                        pixels,_,history = prepare(current)
                        with torch.no_grad(): observed = p0._latent_context(model, pixels, history)[0,-1]
                        packets.append({"packet":j+1,"observed_vector":observed.tolist(),
                            "qpos":np.asarray(env.unwrapped.env.physics.data.qpos,np.float64)[:2].tolist(),
                            "root_rgb_sha256":current["root_rgb_sha256"],"action_sha256":current["advanced_action_sha256"]})
                    measured[name] = {"raw_actions":raw.tolist(),"packets":packets}
                tasks[task] = measured
                print(json.dumps({"seed":record["seed"],"task":task,"phase":"physical","branches":len(measured)}),flush=True)
            row = physical_row(record,tasks)
            write_new(paths["roots"]/f"root_{record['seed']}.json",row)
            rows.append(row)
    finally:
        model.predict = original_predict
        renderer.restore()
        env.close()
    stats = renderer.stats
    gates = {"cuda_only":True,"all8_roots":[r["seed"] for r in rows]==seeds,
        "source_disjoint":len({r["source_episode"] for r in rows})==8 and not ({r["source_episode"] for r in rows}&excluded),
        "all456_physical_branches":len(rows)==8 and all(set(r["tasks"])==set(chart.TASKS) and all(len(t)==19 for t in r["tasks"].values()) for r in rows),
        "all408_certified_metric_qps":sum(t["scores"]["certified_qps"] for r in predictions for t in r["tasks"].values())==408,
        "all_full_split_exact":all(b["full_split_exact"] for r in predictions for t in r["tasks"].values() for b in t["bank"]),
        "parameters_unchanged":shared._parameter_hash(model)==parameters,"buffers_unchanged":all(torch.equal(v,buffers[k]) for k,v in model.named_buffers()),
        "all_modules_eval":not any(m.training for m in model.modules()),
        "predict_sample_accounting":samples==sum(t["actual_predict_samples"] for r in predictions for t in r["tasks"].values()),
        "all_task_compute_receipts":all(t["actual_predict_samples"]==protocol["expected_predict_samples_per_task"] for r in predictions for t in r["tasks"].values()),
        "predictions_unchanged":digest(paths["predictions"])==prediction_sha,
        "weights_unchanged":digest(args.policy/"weights.pt")==OFFICIAL,
        "adapter_unchanged":digest(args.adapter_artifact)==protocol["adapter_sha256"],
        "reference_unchanged":all(digest(Path(b["path"]))==b["sha256"] for b in bindings),
        "implementation_unchanged":all(digest(p)==protocol["implementation_sha256"][k] for k,p in modules.items()),
        "fixed_render_receipts":stats["requests"]>0 and stats["raw_render_passes"]==stats["exact_state_checks"]==2*stats["requests"] and stats["dither_state_changes"]==0}
    if args.goal_encoder_metric:
        encoder_receipt = json.loads(args.output.with_suffix(".encoder-source.json").read_text())
        gates.update(all1536_input_vjps=sum(r["goal_metric"]["input_vjp_rows"] for r in predictions)==1536,
            all104_encoder_forwards=sum(r["goal_metric"]["encoder_forwards"] for r in predictions)==104,
            all408_goal_metric_qps=sum(len(t["goal_scores"]["consistent_Q_costs"]["bank"]) for r in predictions for t in r["tasks"].values())==408,
            all48_goal_decisions=all(set(t["goal_scores"])==set(goal_runtime.MODES) for r in predictions for t in r["tasks"].values()),
            model_encode_unchanged=digest(encode_source)==digest(paths["implementation"]/"model_encode.py")==encoder_receipt["sha256"],
            checkpoint_config_unchanged=digest(args.policy/"config.json")==protocol["checkpoint_config_sha256"])
    result = {"schema":"reacher_d2p_root_metric_confirmation.v1","status":"completed" if all(gates.values()) else "fail",
        "sealed_protocol":protocol,"protocol_sha256":digest(paths["protocol"]),"predictions_sha256":prediction_sha,
        "cuda_host":host,"gates":gates,"rows":rows,"render_stats":stats,"actual_predict_samples":samples,"P1_admission":False}
    write_new(args.output,result)
    print(json.dumps({"gates":gates,"actual_predict_samples":samples}),flush=True)
    if not all(gates.values()): raise RuntimeError("fresh metric diagnostic integrity failed")


if __name__ == "__main__":
    main()
