"""All-root, all-task same-action masked program audit, official frozen CUDA.

Seal every planned action and decision before measuring any future physical
branch. This is a mechanism/code pilot, not calibrated P1 or equal resources.
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
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new
from experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation import bind_exclusions
from experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic import planned_source_roots


def check_raw(raw, task):
    value = np.asarray(raw, np.float32)
    inactive = [j for j in range(2) if j not in chart.TASKS[task]]
    return (value.shape == (15, 2) and bool(np.isfinite(value).all()) and bool(np.max(np.abs(value)) <= 1)
        and np.count_nonzero(value[:, inactive]) == 0)


def physical_row(record, tasks):
    """Bind observations to their own sealed record, never an outer loop root."""
    return {**{k:record[k] for k in ("seed", "source_episode", "source_start_row")}, "tasks":tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202650000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--goal-contract",choices=("joint-feasible-v1","joint-feasible-v2-soft-source"),default="joint-feasible-v1")
    args = parser.parse_args()
    paths = {k: args.output.with_suffix(suffix) for k, suffix in (("protocol", ".protocol.json"), ("predictions", ".predictions.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(p.exists() for p in (args.output, *paths.values())):
        raise FileExistsError("new immutable program audit output/sidecars required")
    seeds = list(range(args.seed, args.seed+8))
    excluded, bindings = bind_exclusions([args.disjoint_reference], seeds)
    reference = json.loads(args.disjoint_reference.read_text())
    prior = reference["sealed_protocol"]
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if (digest(args.policy/"weights.pt") != prior["weights_sha256"] or digest(args.adapter_artifact) != prior["adapter_sha256"]
            or source_sha != prior["source_sha256_receipt"]):
        raise ValueError("unchanged original model/source/normalization required")
    roots = planned_source_roots(dataset, seeds, excluded)
    modules = {"auditor": Path(__file__), "compiler": Path(compiler.__file__), "chart": Path(chart.__file__), "p0": Path(p0.__file__),
        "context": Path(execution.__file__), "goal_contract": Path(goals.__file__), "qp": Path(qp.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename), "loader": Path(shared.load_compatible_pretrained.__code__.co_filename)}
    modules["feasible_goal"] = Path(goals.feasible_goal_delta.__code__.co_filename)
    protocol = {"root_seeds": seeds, "source_roots": roots, "excluded_source_episodes": sorted(excluded), "exclusion_bindings": bindings,
        "weights_sha256": digest(args.policy/"weights.pt"), "adapter_sha256": digest(args.adapter_artifact), "source_sha256_receipt": source_sha,
        "tasks": {k: list(v) for k,v in chart.TASKS.items()}, "prefix_packets": 1, "future_packets": 2,
        "spectral_bank": {"directions": 4, "amplitudes": [.25, .75], "signs": [-1, 1], "literal_zero_index": 0, "count": 17},
        "suffix_gn_iterations": 4, "flat_gn_iterations": 4, "damping": .01, "line_search": [1., .5, .25, .125, .0625, .03125],
        "flat_cem_with_gn_incumbent": {"samples": 128, "steps": 3, "elite": 16},
        "selector": "non_greedy_vs_literal_zero_and_terminal_better_than_greedy_and_certified_common_response_gain_above_zero_and_greedy",
        "non_greedy_margin": 1e-6, "terminal_margin": 1e-6, "target_delta": [.65, -.65], "goal_contract": args.goal_contract,
        "all_predictions_sealed_before_any_future_execution": True, "no_post_training": True, "compute_matched": False,
        "physical_branches": 456, "expected_predict_samples": 59568,
        "implementation_sha256": {k:digest(v) for k,v in modules.items()},
        "scope": "Fresh source-disjoint all-task same-action mechanism/code diagnostic. Not calibration/outcome, equal-resource efficacy, physical controllability theorem or P1 admission."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for k,path in modules.items():
        shutil.copyfile(path, paths["implementation"]/(k+".py"))
    print("HOST CUDA: exact masks, all 8 independent roots and all 17 spectral programs/task. Seal all decisions before outcomes; check split/full recurrence exactly; no P1 claim.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    normalizer = shared.ActionNormalizer(mean,std)
    processor = shared._Preprocessor(normalizer)
    mean_t,std_t = (torch.tensor(v,device="cuda") for v in (normalizer.block_mean,normalizer.block_std))
    model = shared.load_compatible_pretrained(args.policy,device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameters, buffers = shared._parameter_hash(model), {k:v.clone() for k,v in model.named_buffers()}
    predict, samples = model.predict, 0
    def counted(emb,act):
        nonlocal samples
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return predict(emb,act)
    model.predict = counted
    env = goals.goal_wrapper(args.goal_contract)(gym.make("swm/ReacherDMControl-v0",render_mode="rgb_array"),target_delta=(.65,-.65))
    def prepare(ctx):
        return p0._prepare(processor,ctx["frames"],ctx["root_frame"],ctx["goal"],np.asarray(ctx["history_action_blocks"],np.float32),torch.device("cuda"))
    predictions, pending, rows = [], [], []
    try:
        for root in roots:
            seed = root["seed"]
            ctx = shared._source_state_context(env,dataset=dataset,seed=seed)
            if any(ctx[k] != root[k] for k in ("source_episode","source_start_row")):
                raise RuntimeError("source metadata changed")
            ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
            pixels,goal,history = prepare(ctx)
            emb,target = p0._latent_context(model,pixels,history),p0._goal_latent(model,goal)
            tasks = {}
            for task in chart.TASKS:
                start_samples, started = samples,time.monotonic()
                tasks[task] = compiler.compile_programs(model,emb,history,target,task=task,mean=mean_t,std=std_t,seed=seed)
                torch.cuda.synchronize()
                tasks[task]["actual_predict_samples"] = samples-start_samples
                tasks[task]["planning_seconds"] = time.monotonic()-started
                for branch in tasks[task]["bank"]+list(tasks[task]["controls"].values()):
                    if not check_raw(branch["raw_actions"],task):
                        raise RuntimeError("planned action violates exact task mask")
                print(json.dumps({"seed":seed,"task":task,"phase":"predictions","samples":samples-start_samples,"decision":tasks[task]["decision"]}),flush=True)
            record = {**root,"goal_vector":target.tolist(),"root_rgb_sha256":ctx["root_rgb_sha256"],"goal_rgb_sha256":ctx["goal_rgb_sha256"],
                "goal_admission":goals.capture_goal_admission(env,args.goal_contract),"initial_snapshot":ctx["execution_snapshot"],"tasks":tasks}
            predictions.append(record)
            pending.append((ctx,record))
        write_new(paths["predictions"],{"protocol_sha256":digest(paths["protocol"]),"rows":predictions})
        prediction_sha = digest(paths["predictions"])
        print(json.dumps({"all_predictions_presealed":True,"predictions_sha256":prediction_sha}),flush=True)
        for ctx,record in pending:
            seed = record["seed"]
            tasks = {}
            for task,plan in record["tasks"].items():
                branches = [("bank_"+str(b["index"]),b) for b in plan["bank"]]+list(plan["controls"].items())
                measured = {}
                for name,branch in branches:
                    frame,goal = execution.restore_execution_snapshot(env,seed=seed,snapshot=ctx["execution_snapshot"],root_identity="sealed_observation_exact_state")
                    p0._verify_execution_identity(frame,goal,ctx["root_frame"],ctx["goal"])
                    raw = np.asarray(branch["raw_actions"],np.float32)
                    if not check_raw(raw,task):
                        raise RuntimeError("execution mask mismatch")
                    current,packets = ctx,[]
                    for j in range(3):
                        current = execution.advance_executed_context(env,context=current,raw_action_block=raw[j*5:(j+1)*5],action_block=5,verified_root_frame=current["root_frame"])
                        pixels,_,history = prepare(current)
                        with torch.no_grad():
                            observed = p0._latent_context(model,pixels,history)[0,-1]
                        packets.append({"packet":j+1,"observed_vector":observed.tolist(),"qpos":np.asarray(env.unwrapped.env.physics.data.qpos,np.float64)[:2].tolist(),
                            "root_rgb_sha256":current["root_rgb_sha256"],"action_sha256":current["advanced_action_sha256"]})
                    measured[name] = {"raw_actions":raw.tolist(),"packets":packets}
                tasks[task] = measured
                print(json.dumps({"seed":seed,"task":task,"phase":"physical","all19_branches":True}),flush=True)
            row = physical_row(record,tasks)
            write_new(paths["roots"]/f"root_{seed}.json",row)
            rows.append(row)
    finally:
        model.predict = predict
        env.close()
    gates = {"cuda_only":True,"all8_roots":[r["seed"] for r in rows]==seeds,
        "source_disjoint":len({r["source_episode"] for r in rows})==8 and not ({r["source_episode"] for r in rows}&excluded),
        "all456_physical_branches":len(rows)==8 and all(set(r["tasks"])==set(chart.TASKS) and all(len(b)==19 for b in r["tasks"].values()) for r in rows),
        "all_full_split_exact":all(b["full_split_exact"] is True for r in predictions for p in r["tasks"].values() for b in p["bank"]),
        "parameters_unchanged":shared._parameter_hash(model)==parameters,"buffers_unchanged":all(torch.equal(v,buffers[k]) for k,v in model.named_buffers()),
        "all_modules_eval":not any(m.training for m in model.modules()),"predict_sample_accounting":samples==59568,
        "all_task_compute_receipts":all(p["actual_predict_samples"]==2482 for r in predictions for p in r["tasks"].values()),
        "predictions_unchanged":digest(paths["predictions"])==prediction_sha,"reference_unchanged":digest(args.disjoint_reference)==bindings[0]["sha256"],
        "weights_file_unchanged":digest(args.policy/"weights.pt")==protocol["weights_sha256"],
        "implementation_unchanged":all(digest(v)==protocol["implementation_sha256"][k] for k,v in modules.items())}
    write_new(args.output,{"schema":"reacher_d2p_masked_program_pilot.v1","status":"completed" if all(gates.values()) else "fail","cuda_host":host,
        "sealed_protocol":protocol,"protocol_sha256":digest(paths["protocol"]),"predictions_sha256":prediction_sha,"gates":gates,
        "actual_model_predict_samples":samples,"physical_branches":456,"rows":rows})
    print(json.dumps({"gates":gates,"actual_model_predict_samples":samples}),flush=True)


if __name__ == "__main__":
    main()
