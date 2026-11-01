"""Same-bank zero-tangent versus nonlinear-validated reachability audit.

All16 known roots/all3 exact actuator tasks/all17 prefixes. No posttraining,
outcome-selected coefficient, new P1 admission, or global nonlinear QP claim.
All model-only decisions are sealed before all 240 physical branches.
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
from experiments.spectral_reachability import reacher_d2p_joint_distance as metric
from experiments.spectral_reachability import reacher_d2p_masked_action_chart as chart
from experiments.spectral_reachability import reacher_d2p_masked_programs as compiler
from experiments.spectral_reachability import reacher_d2p_recentered_reach as reach
from experiments.spectral_reachability import reacher_d2p_proposal_native_selection as proposal
from experiments.spectral_reachability import bounded_response_qp as qp
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.analyze_reacher_d2p_masked_suffix_search import verify_files
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new

MODES = ("legacy_zero_tangent", "feasible_incumbent", "recentered_feasible", "recentered_greedy", "recentered_zero")


def confirmation_cohorts(reference, entries):
    """Identity-only admission; never inspect outcome values to choose roots."""
    if (reference.get("status") != "completed" or not reference.get("gates")
            or any(v is not True for v in reference["gates"].values())):
        raise ValueError("completed integrity-passing frozen reference required")
    previous = reference["sealed_protocol"]
    if previous["contract"] != reach.CONTRACT or previous["implementation_sha256"]["recenter"] != digest(Path(reach.__file__)):
        raise ValueError("confirmation must freeze the exact original runtime")
    known = {r["seed"] for r in reference["rows"]}
    old_episodes = {r["source_episode"] for r in reference["rows"]}
    roots = [r for _, r in entries]
    fresh = [r for r in roots if r["seed"] not in known]
    aligned = [r for r in roots if r["seed"] in known]
    if (len(fresh) != 8 or len(aligned) != 8 or len({r["seed"] for r in roots}) != 16
            or len({r["source_episode"] for r in roots}) != 16
            or {r["source_episode"] for r in fresh} & old_episodes):
        raise ValueError("eight fresh source-disjoint roots and eight alignment controls required")
    prior = {r["seed"]: r for r in reference["rows"]}
    if any(any(r.get(key) != prior[r["seed"]].get(key) for key in ("source_episode", "source_start_row")) for r in aligned):
        raise ValueError("alignment seeds must retain exact source episode and start row")
    return {"confirmation": sorted(r["seed"] for r in fresh),
        "alignment": sorted(r["seed"] for r in aligned)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmation-reference", type=Path)
    args = parser.parse_args()
    paths = {k: args.output.with_suffix(s) for k, s in (("protocol", ".protocol.json"),
        ("predictions", ".predictions.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if len(args.reference) != 2 or any(p.exists() for p in (args.output, *paths.values())):
        raise ValueError("two original panels and new immutable output paths required")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    references, entries, measured, bindings = [], [], {}, {}
    for path in args.reference:
        verify_files(path)  # independent complete original QP/physical/seal validation
        report = json.loads(path.read_text())
        protocol = report["sealed_protocol"]
        pp = path.with_suffix(".predictions.json")
        pred = json.loads(pp.read_text())
        if (digest(args.policy/"weights.pt") != protocol["weights_sha256"]
                or digest(args.adapter_artifact) != protocol["adapter_sha256"]
                or source_sha != protocol["source_sha256_receipt"]):
            raise ValueError("unchanged official checkpoint/source/statistics required")
        references.append({"path": str(path.resolve()), "sha256": digest(path),
            "predictions_path": str(pp.resolve()), "predictions_sha256": digest(pp)})
        for p in (path, pp, path.with_suffix(".protocol.json")):
            bindings[str(p.resolve())] = digest(p)
        measured.update({r["seed"]: r for r in report["rows"]})
        entries.extend((protocol.get("goal_contract", "joint-feasible-v1"), row) for row in pred["rows"])
    entries.sort(key=lambda v: v[1]["seed"])
    seeds = [row["seed"] for _, row in entries]
    episodes = [row["source_episode"] for _, row in entries]
    if len(seeds) != 16 or len(set(seeds)) != 16 or len(set(episodes)) != 16 or sorted(measured) != seeds:
        raise ValueError("all16 distinct source/root identities required")
    confirmation = None
    if args.confirmation_reference:
        frozen = json.loads(args.confirmation_reference.read_text())
        cohorts = confirmation_cohorts(frozen, entries)
        previous = frozen["sealed_protocol"]
        if any(previous[k] != value for k, value in (("weights_sha256", digest(args.policy/"weights.pt")),
                ("adapter_sha256", digest(args.adapter_artifact)), ("source_sha256_receipt", source_sha))):
            raise ValueError("confirmation checkpoint/source/statistics changed")
        for path in args.reference:
            candidate = json.loads(path.read_text())["sealed_protocol"]
            if set(candidate["root_seeds"]) == set(cohorts["confirmation"]):
                baseline = json.loads(Path(candidate["reference_path"]).read_text())["sealed_protocol"]
                if len(set(baseline.get("excluded_source_episodes", []))) < 216:
                    raise ValueError("confirmation requires at least216 excluded historical episodes")
                break
        else:
            raise ValueError("fresh cohort must be a complete independently sealed panel")
        bindings[str(args.confirmation_reference.resolve())] = digest(args.confirmation_reference)
        confirmation = {"path": str(args.confirmation_reference.resolve()),
            "sha256": digest(args.confirmation_reference), "cohorts": cohorts,
            "runtime_sha256": digest(Path(reach.__file__)), "independent_efficacy_claimed": False}
    modules = {"auditor": Path(__file__), "recenter": Path(reach.__file__), "chart": Path(chart.__file__),
        "compiler": Path(compiler.__file__), "qp": Path(qp.__file__), "proposal": Path(proposal.__file__),
        "context": Path(execution.__file__), "goals": Path(goals.__file__), "p0": Path(p0.__file__),
        "goal_wrapper": Path(goals.feasible_goal_delta.__code__.co_filename), "metric": Path(metric.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename),
        "loader": Path(shared.load_compatible_pretrained.__code__.co_filename)}
    protocol = {"root_seeds": seeds, "source_episodes": episodes, "references": references,
        "input_bindings": bindings, "weights_sha256": digest(args.policy/"weights.pt"),
        "adapter_sha256": digest(args.adapter_artifact), "source_sha256_receipt": source_sha,
        "contract": reach.CONTRACT, "modes": list(MODES), "tasks": list(chart.TASKS), "bank_count": 17,
        "model_only_decisions_sealed_before_physical_execution": True, "prefixes_and_goals_unchanged": True,
        "compute_matched": False, "P1_admission": False, "joint_metric": metric.CONTRACT,
        "expected_predict_samples": 35088, "expected_jacobians": 2448, "expected_vjp_rows": 470016,
        "physical_branches": 240, "expected_exact_old_replays": 96,
        "scope": "Known16-root local-surrogate/solver audit, not independent efficacy, calibrated P1, C2 or a theory rejection.",
        "implementation_sha256": {k: digest(p) for k, p in modules.items()}}
    if confirmation:
        protocol["confirmation"] = confirmation
        protocol["scope"] = "Fixed-runtime eight-fresh-root confirmation plus eight alignment controls; report cohorts separately; no P1 admission or resource-matched claim."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(paths["protocol"], protocol)
    paths["roots"].mkdir()
    paths["implementation"].mkdir()
    for name, path in modules.items(): shutil.copyfile(path, paths["implementation"]/(name+".py"))
    print("HOST CUDA: distinguish a certified zero-tangent QP from an actual nonlinear feasible suffix; fixed2 relinearizations with unchanged absolute box/regularizer. All16 roots/all816 programs, all decisions sealed before 240 actual branches. No P1 admission.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    normalizer = shared.ActionNormalizer(mean, std)
    processor = shared._Preprocessor(normalizer)
    mean_t, std_t = (torch.tensor(v, device="cuda") for v in (normalizer.block_mean, normalizer.block_std))
    model = shared.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    params, buffers = shared._parameter_hash(model), {k: v.clone() for k, v in model.named_buffers()}
    original, count = model.predict, 0
    def counted(emb, actions):
        nonlocal count
        if emb.device.type != "cuda" or actions.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        count += len(emb)
        return original(emb, actions)
    model.predict = counted
    def prepare(ctx):
        return p0._prepare(processor, ctx["frames"], ctx["root_frame"], ctx["goal"],
            np.asarray(ctx["history_action_blocks"], np.float32), torch.device("cuda"))
    forecasts, pending, rows, old_replays = [], [], [], []
    try:
        for contract, old in entries:
            seed = old["seed"]
            env = goals.goal_wrapper(contract)(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
            try:
                ctx = shared._source_state_context(env, dataset=dataset, seed=seed)
                ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
                if (any(ctx[k] != old[k] for k in ("source_episode", "source_start_row", "root_rgb_sha256", "goal_rgb_sha256"))
                        or ctx["execution_snapshot"] != old["initial_snapshot"]):
                    raise RuntimeError("original full source/root/goal snapshot changed")
                pixels, goal_pixels, history = prepare(ctx)
                emb, target = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal_pixels)
                if target.tolist() != old["goal_vector"]: raise RuntimeError("original goal representation changed")
                tasks = {}
                for task in chart.TASKS:
                    width, bank = len(chart.active_columns(task)), []
                    for previous in old["tasks"][task]["bank"]:
                        before = count
                        old_variant = previous["variants"]["cem_then_gn4"]
                        raw = np.asarray(old_variant["raw_actions"], np.float32)
                        prefix = torch.tensor(raw[:5].ravel()[chart.active_columns(task)], device="cuda")
                        initial = torch.tensor(raw[5:].ravel()[chart.active_columns(task, 2)], device="cuda")
                        after, actions, prefix_value = compiler.after_prefix(model, emb, history,
                            prefix.reshape(1, 1, width), task=task, mean=mean_t, std=std_t)
                        if prefix_value[0].tolist() != previous["prefix_vector"]:
                            raise RuntimeError("original prefix forecast changed")
                        def terminal(x):
                            return chart.terminal(model, after, actions, x.reshape(1, 2, width), task=task, mean=mean_t, std=std_t)[0]
                        zero = torch.zeros_like(initial)
                        with torch.no_grad(): zero_value = terminal(zero)
                        response = torch.autograd.functional.jacobian(terminal, zero.requires_grad_(True), vectorize=True).detach()
                        if zero_value.tolist() != previous["zero_suffix_vector"] or response.tolist() != previous["response"]:
                            raise RuntimeError("original zero-tangent full forecast/response changed")
                        old_gain = qp.certified_response_gain(response.double(), (target-zero_value).double(), damping=.01)
                        if old_gain != previous["own_response_gain"]:
                            raise RuntimeError("original zero-tangent QP receipt changed")
                        command, value, receipt = reach.refine(terminal, initial, target)
                        if receipt["initial_vector"] != old_variant["terminal_vector"]:
                            raise RuntimeError("original complete optimized suffix forecast changed")
                        full = torch.cat((prefix, command)).reshape(1, 3, width)
                        old_full = torch.cat((prefix, initial)).reshape(1, 3, width)
                        with torch.no_grad():
                            direct = chart.terminal(model, emb, history, full, task=task, mean=mean_t, std=std_t)[0]
                            original_direct = chart.terminal(model, emb, history, old_full, task=task, mean=mean_t, std=std_t)[0]
                        if direct.tolist() != value.tolist() or original_direct.tolist() != old_variant["terminal_vector"] or count-before != 43:
                            raise RuntimeError("full/split singleton or exact compute accounting changed")
                        bank.append({"index": previous["index"], "prefix_cost": previous["prefix_cost"],
                            "prefix_vector": previous["prefix_vector"], "zero_tangent_objective": old_gain["bounded_quadratic_objective"],
                            "initial_objective": receipt["initial_objective"], "final_objective": receipt["final_objective"],
                            "initial_raw_actions": raw.tolist(), "raw_actions": chart.expand_commands(full, task)[0].reshape(15, 2).tolist(),
                            "terminal_vector": value.tolist(), "receipt": receipt, "full_split_exact": True,
                            "actual_predict_samples": count-before})
                    greedy = min(range(17), key=lambda i: (bank[i]["prefix_cost"], i))
                    legacy = proposal.select(old["tasks"][task]["bank"], "cem_then_gn4")
                    decisions = {"legacy_zero_tangent": legacy["native_index"],
                        "feasible_incumbent": reach.select(bank, "initial_objective")["native_index"],
                        "recentered_feasible": reach.select(bank, "final_objective")["native_index"],
                        "recentered_greedy": greedy, "recentered_zero": 0}
                    tasks[task] = {"bank": bank, "decisions": decisions}
                forecast = {**{k: old[k] for k in ("seed", "source_episode", "source_start_row", "goal_vector", "goal_admission",
                    "root_rgb_sha256", "goal_rgb_sha256", "initial_snapshot")}, "goal_contract": contract, "tasks": tasks}
                forecasts.append(forecast)
                pending.append((ctx, forecast))
                print(json.dumps({"seed": seed, "phase": "model_only", "decisions": {t: v["decisions"] for t, v in tasks.items()}}), flush=True)
            finally: env.close()
        write_new(paths["predictions"], {"protocol_sha256": digest(paths["protocol"]), "rows": forecasts})
        prediction_sha = digest(paths["predictions"])
        for ctx, forecast in pending:
            seed, tasks = forecast["seed"], {}
            env = goals.goal_wrapper(forecast["goal_contract"])(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
            try:
                for task, plans in forecast["tasks"].items():
                    modes = {}
                    for mode in MODES:
                        index = plans["decisions"][mode]
                        branch = plans["bank"][index]
                        raw = np.asarray(branch["initial_raw_actions"] if mode in MODES[:2] else branch["raw_actions"], np.float32)
                        frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed,
                            snapshot=ctx["execution_snapshot"], root_identity="sealed_observation_exact_state")
                        p0._verify_execution_identity(frame, goal_rgb, ctx["root_frame"], ctx["goal"])
                        current, packets = ctx, []
                        for j in range(3):
                            current = execution.advance_executed_context(env, context=current, raw_action_block=raw[j*5:(j+1)*5],
                                action_block=5, verified_root_frame=current["root_frame"])
                            pixels, _, hist = prepare(current)
                            with torch.no_grad(): actual = p0._latent_context(model, pixels, hist)[0, -1]
                            packets.append({"packet": j+1, "observed_vector": actual.tolist(),
                                "qpos": np.asarray(env.unwrapped.env.physics.data.qpos, np.float64)[:2].tolist(),
                                "root_rgb_sha256": current["root_rgb_sha256"], "action_sha256": current["advanced_action_sha256"]})
                        physical = {"raw_actions": raw.tolist(), "packets": packets}
                        if mode in MODES[:2]:
                            expected = measured[seed]["tasks"][task]["bank_"+str(index)]["cem_then_gn4"]
                            if physical != expected: raise RuntimeError("complete original selected physical branch no longer replays exactly")
                            old_replays.append(True)
                        goal = np.asarray(forecast["goal_vector"], np.float64)
                        modes[mode] = {"bank_index": index, **physical,
                            "final_rgb_cost": float(np.mean((np.asarray(packets[-1]["observed_vector"])-goal)**2)),
                            "final_qpos_distance": metric.distance(np.asarray(packets[-1]["qpos"]),
                                np.asarray(forecast["goal_admission"]["feasible_target_qpos"]), forecast["goal_admission"]["limited_joints"])}
                    tasks[task] = modes
                row = {"seed": seed, "source_episode": forecast["source_episode"], "source_start_row": forecast["source_start_row"], "tasks": tasks}
                write_new(paths["roots"]/f"root_{seed}.json", row)
                rows.append(row)
                print(json.dumps({"seed": seed, "phase": "physical", "all15_branches": True}), flush=True)
            finally: env.close()
    finally: model.predict = original
    gates = {"cuda_only": True, "all16_roots": [r["seed"] for r in rows] == seeds,
        "all816_complete_model_programs": all(set(r["tasks"]) == set(chart.TASKS) and all(len(t["bank"]) == 17 for t in r["tasks"].values()) for r in forecasts),
        "all240_physical_branches": len(rows) == 16 and all(set(r["tasks"]) == set(chart.TASKS) and all(set(t) == set(MODES) for t in r["tasks"].values()) for r in rows),
        "all96_original_full_physical_replays_exact": len(old_replays) == 96 and all(old_replays),
        "all_full_split_singletons_exact": all(b["full_split_exact"] is True for r in forecasts for t in r["tasks"].values() for b in t["bank"]),
        "all_nonlinear_incumbents_preserved": all(b["final_objective"] <= b["initial_objective"] for r in forecasts for t in r["tasks"].values() for b in t["bank"]),
        "predict_accounting": count == protocol["expected_predict_samples"], "parameters_unchanged": shared._parameter_hash(model) == params,
        "buffers_unchanged": all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()), "all_modules_eval": not any(m.training for m in model.modules()),
        "prediction_seal_unchanged": digest(paths["predictions"]) == prediction_sha,
        "references_unchanged": all(digest(Path(p)) == sha for p, sha in bindings.items()),
        "weights_unchanged": digest(args.policy/"weights.pt") == protocol["weights_sha256"],
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    write_new(args.output, {"schema": "reacher_d2p_recentered_reach_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "sealed_protocol": protocol, "protocol_sha256": digest(paths["protocol"]), "predictions_sha256": prediction_sha,
        "cuda_host": host, "gates": gates, "actual_predict_samples": count, "physical_branches": 240, "P1_admission": False, "rows": rows})
    print(json.dumps({"gates": gates, "actual_predict_samples": count, "P1_admission": False}), flush=True)


if __name__ == "__main__": main()
