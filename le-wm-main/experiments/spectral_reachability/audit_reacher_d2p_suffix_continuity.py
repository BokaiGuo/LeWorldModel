"""Known-root causal audit of imagined-tail discard versus RGB replanning.

Replay ALL words on ALL reference roots. Seal reconstructed imagined tails
before executing any prefix. At the exact observed successor compare cold,
carried, and warm-refined tails; choose cold/warm using model cost only.
Physical labels are evaluator-only. This is not fresh P1 admission.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_action_words as words
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest


def identity_join(reference, selection):
    if (reference.get("schema") != "reacher_d2p_model_native_action_words.v1"
            or reference.get("status") != "completed" or not reference.get("gates")
            or not all(value is True for value in reference["gates"].values())):
        raise ValueError("complete integrity-passing action-word reference required")
    seeds = reference["sealed_protocol"]["root_seeds"]
    rows, choices = reference["rows"], selection["rows"]
    if (not seeds or len(seeds) != len(set(seeds))
            or any(len(data) != len(seeds) or {row["seed"] for row in data} != set(seeds) for data in (rows, choices))):
        raise ValueError("unique complete root identities required")
    by_root = {row["seed"]: row for row in choices}
    joined = []
    for row in sorted(rows, key=lambda item: item["seed"]):
        choice = by_root[row["seed"]]
        if any(row[key] != choice[key] for key in ("source_episode", "source_start_row")):
            raise ValueError("source root identity mismatch")
        if len(choice["word_params"]) != 17 or set(row["word_results"]) != {f"word_{i}" for i in range(17)}:
            raise ValueError("all17 reference words required")
        joined.append((row, choice))
    return joined


def choose_model_incumbent(costs):
    if set(costs) != {"cold", "warm"} or any(not math.isfinite(float(value)) for value in costs.values()):
        raise ValueError("finite cold/warm model costs required")
    # Retain historical cold action on an exact tie. No outcome argument.
    return min(("cold", "warm"), key=lambda name: (costs[name], name != "cold"))


def write_new(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    selection_path = args.reference.with_suffix(".selection.json")
    selection = json.loads(selection_path.read_text())
    joined = identity_join(reference, selection)
    ref = reference["sealed_protocol"]
    if digest(selection_path) != reference["selection_sha256"]:
        raise ValueError("reference selection changed")
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("reference weights or action normalization changed")
    p0, p1, solver, rescue = words.p0, words.p1, words.solver, words.rescue
    modules = {"auditor": Path(__file__), "words": Path(words.__file__), "p0": Path(p0.__file__),
        "prefix": Path(p1.__file__), "solver": Path(solver.__file__), "rescue": Path(rescue.__file__),
        "context": Path(execution.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename),
        "source_support": Path(words.__file__).with_name("reacher_d2p_packet_support.py"),
        "base_selection": Path(words.refinement.__file__), "stage_admission_dependency": Path(p0.stage_admission.__file__)}
    for key, path in modules.items():
        if key not in ("auditor", "solver") and digest(path) != ref["implementation_sha256"][key]:
            raise ValueError("historical dependency changed: " + key)
    historical_solver = args.reference.with_suffix(".implementation") / "solver.py"
    if digest(historical_solver) != ref["implementation_sha256"]["solver"]:
        raise ValueError("immutable historical solver missing or changed")
    output = args.output.resolve()
    sidecars = {name: output.with_suffix(suffix) for name, suffix in
        (("protocol", ".protocol.json"), ("selection", ".selection.json"), ("roots", ".roots"), ("implementation", ".implementation"))}
    if any(path.exists() for path in [output, *sidecars.values()]):
        raise FileExistsError("new immutable outputs required")
    protocol = {"reference_path": str(args.reference.resolve()), "reference_sha256": digest(args.reference),
        "reference_selection_sha256": digest(selection_path), "root_seeds": ref["root_seeds"],
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": ref["source_sha256"],
        "historical_solver_sha256": digest(historical_solver), "target_delta": ref["target_delta"],
        "conditions": ["cold", "carry", "warm"], "iterations": 8, "damping": .01,
        "incumbent_rule": "minimum observed-RGB model cost of cold/warm, cold on exact ties",
        "all_prefix_choices_and_imagined_tails_sealed_before_prefix_execution": True,
        "all_observed_suffix_actions_sealed_before_terminal_outcomes": True,
        "physical_horizon": 30, "physical_outcomes_used_for_selection": False,
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Known-root all-candidate causal intervention, not independent calibration, efficacy or P1 admission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(sidecars["protocol"], protocol)
    sidecars["roots"].mkdir()
    sidecars["implementation"].mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, sidecars["implementation"] / (key + ".py"))
    print("All8 fixed roots x all17 words. Isolate imagined/observed state, carried tail and cold/warm optimizer basins. Host CUDA only; no P1 override.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    from experiments.spectral_reachability import reacher_d2p_packet_support as support
    if source_sha != ref["source_sha256"] or support._sha256(dataset) != source_sha:
        raise ValueError("official source data changed")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    predict, samples = model.predict, 0

    def counted_predict(emb, act):
        nonlocal samples
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        samples += int(emb.shape[0])
        return predict(emb, act)

    model.predict = counted_predict
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"]))
    lo, hi = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lo, device=device), torch.as_tensor(hi, device=device)

    def prepare(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
            np.asarray(context["history_action_blocks"], np.float32), device)

    def raw(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]

    def compile_tail(goal_latent, emb, history, initial=None):
        start = samples
        params, receipt = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=2,
            action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,),
            accept_update=rescue.accept_with_fine_rescue, initial_params=initial)[8]
        if samples - start != receipt["model_predict_sample_evaluations"]:
            raise RuntimeError("suffix prediction accounting drift")
        return params, receipt

    pending, sealed, rows = [], [], []
    try:
        # Reconstruct and seal EVERY imagined tail before ANY candidate rollout.
        for old, choice in joined:
            seed = old["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if (any(context[key] != old[key] for key in ("source_episode", "source_start_row"))
                    or any(context[key] != choice[key] for key in ("root_rgb_sha256", "goal_rgb_sha256"))):
                raise RuntimeError("historical source/RGB identity changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            pixels, goal, history = prepare(context)
            goal_latent = p0._goal_latent(model, goal)
            bank = torch.tensor(choice["word_params"], device=device, dtype=history.dtype)
            tails, tail_records = [], []
            for index in range(17):
                emb, actions = p1._state_after_prefix(model, pixels, history, bank[index:index + 1], action_lower=lower, action_upper=upper)
                tail, receipt = compile_tail(goal_latent, emb, actions)
                original = choice["receipt"]["proposal_rows"][index]
                if receipt != original["suffix_receipt"]:
                    raise RuntimeError(f"historical imagined solver receipt not exact: {seed}/{index}")
                tails.append(tail)
                tail_records.append({"proposal_index": index, "params": tail.tolist(), "receipt": receipt,
                    "imagined_terminal_cost": receipt["trace"][-1]["cost_after"]})
            sealed.append({"seed": seed, "decision": old["decision"], "tails": tail_records})
            pending.append((old, choice, context, bank, tails))
            print(json.dumps({"seed": seed, "imagined_tails_exact": len(tails)}), flush=True)
        write_new(sidecars["selection"], {"protocol_sha256": digest(sidecars["protocol"]), "rows": sealed})
        selection_sha = digest(sidecars["selection"])
        print("All136 imagined tails sealed. Begin exact prefix replays and three-way suffix intervention.", flush=True)
        for old, choice, context, bank, tails in pending:
            seed, cells = old["seed"], []
            for index, carry in enumerate(tails):
                original = old["word_results"][f"word_{index}"]
                first = raw(bank[index:index + 1])
                root_frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                identity = p0._verify_execution_identity(root_frame, goal_rgb, context["root_frame"], context["goal"])
                advanced, boundaries = words.advance_program(env, context, first, verified_root_frame=root_frame)
                pixels, goal, history = prepare(advanced)
                emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                prefix_cost = float((emb[0, -1] - goal_latent).square().mean())
                cold, cold_receipt = compile_tail(goal_latent, emb, history)
                warm, warm_receipt = compile_tail(goal_latent, emb, history, carry)
                suffixes = {"cold": cold, "carry": carry, "warm": warm}
                with torch.no_grad():
                    model_costs = {name: float((p0._planning_coordinate(model,
                        p1._terminal_from_state(model, emb, history, params, action_lower=lower, action_upper=upper)[0], role="predicted")
                        - p0._planning_coordinate(model, goal_latent, role="goal")).square().mean()) for name, params in suffixes.items()}
                incumbent = choose_model_incumbent({name: model_costs[name] for name in ("cold", "warm")})
                gates = {"prefix_actions_exact": np.array_equal(first, np.asarray(original["action_witness_raw"], np.float32)[:20]),
                    "prefix_history_exact": boundaries == original["prefix_boundaries"], "root_identity_exact": identity == original["execution_identity"],
                    "prefix_rgb_cost_exact": prefix_cost == original["observed_prefix_rgb_cost"],
                    "cold_receipt_exact": cold_receipt == original["suffix_receipt"],
                    "cold_actions_exact": np.array_equal(raw(cold), np.asarray(original["action_witness_raw"], np.float32)[20:]),
                    "warm_model_cost_nonincreasing": warm_receipt["cost_nonincreasing"] and model_costs["warm"] <= model_costs["carry"]}
                if not all(gates.values()):
                    raise RuntimeError(f"historical replay or warm solver contract failed: {seed}/{index}: {gates}")
                suffix_record = {"seed": seed, "proposal_index": index, "incumbent": incumbent, "model_costs": model_costs,
                    "suffix_params": {name: params.tolist() for name, params in suffixes.items()},
                    "cold_receipt": cold_receipt, "warm_receipt": warm_receipt, "gates": gates}
                suffix_path = sidecars["roots"] / f"suffix_{seed}_{index}.selection.json"
                write_new(suffix_path, suffix_record)
                suffix_sha = digest(suffix_path)
                outcomes = {}
                for name, params in suffixes.items():
                    execution.restore_execution_snapshot(env, seed=seed, snapshot=advanced["execution_snapshot"], root_identity="sealed_observation_exact_state")
                    advanced_final, final_boundaries = words.advance_program(env, advanced, raw(params), verified_root_frame=advanced["root_frame"])
                    final_pixels, final_goal, final_history = prepare(advanced_final)
                    with torch.no_grad():
                        actual = p0._latent_context(model, final_pixels, final_history)[0, -1]
                        actual_rgb_cost = float((actual - p0._goal_latent(model, final_goal)).square().mean())
                    physics = env.unwrapped.env.physics
                    distance = p0._qpos_distance(np.asarray(physics.data.qpos), np.asarray(env.unwrapped.env.task.target_qpos))
                    outcomes[name] = {"final_qpos_distance": distance, "final_rgb_cost": actual_rgb_cost,
                        "suffix_raw": raw(params).tolist(), "final_rgb_sha256": advanced_final["root_rgb_sha256"], "suffix_boundaries": final_boundaries}
                gates.update({"cold_physical_result_exact": outcomes["cold"]["final_qpos_distance"] == original["physical"]["final_qpos_distance"],
                    "suffix_selection_unchanged": digest(suffix_path) == suffix_sha})
                cell = {"proposal_index": index, "imagined_terminal_cost": choice["receipt"]["proposal_rows"][index]["refined_goal_cost"],
                    "observed_prefix_rgb_cost": prefix_cost, "model_costs": model_costs, "incumbent": incumbent,
                    "outcomes": outcomes, "gates": gates, "suffix_selection_sha256": suffix_sha}
                cells.append(cell)
                write_new(sidecars["roots"] / f"cell_{seed}_{index}.json", cell)
                if not all(gates.values()):
                    raise RuntimeError(f"physical suffix replay failed: {seed}/{index}: {gates}")
            selected = old["decision"]["selected_program_index"]
            row = {"seed": seed, "source_episode": old["source_episode"], "source_start_row": old["source_start_row"],
                "decision": old["decision"], "cells": cells, "selected": cells[selected]}
            rows.append(row)
            print(json.dumps({"seed": seed, "selected": selected,
                "distance": {name: cell["final_qpos_distance"] for name, cell in cells[selected]["outcomes"].items()},
                "model_incumbent": cells[selected]["incumbent"], "all17_replayed": True}), flush=True)
    finally:
        model.predict = predict
        env.close()
    all_cells = [cell for row in rows for cell in row["cells"]]
    aggregate = {"roots": len(rows), "candidates": len(all_cells), "physical_suffix_branches": len(all_cells) * 3,
        "warm_model_wins_vs_cold": sum(cell["model_costs"]["warm"] < cell["model_costs"]["cold"] for cell in all_cells),
        "warm_physical_wins_vs_cold": sum(cell["outcomes"]["warm"]["final_qpos_distance"] < cell["outcomes"]["cold"]["final_qpos_distance"] for cell in all_cells),
        "mean_selected_distance": {name: float(np.mean([row["selected"]["outcomes"][name]["final_qpos_distance"] for row in rows])) for name in ("cold", "carry", "warm")},
        "incumbent_mean_selected_distance": float(np.mean([row["selected"]["outcomes"][row["selected"]["incumbent"]]["final_qpos_distance"] for row in rows]))}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_roots_retained": [row["seed"] for row in rows] == sorted(ref["root_seeds"]),
        "all_candidates_retained": len(all_cells) == 17 * len(joined),
        "all_replay_gates": all(all(cell["gates"].values()) for cell in all_cells),
        "selection_unchanged": digest(sidecars["selection"]) == selection_sha,
        "reference_unchanged": digest(args.reference) == protocol["reference_sha256"],
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    result = {"schema": "reacher_d2p_suffix_continuity_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "selection_sha256": selection_sha,
        "gates": gates, "aggregate": aggregate, "rows": rows, "actual_model_predict_samples": samples}
    write_new(output, result)
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
