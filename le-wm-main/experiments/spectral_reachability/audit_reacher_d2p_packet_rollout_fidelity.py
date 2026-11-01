"""Fixed-eight-root packetwise free/observed-history prediction diagnosis.

All sealed native and neutral prefixes are retained. Teacher forcing uses
evaluator-only future RGB; it is never a selector or a deployed correction.
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
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability.analyze_reacher_d2p_state_native_selector import summarize as validate_reference
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def validate_packet_rows(rows):
    if len(rows) != 4 or [r["packet"] for r in rows] != [1, 2, 3, 4]:
        raise ValueError("all four ordered packet observations required")
    for row in rows:
        if set(row["vectors"]) != {"free", "teacher_forced", "observed", "goal"}:
            raise ValueError("all full-coordinate packet vectors required")
        arrays = {k: np.asarray(v, np.float64) for k, v in row["vectors"].items()}
        if any(v.shape != (192,) or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("finite 192-coordinate vectors required")
        for kind in ("free", "teacher_forced", "observed"):
            cost = float(np.mean((arrays[kind] - arrays["goal"]) ** 2))
            if not np.isclose(cost, row["costs"][kind], rtol=0, atol=2e-6):
                raise ValueError("stored packet cost differs from vectors")
    if rows[0]["vectors"]["free"] != rows[0]["vectors"]["teacher_forced"]:
        raise ValueError("first packet must share the identical actual-history prediction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    side_paths = {k: args.reference.with_suffix(ext) for k, ext in (("prefix", ".selection.json"), ("suffix", ".suffix.selection.json"), ("protocol", ".protocol.json"))}
    prefix, suffix = (json.loads(side_paths[k].read_text()) for k in ("prefix", "suffix"))
    if (digest(side_paths["prefix"]) != reference["selection_sha256"] or digest(side_paths["suffix"]) != reference["suffix_selection_sha256"]
            or json.loads(side_paths["protocol"].read_text()) != reference["sealed_protocol"]
            or prefix["protocol_sha256"] != digest(side_paths["protocol"]) or suffix["protocol_sha256"] != digest(side_paths["protocol"])
            or suffix["prefix_selection_sha256"] != digest(side_paths["prefix"])):
        raise ValueError("complete reference decision seals required")
    prior = validate_reference(reference, prefix, suffix)
    if prior["goal_contract"] != "joint-feasible-v1":
        raise ValueError("explicit feasible-goal reference required")
    ref = reference["sealed_protocol"]
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if (digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]
            or source_sha != ref["source_sha256"]):
        raise ValueError("original checkpoint/source/normalization required")
    outputs = {k: args.output.with_suffix(ext) for k, ext in (("protocol", ".protocol.json"), ("implementation", ".implementation"), ("roots", ".roots"))}
    if any(p.exists() for p in [args.output, *outputs.values()]):
        raise FileExistsError("new immutable packet-audit outputs required")
    modules = {"auditor": Path(__file__), "p0": Path(p0.__file__), "context": Path(execution.__file__),
        "goal_contract": Path(goals.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename),
        "loader": Path(shared.load_compatible_pretrained.__code__.co_filename), "released_model": Path(released.__file__)}
    for key in ("p0", "context", "goal_contract", "source_context"):
        if digest(modules[key]) != ref["implementation_sha256"][key]:
            raise ValueError("reference implementation changed: " + key)
    protocol = {"root_seeds": prior["root_seeds"], "reference_path": str(args.reference.resolve()), "reference_sha256": digest(args.reference),
        "prefix_sha256": digest(side_paths["prefix"]), "suffix_sha256": digest(side_paths["suffix"]),
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": source_sha,
        "goal_contract": prior["goal_contract"], "target_delta": ref["target_delta"], "families": ["native_dispatch", "zero"],
        "packets_per_branch": 4, "raw_steps_per_packet": 5, "physical_labels_used_for_planning": False,
        "released_rollout_equivalence": True, "native_rollout_atol": 2e-6,
        "teacher_forcing": "evaluator-only actual RGB history before each sealed packet; no feedback selection or oracle deployment",
        "source_verification": "Existing SHA-bound official source receipt reused, no new complete-file SHA claimed.",
        "implementation_sha256": {k: digest(p) for k, p in modules.items()},
        "scope": "Known all-eight-root fixed-prefix causal error localization, not new-root efficacy, calibration, P1 or theory rejection."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(outputs["protocol"], protocol)
    outputs["implementation"].mkdir()
    outputs["roots"].mkdir()
    for k, p in modules.items():
        shutil.copyfile(p, outputs["implementation"] / (k + ".py"))
    print("HOST CUDA: replay all8 sealed native/neutral four-packet prefixes. Compare free rollout and evaluator-only actual-history predictions; no action changes, training, or P1 admission.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    params_before = shared._parameter_hash(model)
    buffers_before = {k: v.detach().clone() for k, v in model.named_buffers()}
    predict, count = model.predict, 0

    def counted(emb, actions):
        nonlocal count
        if emb.device.type != "cuda" or actions.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        count += int(emb.shape[0])
        return predict(emb, actions)

    model.predict = counted
    env = goals.goal_wrapper(prior["goal_contract"])(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"]))
    lo, hi = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = (torch.as_tensor(v, device="cuda") for v in (lo, hi))
    choices = {r["seed"]: r for r in prefix["rows"]}
    rows = []

    def prepare(ctx):
        return p0._prepare(processor, ctx["frames"], ctx["root_frame"], ctx["goal"], np.asarray(ctx["history_action_blocks"], np.float32), torch.device("cuda"))

    try:
        for original in sorted(reference["rows"], key=lambda r: r["seed"]):
            seed, branches = original["seed"], {}
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[k] != choices[seed][k] for k in ("source_episode", "source_start_row", "root_rgb_sha256", "goal_rgb_sha256")):
                raise RuntimeError("reference source root/goal changed")
            goals.capture_goal_admission(env, prior["goal_contract"])
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            pixels, goal, history = prepare(context)
            target = p0._goal_latent(model, goal)
            bank = torch.as_tensor(choices[seed]["word_params"], device="cuda", dtype=torch.float32)
            for family in protocol["families"]:
                prior_branch = original["branches"][family]
                index = prior_branch["prefix_index"]
                params = bank[index:index + 1]
                raw = p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]
                if not np.array_equal(raw, np.asarray(prior_branch["prefix_actions_raw"], np.float32)):
                    raise RuntimeError("sealed complete raw actions changed")
                frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                p0._verify_execution_identity(frame, goal_rgb, context["root_frame"], context["goal"])
                current, packets = context, []
                normalized = p0._bounded_normalized_actions(params, lower, upper)
                free_emb = p0._latent_context(model, pixels, history)
                free_actions = torch.cat((history[:, 1:], normalized[:, :1]), dim=1)
                with torch.no_grad():
                    terminal = p0._terminal_batch(model, pixels, goal, history, params, action_lower=lower, action_upper=upper)[0]
                    native_info = model.rollout({"pixels": pixels[:, None]},
                        torch.cat((history[:, 1:], normalized), dim=1)[:, None], history_size=3)
                    native_vectors = native_info["predicted_emb"][0, 0, -4:]
                    for step in range(4):
                        free = model.predict(free_emb[:, -3:], model.action_encoder(free_actions[:, -3:]))[0, -1]
                        observed_pixels, _, observed_history = prepare(current)
                        forced = p0._terminal_batch(model, observed_pixels, goal, observed_history, params[:, step:step + 1], action_lower=lower, action_upper=upper)[0]
                        current = execution.advance_executed_context(env, context=current, raw_action_block=raw[step * 5:(step + 1) * 5],
                            action_block=5, verified_root_frame=frame if step == 0 else current["root_frame"])
                        current["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(current["root_frame"])
                        boundary = {"raw_steps_executed": (step + 1) * 5, "root_rgb_sha256": current["root_rgb_sha256"],
                            "history_rgb_sha256": current["history_rgb_sha256"], "action_sha256": current["advanced_action_sha256"]}
                        if boundary != prior_branch["prefix_boundaries"][step]:
                            raise RuntimeError("original packet boundary did not replay exactly")
                        next_pixels, _, next_history = prepare(current)
                        observed = p0._latent_context(model, next_pixels, next_history)[0, -1]
                        vectors = {"free": free, "teacher_forced": forced, "observed": observed, "goal": target}
                        packets.append({"packet": step + 1, "boundary": boundary, "vectors": {k: v.tolist() for k, v in vectors.items()},
                            "costs": {k: float((v - target).square().mean()) for k, v in vectors.items() if k != "goal"}})
                        free_emb = torch.cat((free_emb, free[None, None]), dim=1)
                        if step < 3:
                            free_actions = torch.cat((free_actions, normalized[:, step + 1:step + 2]), dim=1)
                validate_packet_rows(packets)
                gates = {"manual_rollout_terminal_bit_exact": terminal.tolist() == packets[-1]["vectors"]["free"],
                    "released_rollout_all4_equivalent": max(float((v - torch.tensor(p["vectors"]["free"], device="cuda")).abs().max())
                        for v, p in zip(native_vectors, packets)) <= protocol["native_rollout_atol"],
                    "original_model_cost_replay": abs(packets[-1]["costs"]["free"] - choices[seed]["score_rows"][index]["immediate_goal_cost"]) <= 2e-6,
                    "original_actual_cost_replay": packets[-1]["costs"]["observed"] == prior_branch["prefix_rgb_cost"],
                    "all_boundaries_and_actions_exact": True}
                if not all(gates.values()):
                    raise RuntimeError("packet trajectory baseline integrity failed")
                branches[family] = {"prefix_index": index, "packets": packets, "gates": gates,
                    "released_rollout_vectors": native_vectors.tolist(),
                    "released_rollout_max_abs_difference": max(float(np.max(np.abs(np.asarray(v) - np.asarray(p["vectors"]["free"]))))
                        for v, p in zip(native_vectors.tolist(), packets))}
            row = {"seed": seed, "native_witness": choices[seed]["native_decision"]["reconfiguration_witness_found"], "branches": branches}
            write_new(outputs["roots"] / f"root_{seed}.json", row)
            rows.append(row)
            print(json.dumps({"seed": seed, "all_branch_gates": True, "final_margins": {k: branches["native_dispatch"]["packets"][-1]["costs"][k] - branches["zero"]["packets"][-1]["costs"][k] for k in ("free", "teacher_forced", "observed")}}), flush=True)
    finally:
        model.predict = predict
        env.close()
    gates = {"cuda_only": True, "all8_roots": [r["seed"] for r in rows] == protocol["root_seeds"],
        "all16_fixed_prefixes_and64_boundaries_exact": len(rows) == 8 and all(all(all(v is True for v in b["gates"].values()) for b in r["branches"].values()) for r in rows),
        "parameters_unchanged": shared._parameter_hash(model) == params_before,
        "buffers_unchanged": all(torch.equal(v, buffers_before[k]) for k, v in model.named_buffers()),
        "all_modules_eval": not any(m.training for m in model.modules()), "predict_sample_accounting": count == 256,
        "reference_and_decisions_unchanged": digest(args.reference) == protocol["reference_sha256"] and digest(side_paths["prefix"]) == protocol["prefix_sha256"] and digest(side_paths["suffix"]) == protocol["suffix_sha256"],
        "weights_file_unchanged": digest(args.policy / "weights.pt") == protocol["weights_sha256"],
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    write_new(args.output, {"schema": "reacher_d2p_packet_rollout_fidelity.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "protocol_sha256": digest(outputs["protocol"]), "gates": gates, "rows": rows,
        "actual_model_predict_samples": count})
    print(json.dumps({"gates": gates, "actual_model_predict_samples": count}), flush=True)


if __name__ == "__main__":
    main()
