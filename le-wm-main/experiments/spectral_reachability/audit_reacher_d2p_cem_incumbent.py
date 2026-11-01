"""Same-root CUDA replay of CEM final-generation vs best-ever output.

The sampling trajectory, model calls, action bounds and observed-successor
suffix are unchanged. Old runtime results are controls, never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_causal_innovation as replay
from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_cem_incumbent as cem
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner


def legacy_replay_gates(current, reference):
    return {"legacy_sampling_trace_exact": current["first_receipt"]["trace"] == reference["first_receipt"]["trace"],
        "legacy_full_actions_exact": current["action_witness_raw"] == reference["action_witness_raw"],
        "legacy_observed_first_chunk_exact": current["observed_first_chunk"] == reference["observed_first_chunk"],
        "legacy_final_distance_exact": current["physical"]["final_qpos_distance"] == reference["physical"]["final_qpos_distance"],
        "legacy_execution_identity_exact": current["execution_identity"] == reference["execution_identity"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.policy, args.output = args.policy.resolve(), args.output.resolve()
    report = json.loads(args.reference.read_text())
    replay.validate_reference(report)
    ref, digest = report["sealed_protocol"], replay.digest
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("reference checkpoint/action stats changed")
    implementations = {"auditor": Path(__file__), "cem": Path(cem.__file__), "context": Path(execution.__file__),
        "p0": Path(p0.__file__), "runner": Path(runner.__file__), "selector": Path(p1.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename)}
    if any(digest(implementations[key]) != ref["implementation_sha256"][key] for key in ("context", "p0", "runner", "selector", "source_context")):
        raise ValueError("reference execution/selector implementation changed")
    output = args.output
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"))):
        raise FileExistsError("new immutable output required")
    protocol = {"root_seeds": ref["root_seeds"], "reference_sha256": digest(args.reference),
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": ref["source_sha256"],
        "cem": ref["cem"], "changed_component": "best-ever output retention only; no proposal injection or extra scoring",
        "bootstrap_seed": 202626711, "legacy_replay_identity": "exact trace, all actions, observed scores, distance, execution identities",
        "implementation_sha256": {key: digest(path) for key, path in implementations.items()},
        "scope": "Known-root baseline selection diagnostic, no fresh efficacy, no complete resource-matched D2P comparison, no P1 override."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    print("Replaying unchanged CEM samples with last-generation and best-ever output; host CUDA only, no physical-informed selection.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha != ref["source_sha256"]:
        raise ValueError("source receipt changed")
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"]))
    low, high = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
    predict, calls, samples = model.predict, 0, 0

    def counted_predict(emb, act):
        nonlocal calls, samples
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU fallback forbidden")
        calls += 1
        samples += int(emb.shape[0])
        return predict(emb, act)

    model.predict = counted_predict
    rows = []
    try:
        for source in sorted(report["rows"], key=lambda row: row["seed"]):
            seed = source["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[key] != source[key] for key in ("source_episode", "source_start_row")):
                raise RuntimeError("source root changed")
            methods = {name: source["methods"][name] for name in ("dispatch", "same_bank_greedy")}
            for name in ("legacy_cem", "global_best_cem"):
                def program(pixels, goal, history, root_seed):
                    best, legacy, receipt = cem.search_with_incumbent(
                        lambda params: p0._cost_fn(model, pixels, goal, history, params, action_lower=lower, action_upper=upper),
                        future_horizon=ref["cem"]["horizon"], action_dim=history.shape[-1], device=device,
                        samples=ref["cem"]["samples"], steps=ref["cem"]["steps"], elite=ref["cem"]["elite"], seed=root_seed)
                    receipt["executed_output_selection"] = name
                    return (legacy if name == "legacy_cem" else best)[:, :1].detach(), receipt

                before_calls, before_samples = calls, samples
                methods[name] = runner._two_stage(name=name, first_program=program, env=env, seed=seed, context=context,
                    preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                    action_block=5, progress_horizon=2, damping=.01, device=device)
                methods[name]["actual_compute"] = {"model_predict_calls": calls - before_calls,
                    "model_predict_sample_evaluations": samples - before_samples,
                    "logical_backward_vjp_rows": methods[name]["suffix_receipt"]["backward_vjp_rows"]}
            legacy, best = methods["legacy_cem"], methods["global_best_cem"]
            gains = {"model_goal_cost_improvement": best["first_receipt"]["legacy_last_iteration_goal_cost"] - best["first_receipt"]["best_ever_goal_cost"],
                "same_prefix": legacy["first_action_witness_raw"] == best["first_action_witness_raw"]}
            gates = {**legacy_replay_gates(legacy, source["methods"]["lookahead_cem"]),
                "both_search_sampling_traces_exact": legacy["first_receipt"]["trace"] == best["first_receipt"]["trace"],
                "no_extra_model_or_vjp_work": legacy["actual_compute"] == best["actual_compute"],
                "same_root_goal_for_all_controls": all(method["execution_identity"]["initial"] == methods["dispatch"]["execution_identity"]["initial"] for method in methods.values()),
                "model_incumbent_not_worse": gains["model_goal_cost_improvement"] >= 0}
            row = {"seed": seed, "source_episode": source["source_episode"], "source_start_row": source["source_start_row"],
                "methods": methods, "gains": gains, "gates": gates}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "gates": gates, "gains": gains,
                "final_distances": {name: method["physical"]["final_qpos_distance"] for name, method in methods.items()}}), flush=True)
    finally:
        model.predict = predict
        env.close()
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_reference_roots_retained": [row["seed"] for row in rows] == sorted(ref["root_seeds"]),
        "all_replay_and_selection_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in implementations.items())}
    aggregate = {"strict_model_improvement_roots": sum(row["gains"]["model_goal_cost_improvement"] > 0 for row in rows),
        "changed_prefix_roots": sum(not row["gains"]["same_prefix"] for row in rows),
        "paired": {name: runner._paired(rows, "global_best_cem", name, protocol["bootstrap_seed"]) for name in ("legacy_cem", "dispatch", "same_bank_greedy")}}
    result = {"schema": "reacher_d2p_cem_incumbent_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "aggregate": aggregate, "rows": rows,
        "data_verification": "Prior source SHA receipt reused; original CEM full actions, root/goal and scores replayed exactly. No new full-source SHA claimed.",
        "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
