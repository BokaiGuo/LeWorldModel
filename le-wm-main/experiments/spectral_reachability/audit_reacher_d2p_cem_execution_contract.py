"""Fixed-prefix intervention on the CEM search/execution suffix contract.

Reconstruct the identical initial search. Compare its saved tail to the
historical shared DLS suffix and a newly searched observed-RGB CEM suffix.
All known roots and both CEM output policies are retained. Also apply the
observed-CEM suffix to the fixed dispatcher/greedy prefixes. Physical labels
are evaluator-only; this is not fresh efficacy or formal P1 admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_cem_incumbent as cem
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reference(report):
    if (report.get("schema") != "reacher_d2p_cem_incumbent_audit.v1" or report.get("status") != "completed"
            or not report.get("gates") or not all(report["gates"].values())):
        raise ValueError("complete integrity-passing incumbent audit required")
    seeds = report["sealed_protocol"]["root_seeds"]
    rows = report["rows"]
    observed = [row["seed"] for row in rows]
    if len(seeds) != len(set(seeds)) or len(observed) != len(set(observed)) or set(seeds) != set(observed):
        raise ValueError("unique and complete declared root identities required")
    return sorted(rows, key=lambda row: row["seed"])


def suffix_conditions(name):
    if name in ("legacy_cem", "global_best_cem"):
        return ("shared_dls", "stored_cem_tail", "observed_cem")
    if name in ("dispatch", "same_bank_greedy"):
        return ("shared_dls", "observed_cem")
    raise ValueError("unknown prefix policy")


def fixed_prefix_gate(actions, prefix):
    actions, prefix = np.asarray(actions), np.asarray(prefix)
    return actions.shape == (15, 2) and prefix.shape == (5, 2) and np.array_equal(actions[:5], prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--original-runtime", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.policy = args.policy.resolve()
    report = json.loads(args.reference.read_text())
    source_rows = validate_reference(report)
    ref = report["sealed_protocol"]
    if any(ref["cem"].get(key) != value for key, value in {"horizon": 3, "samples": 128, "steps": 5, "elite": 16}.items()):
        raise ValueError("this replay requires the declared 3-packet 128x5 CEM reference")
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("reference weights or normalizer changed")
    modules = {"context": Path(execution.__file__), "p0": Path(p0.__file__), "runner": Path(runner.__file__),
        "cem": Path(cem.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename)}
    if any(digest(path) != ref["implementation_sha256"][key] for key, path in modules.items()):
        raise ValueError("reference implementation changed")
    output = args.output.resolve()
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"))):
        raise FileExistsError("new immutable output and sidecars required")
    protocol = {"root_seeds": ref["root_seeds"], "reference_sha256": digest(args.reference),
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": ref["source_sha256"],
        "initial_cem": ref["cem"], "observed_cem": {"horizon": 2, "samples": 128, "steps": 5, "elite": 16,
            "seed_offset": 50000, "output_selection": "best_ever"},
        "bootstrap_seed": 202626811,
        "intervention": "exact fixed prefix; replace only suffix by stored optimized tail or observed-RGB CEM",
        "resource_contract": "causal suffix test, not resource matched; search/diagnostic work explicitly separated",
        "auditor_sha256": digest(Path(__file__)),
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Known-root search/execution causal diagnostic; no new efficacy, no P1 override, no learned components."}
    # Obtain the goal binding from the original runtime rather than assuming
    # the incumbent audit's omitted field. Follow and verify its hash receipt.
    original = args.original_runtime.resolve()
    if digest(original) != ref["reference_sha256"]:
        raise ValueError("original runtime goal protocol unavailable or changed")
    protocol["target_delta"] = json.loads(original.read_text())["sealed_protocol"]["target_delta"]
    protocol["original_runtime_sha256"] = digest(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    print("Fixed-prefix search/execution intervention: shared DLS, saved CEM tail, observed-RGB CEM; host CUDA only.", flush=True)
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
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(protocol["target_delta"]))
    low, high = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
    predict, calls, samples = model.predict, 0, 0

    def counted_predict(emb, act):
        nonlocal calls, samples
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        calls += 1
        samples += int(emb.shape[0])
        return predict(emb, act)

    model.predict = counted_predict

    def prepare(context):
        return p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
            np.asarray(context["history_action_blocks"], np.float32), device)

    def raw_actions(params):
        return p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)[0]

    def search(context_tensors, *, horizon, seed):
        started = time.perf_counter()
        c, s = calls, samples
        pixels, goal, history = context_tensors
        best, legacy, receipt = cem.search_with_incumbent(
            lambda params: p0._cost_fn(model, pixels, goal, history, params, action_lower=lower, action_upper=upper),
            future_horizon=horizon, action_dim=history.shape[-1], device=device,
            samples=128, steps=5, elite=16, seed=seed)
        receipt["actual_compute"] = {"model_predict_calls": calls - c, "model_predict_sample_evaluations": samples - s,
            "logical_backward_vjp_rows": 0, "wall_seconds": time.perf_counter() - started}
        return best, legacy, receipt

    rows = []
    try:
        for source in source_rows:
            seed = source["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[key] != source[key] for key in ("source_episode", "source_start_row")):
                raise RuntimeError("reference source identity changed")
            tensors = prepare(context)
            best, legacy, initial_receipt = search(tensors, horizon=3, seed=seed)
            full_plans = {"legacy_cem": legacy, "global_best_cem": best}
            methods, gates = {}, {}
            for name in ("legacy_cem", "global_best_cem", "dispatch", "same_bank_greedy"):
                original_method = source["methods"][name]
                prefix = np.asarray(original_method["first_action_witness_raw"], np.float32)
                if name in full_plans:
                    full_raw = raw_actions(full_plans[name])
                    gates[f"{name}_initial_search_exact"] = initial_receipt["trace"] == original_method["first_receipt"]["trace"]
                    gates[f"{name}_search_prefix_exact"] = np.array_equal(full_raw[:5], prefix)
                root, goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
                identity = p0._verify_execution_identity(root, goal, context["root_frame"], context["goal"])
                gates[f"{name}_root_identity_exact"] = identity == original_method["execution_identity"]["initial"]
                advanced = execution.advance_executed_context(env, context=context, raw_action_block=prefix, action_block=5, verified_root_frame=root)
                gates[f"{name}_successor_rgb_exact"] = advanced["root_rgb_sha256"] == original_method["observed_successor_context"]["root_rgb_sha256"]
                if not all(gates.values()):
                    raise RuntimeError(f"fixed-prefix replay failed: {gates}")
                successor_tensors = prepare(advanced)
                replan, _, replan_receipt = search(successor_tensors, horizon=2, seed=seed + 50000)
                suffixes = {"shared_dls": np.asarray(original_method["action_witness_raw"], np.float32)[5:],
                    "observed_cem": raw_actions(replan)}
                if name in full_plans:
                    suffixes["stored_cem_tail"] = full_raw[5:]
                c, s = calls, samples
                with torch.no_grad():
                    goal_latent = p0._goal_latent(model, successor_tensors[1])
                for condition in suffix_conditions(name):
                    suffix = suffixes[condition]
                    full_actions = np.concatenate((prefix, suffix))
                    gates[f"{name}_{condition}_fixed_prefix"] = fixed_prefix_gate(full_actions, prefix)
                    physical, successor_identity = p0._execute(env, seed, suffix, expected_frame=advanced["root_frame"],
                        expected_goal=advanced["goal"], execution_snapshot=advanced["execution_snapshot"])
                    frame = np.ascontiguousarray(env.render(), dtype=np.uint8)
                    prepared = processor._prepare_info({"pixels": frame[None, None], "goal": context["goal"][None, None],
                        "action": np.zeros((1, 1, mean.size), np.float32)})
                    with torch.no_grad():
                        latent = p0._goal_latent(model, prepared["pixels"].to(device))
                    key = f"{name}/{condition}"
                    methods[key] = {"physical": physical, "observed_final_rgb_goal_cost": float((latent - goal_latent).square().mean()),
                        "action_witness_raw": full_actions.tolist(), "execution_identity": {"initial": identity, "successor": successor_identity},
                        "suffix_search_receipt": replan_receipt if condition == "observed_cem" else None,
                        "resource_note": "observed CEM adds 1280 predict samples; shared DLS is stored replay (original 14 samples +192 VJP rows); saved tail adds zero search"}
                    if condition == "shared_dls":
                        gates[f"{name}_shared_dls_exact"] = physical["final_qpos_distance"] == original_method["physical"]["final_qpos_distance"]
                    if condition == "stored_cem_tail":
                        # Ensure the split does not itself change execution.
                        full_physical, _ = p0._execute(env, seed, full_actions, expected_frame=context["root_frame"],
                            expected_goal=context["goal"], execution_snapshot=context["execution_snapshot"])
                        gates[f"{name}_tail_split_full_replay_exact"] = full_physical["final_qpos_distance"] == physical["final_qpos_distance"]
                methods[f"{name}/observed_cem"]["evaluator_extra_predict_samples"] = samples - s
                methods[f"{name}/observed_cem"]["evaluator_extra_predict_calls"] = calls - c
            row = {"seed": seed, "source_episode": source["source_episode"], "source_start_row": source["source_start_row"],
                "initial_search_receipt": initial_receipt, "methods": methods, "gates": gates}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "all_gates": all(gates.values()),
                "final_distances": {key: value["physical"]["final_qpos_distance"] for key, value in methods.items()}}), flush=True)
    finally:
        model.predict = predict
        env.close()
    aggregate = {}
    for name in ("legacy_cem", "global_best_cem", "dispatch", "same_bank_greedy"):
        for condition in suffix_conditions(name)[1:]:
            key = f"{name}/{condition}"
            aggregate[key] = runner._paired(rows, key, f"{name}/shared_dls", protocol["bootstrap_seed"])
            aggregate[key]["vs_dispatch_shared_dls"] = runner._paired(rows, key, "dispatch/shared_dls", protocol["bootstrap_seed"])
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_declared_roots_retained": [row["seed"] for row in rows] == sorted(ref["root_seeds"]),
        "all_replay_and_intervention_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())
            and digest(Path(__file__)) == protocol["auditor_sha256"]}
    result = {"schema": "reacher_d2p_cem_execution_contract.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "aggregate": aggregate, "rows": rows,
        "data_verification": "Prior source SHA receipt reused, exact roots and successor RGB verified. No new full-source SHA claimed.",
        "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
