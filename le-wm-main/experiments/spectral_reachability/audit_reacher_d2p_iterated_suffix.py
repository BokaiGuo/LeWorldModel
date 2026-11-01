"""Known-root, fixed-prefix SVD relinearization ablation on host CUDA.

First iteration must replay the historical suffix exactly. Retain all 1/2/4/8
iteration checkpoints; do not select one using physical outcomes. A separately
completed CEM suffix audit supplies same-prefix controls, not a budget match.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest, validate_reference, fixed_prefix_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--execution-reference", type=Path, required=True)
    parser.add_argument("--fine-rescue", action="store_true")
    parser.add_argument("--coarse-reference", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.policy = args.policy.resolve()
    report = json.loads(args.reference.read_text())
    source_rows = validate_reference(report)
    ref = report["sealed_protocol"]
    controls = json.loads(args.execution_reference.read_text())
    if (controls.get("schema") != "reacher_d2p_cem_execution_contract.v1" or controls.get("status") != "completed"
            or not controls.get("gates") or not all(controls["gates"].values())
            or controls["sealed_protocol"]["reference_sha256"] != digest(args.reference)):
        raise ValueError("complete CEM suffix intervention on exact same reference required")
    control_rows = {row["seed"]: row for row in controls["rows"]}
    if len(control_rows) != len(controls["rows"]) or set(control_rows) != set(ref["root_seeds"]):
        raise ValueError("CEM control roots differ or contain duplicates")
    coarse_rows = {}
    if args.fine_rescue:
        if args.coarse_reference is None:
            raise ValueError("fine rescue requires a complete coarse-grid reference")
        coarse = json.loads(args.coarse_reference.read_text())
        if (coarse.get("schema") != "reacher_d2p_iterated_suffix_audit.v1" or coarse.get("status") != "completed"
                or not coarse.get("gates") or not all(coarse["gates"].values())
                or coarse["sealed_protocol"]["reference_sha256"] != digest(args.reference)
                or coarse["sealed_protocol"].get("fine_rescue", False)):
            raise ValueError("invalid coarse-grid reference")
        coarse_rows = {row["seed"]: row for row in coarse["rows"]}
        if len(coarse_rows) != len(coarse["rows"]) or set(coarse_rows) != set(ref["root_seeds"]):
            raise ValueError("coarse reference root identities differ")
    elif args.coarse_reference is not None:
        raise ValueError("coarse reference only applies to fine-rescue mode")
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("reference weights/normalizer changed")
    modules = {"context": Path(execution.__file__), "p0": Path(p0.__file__), "runner": Path(runner.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename)}
    if any(digest(path) != ref["implementation_sha256"][key] for key, path in modules.items()):
        raise ValueError("execution implementation changed")
    modules.update({"solver": Path(solver.__file__), "auditor": Path(__file__)})
    if args.fine_rescue:
        modules["rescue"] = Path(rescue.__file__)
    output = args.output.resolve()
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"))):
        raise FileExistsError("new immutable output required")
    protocol = {"root_seeds": ref["root_seeds"], "reference_sha256": digest(args.reference),
        "execution_reference_sha256": digest(args.execution_reference), "weights_sha256": ref["weights_sha256"],
        "adapter_sha256": ref["adapter_sha256"], "source_sha256": ref["source_sha256"],
        "target_delta": controls["sealed_protocol"]["target_delta"], "suffix_horizon": 2,
        "fine_rescue": args.fine_rescue, "fine_rescue_exponents": [5, 16] if args.fine_rescue else None,
        "coarse_reference_sha256": digest(args.coarse_reference) if args.fine_rescue else None,
        "damping": .01, "iteration_checkpoints": [1, 2, 4, 8], "bootstrap_seed": 202626911,
        "changed_component": "fixed prefix; model-only fine line search only at coarse stop" if args.fine_rescue else "suffix SVD relinearization only, exact first-step replay, fixed stored prefix",
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "scope": "Known-root solver ablation, no physical-informed checkpoint selection, no P1 override or equal-budget efficacy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    print(f"Testing fixed 1/2/4/8 SVD relinearizations after exact known prefixes; fine rescue={args.fine_rescue}; host CUDA only.", flush=True)
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
    predict, sample_count = model.predict, 0

    def counted_predict(emb, act):
        nonlocal sample_count
        if emb.device.type != "cuda" or act.device.type != "cuda":
            raise RuntimeError("CPU neural fallback forbidden")
        sample_count += int(emb.shape[0])
        return predict(emb, act)

    model.predict = counted_predict
    rows = []
    try:
        for source in source_rows:
            seed = source["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[key] != source[key] for key in ("source_episode", "source_start_row")):
                raise RuntimeError("reference source root changed")
            methods, gates = {}, {}
            for name in ("dispatch", "same_bank_greedy"):
                old = source["methods"][name]
                prefix = np.asarray(old["first_action_witness_raw"], np.float32)
                root, goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
                identity = p0._verify_execution_identity(root, goal, context["root_frame"], context["goal"])
                gates[f"{name}_root_exact"] = identity == old["execution_identity"]["initial"]
                advanced = execution.advance_executed_context(env, context=context, raw_action_block=prefix, action_block=5, verified_root_frame=root)
                gates[f"{name}_successor_rgb_exact"] = advanced["root_rgb_sha256"] == old["observed_successor_context"]["root_rgb_sha256"]
                pixels, goal, history = p0._prepare(processor, advanced["frames"], advanced["root_frame"], advanced["goal"],
                    np.asarray(advanced["history_action_blocks"], np.float32), device)
                emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
                start_samples = sample_count
                started = time.perf_counter()
                snapshots = solver.compile_snapshots(model, goal_latent, emb, history, future_horizon=2,
                    action_lower=lower, action_upper=upper, damping=.01, checkpoints=(1, 2, 4, 8),
                    accept_update=rescue.accept_with_fine_rescue if args.fine_rescue else None)
                total_seconds = time.perf_counter() - started
                gates[f"{name}_predict_samples_exact"] = sample_count - start_samples == snapshots[8][1]["model_predict_sample_evaluations"]
                for iteration, (params, receipt) in snapshots.items():
                    suffix, _ = p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)
                    actions = np.concatenate((prefix, suffix))
                    gates[f"{name}_{iteration}_fixed_prefix"] = fixed_prefix_gate(actions, prefix)
                    physical, successor_identity = p0._execute(env, seed, suffix, expected_frame=advanced["root_frame"],
                        expected_goal=advanced["goal"], execution_snapshot=advanced["execution_snapshot"])
                    frame = np.ascontiguousarray(env.render(), dtype=np.uint8)
                    prepared = processor._prepare_info({"pixels": frame[None, None], "goal": context["goal"][None, None],
                        "action": np.zeros((1, 1, mean.size), np.float32)})
                    latent = p0._goal_latent(model, prepared["pixels"].to(device))
                    methods[f"{name}/dls_{iteration}"] = {"physical": physical,
                        "observed_final_rgb_goal_cost": float((latent - goal_latent).square().mean()),
                        "action_witness_raw": actions.tolist(), "suffix_receipt": receipt,
                        "execution_identity": {"initial": identity, "successor": successor_identity},
                        "full_8_snapshot_solver_wall_seconds": total_seconds}
                    gates[f"{name}_{iteration}_model_nonincreasing"] = receipt["cost_nonincreasing"]
                    if iteration == 1:
                        rescued = args.fine_rescue and receipt["trace"][0]["fine_rescue_accepted"]
                        gates[f"{name}_first_actions_exact_unless_rescued"] = rescued or actions.tolist() == old["action_witness_raw"]
                        gates[f"{name}_first_distance_exact_unless_rescued"] = rescued or physical["final_qpos_distance"] == old["physical"]["final_qpos_distance"]
                methods[f"{name}/observed_cem"] = control_rows[seed]["methods"][f"{name}/observed_cem"]
                gates[f"{name}_cem_fixed_prefix"] = fixed_prefix_gate(methods[f"{name}/observed_cem"]["action_witness_raw"], prefix)
                if args.fine_rescue:
                    methods[f"{name}/coarse_dls_8"] = coarse_rows[seed]["methods"][f"{name}/dls_8"]
                    gates[f"{name}_coarse_fixed_prefix"] = fixed_prefix_gate(methods[f"{name}/coarse_dls_8"]["action_witness_raw"], prefix)
                    if not any(item["fine_rescue_accepted"] for item in snapshots[8][1]["trace"]):
                        gates[f"{name}_unrescued_full_actions_exact"] = methods[f"{name}/dls_8"]["action_witness_raw"] == methods[f"{name}/coarse_dls_8"]["action_witness_raw"]
            row = {"seed": seed, "source_episode": source["source_episode"], "source_start_row": source["source_start_row"],
                "methods": methods, "gates": gates}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "all_gates": all(gates.values()),
                "final_distances": {key: value["physical"]["final_qpos_distance"] for key, value in methods.items()}}), flush=True)
    finally:
        model.predict = predict
        env.close()
    aggregate = {}
    for name in ("dispatch", "same_bank_greedy"):
        for iteration in (2, 4, 8):
            key = f"{name}/dls_{iteration}"
            aggregate[key] = runner._paired(rows, key, f"{name}/dls_1", protocol["bootstrap_seed"])
            aggregate[key]["vs_same_prefix_observed_cem"] = runner._paired(rows, key, f"{name}/observed_cem", protocol["bootstrap_seed"])
            if args.fine_rescue:
                aggregate[key]["vs_coarse_dls_8"] = runner._paired(rows, key, f"{name}/coarse_dls_8", protocol["bootstrap_seed"])
        if args.fine_rescue:
            aggregate[f"{name}/rescue_counts"] = {
                "coarse_stop_branches": sum(any(item["coarse_stop"] for item in row["methods"][f"{name}/dls_8"]["suffix_receipt"]["trace"]) for row in rows),
                "rescued_branches": sum(any(item["fine_rescue_accepted"] for item in row["methods"][f"{name}/dls_8"]["suffix_receipt"]["trace"]) for row in rows)}
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_declared_roots_retained": [row["seed"] for row in rows] == sorted(ref["root_seeds"]),
        "all_intervention_and_replay_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    result = {"schema": "reacher_d2p_iterated_suffix_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "aggregate": aggregate, "rows": rows,
        "data_verification": "Prior full-source SHA receipt reused; exact source/root/successor identities checked. No new full-source SHA claimed.",
        "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
