"""Measure actual host-CUDA model.predict samples against D2P receipts.

Fixed first reference root and its full candidate bank; no physical outcomes.
Reverse-mode output rows are logical VJP work, not measured kernel counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--rgb-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    reference = json.loads(args.rgb_reference.read_text())
    policy = args.policy.resolve()
    if reference["status"] != "completed" or not any(
        row["path"] == str(policy) and row["weights_sha256"] == digest(policy / "weights.pt")
        for row in reference["inputs"]["policies"]
    ) or reference["inputs"]["adapter_sha256"] != digest(args.adapter_artifact):
        raise ValueError("completed, matching frozen reference required")
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    model = shared.load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    dataset, source_hash = shared.receipt_source(args.adapter_artifact)
    pool, _ = shared.load_packet_pool(dataset, expected_sha256=source_hash)
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    samples = 0
    calls = 0
    predict = model.predict

    def counted_predict(embeddings, actions):
        nonlocal samples, calls
        if embeddings.device.type != "cuda" or actions.device.type != "cuda":
            raise RuntimeError("CPU fallback forbidden")
        samples += int(embeddings.shape[0])
        calls += 1
        return predict(embeddings, actions)

    model.predict = counted_predict
    try:
        source = reference["rows"][0]
        context = shared._source_state_context(env, dataset=dataset, seed=source["seed"])
        if any(context[key] != source[key] for key in ("root_rgb_sha256", "goal_rgb_sha256", "source_episode", "source_start_row")):
            raise RuntimeError("reference root identity changed")
        lower_np, upper_np = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(lower_np, device=device), torch.as_tensor(upper_np, device=device)
        count = len(source["candidate_raw_packets"])
        bank, support = shared.proposal_parameters(pool, proposals=count, horizon=1, seed=source["seed"],
            mean=mean, std=std, lower=lower, upper=upper)
        if support != source["proposal_support"]:
            raise RuntimeError("reference bank changed")
        processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
        pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
            np.asarray(context["history_action_blocks"], np.float32), device)
        _, prefix_receipt = p1._compile_reconfiguration(model, pixels, goal, history,
            reconfiguration_horizon=1, future_horizon=2, action_lower=lower, action_upper=upper,
            damping=10.0, proposals=count, non_greedy_margin=1e-6, reachability_margin=1e-6,
            seed=source["seed"], proposal_bank=bank, future_reference="same_bank_greedy")
        prefix_measured = {"model_predict_sample_evaluations": samples, "model_predict_calls": calls}
        samples = calls = 0
        params, suffix_receipt = p1._bounded_single_step_dls_program(model, pixels, goal, history,
            future_horizon=2, action_lower=lower, action_upper=upper, damping=10.0)
        suffix_measured = {"model_predict_sample_evaluations": samples, "model_predict_calls": calls}
        with torch.no_grad():
            replay_cost = float(p0._cost_fn(model, pixels, goal, history, params,
                action_lower=lower, action_upper=upper)[0])
    finally:
        model.predict = predict
        env.close()
    gates = {
        "cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "prefix_sample_count_exact": prefix_receipt["model_predict_sample_evaluations"] == prefix_measured["model_predict_sample_evaluations"],
        "suffix_sample_count_exact": suffix_receipt["model_predict_sample_evaluations"] == suffix_measured["model_predict_sample_evaluations"],
        "suffix_logical_vjp_rows_equal_output_dimension": suffix_receipt["backward_vjp_rows"] == suffix_receipt["jacobian_output_dimension"],
        "accepted_actions_replay_the_reported_cost": abs(replay_cost - suffix_receipt["predicted_cost_after"]) <= 1e-6,
        "suffix_model_objective_nonincreasing": suffix_receipt["actual_model_cost_nonincreasing"],
    }
    result = {"schema": "reacher_d2p_compute_accounting.v1", "status": "pass" if all(gates.values()) else "fail",
        "cuda_host": host, "seed": source["seed"], "gates": gates,
        "prefix_measured": prefix_measured, "prefix_receipt": prefix_receipt,
        "suffix_measured": suffix_measured, "suffix_receipt": suffix_receipt, "replayed_suffix_cost": replay_cost,
        "inputs": {"reference_sha256": digest(args.rgb_reference), "weights_sha256": digest(policy / "weights.pt"),
                   "adapter_sha256": digest(args.adapter_artifact), "selector_sha256": digest(Path(p1.__file__)),
                   "auditor_sha256": digest(Path(__file__)), "setup_module_sha256": digest(Path(shared.__file__))},
        "claim_boundary": "Instrumentation audit only. Logical VJP rows are not GPU-kernel counts. No efficacy or compute-matched comparison claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "gates": gates, "prefix_measured": prefix_measured,
                      "suffix_measured": suffix_measured}), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
