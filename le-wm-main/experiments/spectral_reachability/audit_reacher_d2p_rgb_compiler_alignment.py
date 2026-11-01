"""Host-CUDA check of batch, singleton and P1-prefix scoring on sealed RGB roots.

This checks executable code paths, not model-to-reality accuracy or P1 efficacy.
It never changes a selector, threshold, checkpoint, or reference observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability.audit_reacher_d2p_runtime_support import _source_state_context
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds, _parameter_hash
from experiments.spectral_reachability.reacher_d2p_packet_support import load_packet_pool, proposal_parameters, receipt_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-report", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    host = require_cuda_allocation("cuda")
    reference = json.loads(args.rgb_report.read_text())
    if reference["status"] != "completed":
        raise ValueError("completed reference RGB audit required")
    policy = args.policy.resolve()
    weight_hash = hashlib.sha256((policy / "weights.pt").read_bytes()).hexdigest()
    matches = [item for item in reference["inputs"]["policies"] if item["path"] == str(policy)]
    if len(matches) != 1 or matches[0]["weights_sha256"] != weight_hash:
        raise ValueError("reference weights do not match")
    install_missing_field_compat()
    device = torch.device("cuda")
    model = load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = _parameter_hash(model)
    mean, std = _load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    if hashlib.sha256(args.adapter_artifact.read_bytes()).hexdigest() != reference["inputs"]["adapter_sha256"]:
        raise ValueError("reference adapter does not match")
    dataset, digest = receipt_source(args.adapter_artifact)
    pool, _ = load_packet_pool(dataset, expected_sha256=digest)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    lower_np, upper_np = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lower_np, device=device), torch.as_tensor(upper_np, device=device)
    rows = []
    try:
        with torch.no_grad():
            for row in reference["rows"]:
                context = _source_state_context(env, dataset=dataset, seed=row["seed"])
                if any(context[key] != row[key] for key in ("root_rgb_sha256", "goal_rgb_sha256", "source_episode", "source_start_row")):
                    raise RuntimeError("reference RGB/state identity differs; no root removal allowed")
                count = len(row["candidate_raw_packets"])
                bank, receipt = proposal_parameters(pool, proposals=count, horizon=1, seed=row["seed"],
                    mean=mean, std=std, lower=lower, upper=upper)
                if receipt != row["proposal_support"]:
                    raise RuntimeError("reference candidate bank differs")
                pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"],
                    context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)
                goal_z = p0._goal_latent(model, goal)
                batch_costs = p0._cost_fn(model, pixels, goal, history, bank, action_lower=lower, action_upper=upper).cpu().numpy()
                single_costs, prefix_errors, suffix_errors = [], [], []
                suffix = torch.zeros((1, 2, bank.shape[-1]), device=device)
                for index in range(count):
                    params = bank[index:index + 1]
                    raw, _ = p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)
                    if not np.array_equal(raw, np.asarray(row["candidate_raw_packets"][index], np.float32)):
                        raise RuntimeError("reference raw packet differs")
                    terminal = p0._terminal_batch(model, pixels, goal, history, params, action_lower=lower, action_upper=upper)
                    state, actions = p1._state_after_prefix(model, pixels, history, params, action_lower=lower, action_upper=upper)
                    single_costs.append(float((terminal[0] - goal_z).square().mean()))
                    prefix_errors.append(float((terminal - state[:, -1]).abs().max()))
                    full = p0._terminal_batch(model, pixels, goal, history, torch.cat((params, suffix), dim=1), action_lower=lower, action_upper=upper)
                    recompiled = p1._terminal_from_state(model, state, actions, suffix, action_lower=lower, action_upper=upper)
                    suffix_errors.append(float((full - recompiled).abs().max()))
                single_costs = np.asarray(single_costs)
                original = next(item for item in row["policies"] if item["path"] == str(policy))
                rows.append({"seed": row["seed"], "batch_vs_singleton_max_cost_error": float(np.max(np.abs(batch_costs - single_costs))),
                    "batch_argmin": int(batch_costs.argmin()), "singleton_argmin": int(single_costs.argmin()),
                    "zero_relative_order_flip_count": int(np.count_nonzero(np.sign(batch_costs - batch_costs[0]) != np.sign(single_costs - single_costs[0]))),
                    "reference_singleton_max_cost_error": float(np.max(np.abs(single_costs - original["predicted_rgb_goal_costs"]))),
                    "prefix_terminal_max_latent_error": max(prefix_errors), "recompiled_suffix_max_latent_error": max(suffix_errors)})
                print(json.dumps(rows[-1]), flush=True)
    finally:
        env.close()
    gates = {"weights_unchanged": before == _parameter_hash(model),
             "reference_singleton_exact": all(row["reference_singleton_max_cost_error"] == 0 for row in rows),
             "prefix_and_suffix_time_alignment_exact": all(row["prefix_terminal_max_latent_error"] == 0 and row["recompiled_suffix_max_latent_error"] == 0 for row in rows),
             "batch_selection_order_unchanged": all(row["batch_argmin"] == row["singleton_argmin"] and row["zero_relative_order_flip_count"] == 0 for row in rows)}
    result = {"schema": "reacher_d2p_rgb_compiler_alignment.v1", "status": "pass" if all(gates.values()) else "fail", "cuda_host": host,
              "gates": gates, "rows": rows, "weights_sha256": weight_hash,
              "reference_report": str(args.rgb_report.resolve()), "reference_sha256": hashlib.sha256(args.rgb_report.read_bytes()).hexdigest(),
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "claim_boundary": "Code-path alignment only, not model accuracy, selected non-greedy-prefix calibration, or P1 efficacy."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "gates": gates}))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
