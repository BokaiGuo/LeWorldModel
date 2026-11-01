"""Compare frozen Reacher predictions with actual RGB successors, no readout.

Multiple explicitly named checkpoints share exactly the same source-native
root, fixed source-action bank and physical successors.  This is a diagnostic
on sampled source episodes, not independent episode-held-out P1 efficacy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.audit_reacher_d2p_runtime_support import _source_state_context
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds, _parameter_hash
from experiments.spectral_reachability.reacher_d2p_context import advance_executed_context, capture_execution_snapshot, restore_execution_snapshot, rgb_digest, pack_rgb_observation
from experiments.spectral_reachability.reacher_d2p_packet_support import load_packet_pool, proposal_parameters, receipt_source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_summary(predicted: list[float], observed: list[float]) -> dict:
    pred, obs = np.asarray(predicted, np.float64), np.asarray(observed, np.float64)
    if pred.shape != obs.shape or pred.ndim != 1 or len(pred) < 2 or not np.isfinite(pred).all() or not np.isfinite(obs).all():
        raise ValueError("RGB ranking needs finite, paired candidate costs")
    rho = float(spearmanr(pred, obs).statistic)
    if not np.isfinite(rho):
        raise ValueError("RGB candidate rank correlation is undefined")
    best, actual = int(pred.argmin()), int(obs.argmin())
    return {"spearman": rho, "predicted_argmin": best, "observed_argmin": actual,
            "argmin_agrees": best == actual,
            "observed_regret": float(obs[best] - obs[actual]),
            "selected_vs_zero_order_agrees": bool((pred[best] >= pred[0]) == (obs[best] >= obs[0]))}


def rgb_difference(actual: np.ndarray, expected: np.ndarray) -> dict:
    """Diagnostic only; equality is never replaced with an image tolerance."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    details = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
               "actual_sha256": rgb_digest(actual), "expected_sha256": rgb_digest(expected),
               "exact": bool(np.array_equal(actual, expected))}
    if actual.shape == expected.shape:
        changed = np.argwhere(actual != expected)
        details.update(max_pixel_error=int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max()),
                       changed_components=len(changed), changed_coordinates=changed[:32].tolist())
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, action="append", required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--proposals", type=int, default=128)
    parser.add_argument("--seed", type=int, default=202621000)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--execution-order", choices=("interleaved", "staged"), default="interleaved",
                        help="Staged separates singleton CUDA predictions, physical RGB collection, and RGB encoding; same bank/metric.")
    parser.add_argument("--root-identity", choices=("exact_render", "sealed_observation_exact_state"), default="exact_render",
                        help="Explicit diagnostic protocol; sealed mode reuses original measured root only after exact integration/scene verification.")
    args = parser.parse_args()
    if args.output.exists() or args.roots < 4 or args.proposals < 4:
        raise ValueError("new output, at least four roots/candidates required")
    policies = [path.resolve() for path in args.policy]
    if len(set(policies)) != len(policies):
        raise ValueError("duplicate policy paths")
    host = require_cuda_allocation(args.device)
    install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = _load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    dataset, digest = receipt_source(args.adapter_artifact)
    pool, support = load_packet_pool(dataset, expected_sha256=digest)
    models = []
    for path in policies:
        model = load_compatible_pretrained(path, device=device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        models.append(model)
    before = [_parameter_hash(model) for model in models]
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    lower_np, upper_np = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lower_np, device=device), torch.as_tensor(upper_np, device=device)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    rows = []
    root_records = args.output.with_suffix(".roots")
    root_records.mkdir(parents=True, exist_ok=False)
    seed, candidate, context = args.seed, None, None
    identity_failure = None
    try:
        with torch.no_grad():
            for offset in range(args.roots):
                seed = args.seed + offset
                context = _source_state_context(env, dataset=dataset, seed=seed)
                context["execution_snapshot"]["observed_root_rgb"] = pack_rgb_observation(context["root_frame"])
                bank, bank_receipt = proposal_parameters(pool, proposals=args.proposals, horizon=1, seed=seed,
                    mean=mean, std=std, lower=lower, upper=upper)
                pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"],
                    context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)
                goals = [p0._goal_latent(model, goal) for model in models]
                predicted, observed = [[] for _ in models], [[] for _ in models]
                raw_packets, successor_hashes = [], []
                staged_successors = []
                if args.execution_order == "staged":
                    for index in range(args.proposals):
                        for m, model in enumerate(models):
                            z = p0._terminal_batch(model, pixels, goal, history, bank[index:index + 1], action_lower=lower, action_upper=upper)[0]
                            predicted[m].append(float((z - goals[m]).square().mean()))
                    torch.cuda.synchronize()
                for candidate in range(args.proposals):
                    params = bank[candidate:candidate + 1]
                    if args.execution_order == "interleaved":
                        for m, model in enumerate(models):
                            z = p0._terminal_batch(model, pixels, goal, history, params, action_lower=lower, action_upper=upper)[0]
                            predicted[m].append(float((z - goals[m]).square().mean()))
                    root, rendered_goal = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity=args.root_identity)
                    if not np.array_equal(root, context["root_frame"]) or not np.array_equal(rendered_goal, context["goal"]):
                        repeats = [np.asarray(env.render(), np.uint8).copy() for _ in range(4)]
                        identity_failure = {"root": rgb_difference(root, context["root_frame"]),
                                            "goal": rgb_difference(rendered_goal, context["goal"]),
                                            "repeat_root_renders": [rgb_difference(frame, context["root_frame"]) for frame in repeats]}
                        np.savez_compressed(args.output.with_suffix(".failure_rgb.npz"), expected_root=context["root_frame"],
                                            actual_root=root, expected_goal=context["goal"], actual_goal=rendered_goal,
                                            repeat_root_renders=np.stack(repeats))
                        raise RuntimeError("source-native RGB audit failed exact root/goal restoration: " + json.dumps(identity_failure))
                    raw, normalized = p0._to_raw_actions(params, mean, std, lower, upper, env.action_space.low, env.action_space.high)
                    expected_normalized = (raw.reshape(1, -1) - mean) / std
                    if not np.allclose(normalized, expected_normalized, atol=2e-6, rtol=0):
                        raise RuntimeError("scored/physical packet normalization differs")
                    advanced = advance_executed_context(env, context=context, raw_action_block=raw, action_block=5, verified_root_frame=root)
                    if args.execution_order == "staged":
                        # Store observations, not latent proxies. No model call
                        # occurs during this physical-render phase.
                        staged_successors.append(advanced)
                    else:
                        successor_pixels, _, successor_history = p0._prepare(preprocessor, advanced["frames"], advanced["root_frame"],
                            advanced["goal"], np.asarray(advanced["history_action_blocks"], np.float32), device)
                        for m, model in enumerate(models):
                            z = p0._latent_context(model, successor_pixels, successor_history)[0, -1]
                            observed[m].append(float((z - goals[m]).square().mean()))
                    raw_packets.append(raw.tolist())
                    successor_hashes.append(rgb_digest(advanced["root_frame"]))
                for advanced in staged_successors:
                    successor_pixels, _, successor_history = p0._prepare(preprocessor, advanced["frames"], advanced["root_frame"],
                        advanced["goal"], np.asarray(advanced["history_action_blocks"], np.float32), device)
                    for m, model in enumerate(models):
                        z = p0._latent_context(model, successor_pixels, successor_history)[0, -1]
                        observed[m].append(float((z - goals[m]).square().mean()))
                rows.append({"seed": seed, "source_episode": context["source_episode"], "source_start_row": context["source_start_row"],
                    "root_warmup_policy": context["warmup_policy"], "root_rgb_sha256": context["root_rgb_sha256"],
                    "goal_rgb_sha256": context["goal_rgb_sha256"], "proposal_support": bank_receipt,
                    "candidate_raw_packets": raw_packets, "successor_rgb_sha256": successor_hashes,
                    "policies": [{"path": str(path), **rank_summary(predicted[m], observed[m]),
                                  "predicted_rgb_goal_costs": predicted[m], "observed_rgb_goal_costs": observed[m]}
                                 for m, path in enumerate(policies)]})
                with (root_records / f"root_{seed}.json").open("x") as handle:
                    handle.write(json.dumps(rows[-1], indent=2, sort_keys=True) + "\n")
                print(json.dumps({"root_completed": seed, "spearman": [item["spearman"] for item in rows[-1]["policies"]]}), flush=True)
    except Exception as error:
        failure = {"schema": "reacher_d2p_source_native_rgb_rank_failure.v1", "status": "execution_failed",
                   "error": str(error), "error_type": type(error).__name__, "seed": seed,
                   "candidate_index": candidate, "completed_roots": len(rows),
                   "execution_order": args.execution_order, "root_identity": args.root_identity, "identity_failure": identity_failure,
                   "observed_snapshot": capture_execution_snapshot(env),
                   "expected_snapshot": context["execution_snapshot"] if context is not None else None,
                   "scope": "Failure evidence only; partial roots must not be promoted to full confirmation."}
        with args.output.with_suffix(".failure.json").open("x") as handle:
            handle.write(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    finally:
        env.close()
    unchanged = before == [_parameter_hash(model) for model in models]
    aggregate = []
    for m, path in enumerate(policies):
        metrics = [row["policies"][m] for row in rows]
        aggregate.append({"policy": str(path), "mean_spearman": float(np.mean([item["spearman"] for item in metrics])),
            "min_spearman": min(item["spearman"] for item in metrics),
            "argmin_agreement_count": sum(item["argmin_agrees"] for item in metrics),
            "selected_vs_zero_order_agreement_count": sum(item["selected_vs_zero_order_agrees"] for item in metrics)})
    gates = {"cuda_only": True, "weights_unchanged": unchanged, "source_root_goal_bound_before_history": True,
             "same_root_physical_successors_for_all_policies": True, "no_qpos_readout_or_latent_adapter": True,
             "same_rgb_latent_mse_predicted_and_observed": True, "singleton_candidate_prediction": True}
    result = {"schema": "reacher_d2p_source_native_rgb_rank.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "aggregate": aggregate, "rows": rows,
        "inputs": {"adapter": str(args.adapter_artifact.resolve()), "adapter_sha256": _sha256(args.adapter_artifact),
                   "source_support": support, "policies": [{"path": str(path), "weights_sha256": _sha256(path / "weights.pt")} for path in policies],
                   "auditor_sha256": _sha256(Path(__file__))},
        "protocol": {"root_seeds": [row["seed"] for row in rows], "proposals": args.proposals,
                     "execution_order": args.execution_order,
                     "root_identity": args.root_identity,
                     "root_context": "official_source_state_history_v3_goal_bound_before_history_render",
                     "root_identity_validation": ("original measured root RGB sealed and hash verified; exact integration state, complete compiled model hash, marker and bound goal before execution; successors always freshly rendered" if args.root_identity == "sealed_observation_exact_state" else "actual restored RGB exact, then fresh exact integration state, complete compiled model hash and bound goal before execution; no second root render"),
                     "metric": "mean squared RGB encoder latent distance to same rendered source-root goal"},
        "claim_boundary": "Source-episode diagnostic only, not episode-held-out P1 calibration, efficacy, or proof of a theoretical limitation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "aggregate": aggregate}, sort_keys=True))
    if not all(gates.values()):
        raise SystemExit("RGB ranking audit integrity failure")


if __name__ == "__main__":
    main()
