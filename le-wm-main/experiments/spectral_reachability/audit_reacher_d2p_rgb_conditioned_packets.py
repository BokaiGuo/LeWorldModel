"""Host-CUDA paired random/temporal-RGB-retrieved packet support audit.

Retrieval is frozen, training-free, causal and episode-disjoint. Every fixed
candidate is physically measured, not only a favorable model-selected subset.
This isolates proposal conditioning, NOT a new compiler or a P1 efficacy run.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_causal_innovation as replay
from experiments.spectral_reachability import audit_reacher_d2p_packet_interpolation as interpolation
from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_rgb_context_support as support


def aggregate_banks(rows, *, seed, bootstrap_mean):
    metrics = ("mean_abs_cost_error", "mean_abs_zero_relative_margin_error", "pairwise_agreement_fraction")
    return {metric: {"means": {name: float(np.mean([row["banks"][name]["metrics"][metric] for row in rows])) for name in ("random", "retrieved")},
        "retrieved_minus_random_ci95": bootstrap_mean(np.asarray([row["banks"]["retrieved"]["metrics"][metric] - row["banks"]["random"]["metrics"][metric] for row in rows], np.float64), seed=seed)} for metric in metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--exclusion-reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-size", type=int, default=4096)
    parser.add_argument("--index-seed", type=int, default=202640000)
    parser.add_argument("--proposals", type=int, default=32)
    args = parser.parse_args()
    if args.proposals < 2 or args.index_size < args.proposals - 1:
        raise ValueError("positive support and at least two proposals required")
    reference = json.loads(args.reference.read_text())
    replay.validate_reference(reference)
    ref, digest = reference["sealed_protocol"], replay.digest
    exclusions = json.loads(args.exclusion_reference.read_text())
    if exclusions.get("status") != "completed" or not exclusions.get("gates") or not all(exclusions["gates"].values()):
        raise ValueError("completed exclusion metadata receipt required")
    excluded = set(exclusions["sealed_protocol"]["excluded_source_episodes"]) | {row["source_episode"] for row in exclusions["rows"]} | {row["source_episode"] for row in reference["rows"]}
    p0, p0_path = interpolation.load_historical_p0(args.reference, ref)
    bootstrap_mean = p0._bootstrap_mean  # Resolve scalar API before any GPU work.
    modules = {"auditor": Path(__file__), "support": Path(support.__file__), "historical_p0": p0_path,
        "context": Path(execution.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename)}
    if any(digest(modules[key]) != ref["implementation_sha256"][key] for key in ("context", "source_context")):
        raise ValueError("historical execution/source context changed")
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"]:
        raise ValueError("reference checkpoint/statistics changed")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if source_sha != ref["source_sha256"]:
        raise ValueError("source receipt changed")
    with h5py.File(dataset, "r", swmr=True) as handle:
        anchors = support.source_anchors(np.asarray(handle["ep_len"]), np.asarray(handle["ep_offset"]), excluded=excluded,
            count=args.index_size, seed=args.index_seed)
    output = args.output.resolve()
    sidecars = [output.with_suffix(ext) for ext in (".protocol.json", ".implementation", ".roots", ".index.npz", ".banks.json")]
    if output.exists() or any(path.exists() for path in sidecars):
        raise FileExistsError("new immutable output and sidecars required")
    protocol = {"root_seeds": sorted(ref["root_seeds"]), "reference_sha256": digest(args.reference),
        "exclusion_reference_sha256": digest(args.exclusion_reference), "excluded_source_episodes": sorted(excluded),
        "anchors": anchors, "index_seed": args.index_seed, "proposals": args.proposals,
        "query": "all three ordered frozen RGB embeddings, no qpos, goal or successor labels",
        "banks": "original byte-exact neutral control plus31 retrieved vs same control plus31 random, SAME4096-episode index, all candidates executed",
        "extra_control": "literal raw-zero measured separately; never replace historical near-zero action bytes",
        "weights_sha256": ref["weights_sha256"], "adapter_sha256": ref["adapter_sha256"], "source_sha256": source_sha,
        "raw_bridge_atol": 1e-6, "all_banks_sealed_before_any_successor": True,
        "implementation_sha256": {key: digest(path) for key, path in modules.items()},
        "data_verification": "Prior signed full-source SHA receipt reused; no new full-archive SHA claimed.",
        "scope": "Known-root paired proposal-support diagnosis only, not P1, new-root efficacy, or theory rejection."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".implementation").mkdir()
    output.with_suffix(".roots").mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / f"{key}.py")
    print("Testing proposal/state coupling, same frozen4096-source index and32-candidate budget, ALL32 old roots retained. No model training or CPU neural fallback.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    features, packets = [], []
    with h5py.File(dataset, "r", swmr=True) as handle:
        for start in range(0, len(anchors), 32):
            batch = anchors[start:start + 32]
            pixels = np.stack([np.asarray(handle["pixels"][[anchor["root_row"] - 10, anchor["root_row"] - 5, anchor["root_row"]]], np.uint8) for anchor in batch])
            history = np.stack([np.asarray(handle["action"][anchor["root_row"] - 15:anchor["root_row"]], np.float32).reshape(3, 10) for anchor in batch])
            prepared = processor._prepare_info({"pixels": pixels, "action": history})
            features.append(p0._latent_context(model, prepared["pixels"].to(device), prepared["action"].to(device)))
            packets.extend(np.asarray(handle["action"][anchor["root_row"]:anchor["root_row"] + 5], np.float32) for anchor in batch)
            if start % 512 == 0:
                print(json.dumps({"encoded_index_contexts": start + len(batch), "index_size": len(anchors)}), flush=True)
    index, packets = torch.cat(features), np.stack(packets)
    with output.with_suffix(".index.npz").open("xb") as handle:
        np.savez_compressed(handle, embeddings=index.cpu().numpy(), raw_packets=packets)
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"]))
    rows, planned, contexts = [], {}, {}
    try:
        # Seal every root's bank before measuring ANY counterfactual successor.
        for original in sorted(reference["rows"], key=lambda row: row["seed"]):
            seed = original["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            identity = original["methods"]["zero"]["execution_identity"]["initial"]
            if any(context[key] != original[key] for key in ("source_episode", "source_start_row")) or context["root_rgb_sha256"] != identity["reset_rgb_sha256"] or context["goal_rgb_sha256"] != identity["goal_rgb_sha256"]:
                raise RuntimeError("reference root changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            contexts[seed] = context
            pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)
            nearest, distances = support.nearest_context_indices(p0._latent_context(model, pixels, history)[0], index, count=args.proposals - 1)
            random = np.random.default_rng(np.random.SeedSequence([seed, args.index_seed])).choice(args.index_size, args.proposals - 1, replace=False)
            planned[seed] = {"retrieved": nearest.cpu().tolist(), "random": random.tolist(),
                "retrieved_distances": distances[nearest].cpu().tolist(), "random_distances": distances[torch.as_tensor(random, device=device)].cpu().tolist()}
        with output.with_suffix(".banks.json").open("x") as handle:
            handle.write(json.dumps(planned, indent=2, sort_keys=True) + "\n")
        print("All root banks sealed before physical successors. Evaluating every candidate in both banks.", flush=True)
        for original in sorted(reference["rows"], key=lambda row: row["seed"]):
            seed, context = original["seed"], contexts[original["seed"]]
            pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)
            emb, goal_latent = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)
            cache, banks, gates = {}, {}, {}
            for name in ("random", "retrieved"):
                candidates = [(-1, np.asarray(original["methods"]["zero"]["first_action_witness_raw"], np.float32)),
                    (-2, np.zeros((5, 2), np.float32))] + [(index_, packets[index_]) for index_ in planned[seed][name]]
                cells = []
                for index_, raw in candidates:
                    if index_ not in cache:
                        normalized = (raw.reshape(1, 1, 10) - mean) / std
                        token = torch.as_tensor(normalized, device=device, dtype=history.dtype)
                        with torch.no_grad():
                            predicted = model.predict(emb, model.action_encoder(torch.cat((history[:, 1:], token), dim=1)))[0, -1]
                        root, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                        p0._verify_execution_identity(root, goal_rgb, context["root_frame"], context["goal"])
                        advanced = execution.advance_executed_context(env, context=context, raw_action_block=raw, action_block=5, verified_root_frame=root)
                        next_pixels, _, next_history = p0._prepare(processor, advanced["frames"], advanced["root_frame"], advanced["goal"], np.asarray(advanced["history_action_blocks"], np.float32), device)
                        observed = p0._latent_context(model, next_pixels, next_history)[0, -1]
                        bridge = p0._normalized_blocks_to_raw_actions(normalized.reshape(1, 10), mean, std, action_dim=2)
                        cell = {"index": index_, "predicted_cost": float((predicted - goal_latent).square().mean()),
                            "observed_cost": float((observed - goal_latent).square().mean()), "latent_l2_error": float((predicted - observed).norm()),
                            "raw_bridge_max_error": float(np.max(np.abs(bridge - raw))), "rgb_sha256": advanced["root_rgb_sha256"],
                            "action_sha256": advanced["advanced_action_sha256"]}
                        gates[f"candidate_{index_}_raw_bridge"] = cell["raw_bridge_max_error"] <= protocol["raw_bridge_atol"]
                        gates[f"candidate_{index_}_finite"] = bool(torch.isfinite(predicted).all() and torch.isfinite(observed).all())
                        gates[f"candidate_{index_}_bounded"] = bool(np.all(raw >= env.action_space.low - 1e-6) and np.all(raw <= env.action_space.high + 1e-6))
                        cache[index_] = cell
                    if index_ != -2:
                        cells.append(cache[index_])
                banks[name] = {"cells": cells, "metrics": support.ranking_metrics([cell["predicted_cost"] for cell in cells], [cell["observed_cost"] for cell in cells])}
            old_zero = original["methods"]["zero"]
            gates["original_zero_successor_exact"] = cache[-1]["rgb_sha256"] == old_zero["observed_successor_context"]["root_rgb_sha256"] and cache[-1]["observed_cost"] == old_zero["observed_first_chunk"]["candidate_goal_cost"]
            row = {"seed": seed, "source_episode": original["source_episode"], "source_start_row": original["source_start_row"],
                "banks": banks, "literal_zero_control": cache[-2], "gates": gates}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "all_gates": all(gates.values()), "metrics": {name: bank["metrics"] for name, bank in banks.items()}}), flush=True)
            if not all(gates.values()):
                raise RuntimeError("candidate integrity failed, partial evidence retained")
    finally:
        env.close()
    aggregate = aggregate_banks(rows, seed=args.index_seed, bootstrap_mean=bootstrap_mean)
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_reference_roots_retained": [row["seed"] for row in rows] == sorted(ref["root_seeds"]),
        "all_candidate_integrity": all(all(row["gates"].values()) for row in rows),
        "index_episodes_disjoint": not ({anchor["source_episode"] for anchor in anchors} & excluded),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in modules.items())}
    result = {"schema": "reacher_d2p_rgb_conditioned_packets.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "aggregate": aggregate, "rows": rows,
        "index_sha256": digest(output.with_suffix(".index.npz")), "banks_sha256": digest(output.with_suffix(".banks.json")), "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
