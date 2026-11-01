"""Isolate strict prefix replay identity before running response probes.

All eight roots and eight branch actions are replayed twice, independently
restoring the same exact initial state. Differences are findings, never a
reason to loosen RGB identity or erase a failed audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_goal_contract as goals
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import analyze_reacher_d2p_waypoint_tracking as upstream
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream.load(args.reference)
    report = json.loads(args.reference.read_text())
    plans = {r["seed"]: r for r in json.loads(args.reference.with_suffix(".selection.json").read_text())["rows"]}
    ref = report["sealed_protocol"]
    if args.output.exists() or args.output.with_suffix(".protocol.json").exists():
        raise FileExistsError("new immutable replay audit required")
    dataset, source_sha = shared.receipt_source(args.adapter_artifact)
    if digest(args.policy / "weights.pt") != ref["weights_sha256"] or digest(args.adapter_artifact) != ref["adapter_sha256"] or source_sha != ref["source_sha256"]:
        raise ValueError("same official frozen model/source required")
    modules = {"auditor": Path(__file__), "context": Path(execution.__file__), "p0": Path(p0.__file__)}
    protocol = {"reference_sha256": digest(args.reference), "root_seeds": ref["root_seeds"], "repeats": 2, "prefix_packets": 4,
        "implementation_sha256": {k: digest(v) for k, v in modules.items()}, "weights_sha256": ref["weights_sha256"],
        "scope": "Known-root strict replay diagnosis; historical matches are findings, not P1 success or tolerance changes."}
    write_new(args.output.with_suffix(".protocol.json"), protocol)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    model = shared.load_compatible_pretrained(args.policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    env = goals.goal_wrapper("joint-feasible-v1")(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(ref["target_delta"]))
    rows = []
    print("HOST CUDA: replay every fixed prefix twice; preserve hashes, complete physical states and all mismatches without tolerances.", flush=True)
    try:
        for old in sorted(report["rows"], key=lambda r: r["seed"]):
            seed = old["seed"]
            ctx = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(ctx[k] != plans[seed][k] for k in ("source_episode", "source_start_row", "root_rgb_sha256", "goal_rgb_sha256")):
                raise RuntimeError("initial source/root identities changed")
            ctx["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(ctx["root_frame"])
            cells = {}
            for name, branch in old["methods"].items():
                repeats = []
                for repeat in range(2):
                    execution.restore_execution_snapshot(env, seed=seed, snapshot=ctx["execution_snapshot"], root_identity="sealed_observation_exact_state")
                    current, packets = ctx, []
                    raw = np.asarray(branch["action_witness_raw"], np.float32)
                    for i in range(4):
                        current = execution.advance_executed_context(env, context=current, raw_action_block=raw[i * 5:(i + 1) * 5], action_block=5, verified_root_frame=current["root_frame"])
                        current["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(current["root_frame"])
                        pixels, _, history = p0._prepare(processor, current["frames"], current["root_frame"], current["goal"], np.asarray(current["history_action_blocks"], np.float32), torch.device("cuda"))
                        with torch.no_grad():
                            actual = p0._latent_context(model, pixels, history)[0, -1]
                        previous = branch["packets"][i]
                        packets.append({"packet": i + 1, "root_rgb_sha256": current["root_rgb_sha256"], "history_rgb_sha256": current["history_rgb_sha256"],
                            "root_matches_history": current["root_rgb_sha256"] == previous["root_rgb_sha256"],
                            "frames_match_history": current["history_rgb_sha256"] == previous["history_rgb_sha256"],
                            "actions_match_history": current["advanced_action_sha256"] == previous["action_sha256"],
                            "vector_matches_history": actual.tolist() == previous["observed_vector"],
                            "vector_max_difference": float((actual - torch.tensor(previous["observed_vector"], device="cuda")).abs().max()),
                            "snapshot": current["execution_snapshot"]})
                    repeats.append({"repeat": repeat, "packets": packets})
                cells[name] = repeats
            rows.append({"seed": seed, "methods": cells})
            print(json.dumps({"seed": seed, "mismatched_packets": sum(not p["root_matches_history"] for v in cells.values() for r in v for p in r["packets"])}), flush=True)
    finally:
        env.close()
    failures = [{"seed": r["seed"], "method": name, "repeat": rep["repeat"], "packet": p["packet"],
        "root_matches_history": p["root_matches_history"], "frames_match_history": p["frames_match_history"],
        "actions_match_history": p["actions_match_history"], "vector_max_difference": p["vector_max_difference"]}
        for r in rows for name, reps in r["methods"].items() for rep in reps for p in rep["packets"]
        if not (p["root_matches_history"] and p["frames_match_history"] and p["actions_match_history"] and p["vector_matches_history"])]
    repeat_differences = []
    for r in rows:
        for name, repeats in r["methods"].items():
            for left, right in zip(repeats[0]["packets"], repeats[1]["packets"]):
                if left["root_rgb_sha256"] != right["root_rgb_sha256"] or left["snapshot"] != right["snapshot"]:
                    repeat_differences.append({"seed": r["seed"], "method": name, "packet": left["packet"],
                        "render_exact": left["root_rgb_sha256"] == right["root_rgb_sha256"], "snapshot_exact": left["snapshot"] == right["snapshot"]})
    gates = {"cuda_only": True, "all8_roots_and512_boundaries": len(rows) == 8 and all(len(r["methods"]) == 8 for r in rows),
        "parameters_unchanged": shared._parameter_hash(model) == before, "buffers_unchanged": all(torch.equal(v, buffers[k]) for k, v in model.named_buffers()),
        "implementation_unchanged": all(digest(v) == protocol["implementation_sha256"][k] for k, v in modules.items()), "reference_unchanged": digest(args.reference) == protocol["reference_sha256"]}
    write_new(args.output, {"schema": "reacher_d2p_prefix_replay_stability.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "gates": gates, "historical_mismatches": failures, "repeat_differences": repeat_differences, "rows": rows})
    print(json.dumps({"gates": gates, "historical_mismatches": failures, "repeat_differences": repeat_differences}), flush=True)


if __name__ == "__main__":
    main()
