"""Fresh episode-disjoint host-CUDA test of model-only program dispatch.

This is a separately sealed runtime diagnostic, not a P1 gate replacement.
All roots execute dispatch, greedy, zero, and three-packet CEM-lookahead
controls. Every branch reobserves RGB after five raw steps and uses the same
bounded-DLS suffix. CEM is not claimed fully compute-matched: VJPs differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import audit_reacher_d2p_source_supported_p1 as source
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import reacher_d2p_context as context_runtime
from experiments.spectral_reachability import reacher_d2p_program_dispatch as dispatch
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
from experiments.spectral_reachability.reacher_d2p_packet_support import SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def planned_source_roots(dataset: Path, seeds: list[int], excluded: set[int]) -> list[dict]:
    """Check source episode admission using metadata only, before inference."""
    with h5py.File(dataset, "r", swmr=True) as handle:
        lengths = np.asarray(handle["ep_len"], dtype=np.int64)
        offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
    eligible = np.flatnonzero(lengths >= 20)
    if not len(eligible) or len(seeds) != len(set(seeds)):
        raise ValueError("unique root seeds and eligible source histories required")
    roots, seen = [], set()
    for seed in seeds:
        rng = np.random.default_rng(int(seed) ^ 0x5A7E)
        episode = int(eligible[int(rng.integers(len(eligible)))])
        start = int(offsets[episode]) + int(rng.integers(0, int(lengths[episode]) - 20 + 1))
        if episode in excluded or episode in seen:
            raise ValueError("source-episode collision; no outcome-dependent resampling permitted")
        seen.add(episode)
        roots.append({"seed": int(seed), "source_episode": episode, "source_start_row": start})
    return roots


def root_execution_gates(methods: dict) -> dict:
    decision = methods["dispatch"]["first_receipt"]["dispatch"]
    greedy = methods["same_bank_greedy"]
    routed = methods["dispatch"]
    identities = [method["execution_identity"]["initial"] for method in methods.values()]
    fallback = not decision["reconfiguration_witness_found"]
    return {
        "same_root_goal_for_all_methods": all(identity == identities[0] for identity in identities),
        "same_bank_greedy_identity": decision["same_bank_greedy_index"] == greedy["first_receipt"]["selected_proposal_index"],
        "physical_labels_not_used_for_dispatch": decision["physical_labels_used"] is False,
        "fallback_executes_exact_greedy_actions": not fallback or routed["action_witness_raw"] == greedy["action_witness_raw"],
        "fallback_replays_exact_greedy_final_distance": not fallback or routed["physical"]["final_qpos_distance"] == greedy["physical"]["final_qpos_distance"],
        "all_distances_finite": all(np.isfinite(method["physical"]["final_qpos_distance"]) for method in methods.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--disjoint-reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=202626000)
    parser.add_argument("--roots", type=int, default=32)
    args = parser.parse_args()
    if args.roots < 4:
        raise ValueError("at least four fixed roots required")
    policy, adapter, output = args.policy.resolve(), args.adapter_artifact.resolve(), args.output.resolve()
    paths = [output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"), output.with_suffix(".implementation")]
    if any(path.exists() for path in paths):
        raise FileExistsError("new immutable output and sidecar paths required")
    seeds = list(range(args.seed, args.seed + args.roots))
    excluded, bindings = source._disjoint_reference_binding(args.disjoint_reference, set(seeds))
    dataset, source_hash = shared.receipt_source(adapter)
    roots = planned_source_roots(dataset, seeds, excluded)
    implementations = {"auditor": Path(__file__), "dispatch": Path(dispatch.__file__),
        "selector": Path(p1.__file__), "runner": Path(runner.__file__), "context": Path(context_runtime.__file__),
        "source_context": Path(shared._source_state_context.__code__.co_filename), "p0": Path(p0.__file__)}
    sealed = {"root_seeds": seeds, "source_roots": roots, "excluded_source_episodes": sorted(excluded),
        "disjoint_reference_bindings": bindings, "policy": dispatch.POLICY,
        "proposals": 128, "damping": .01, "reconfiguration_horizon": 1, "suffix_horizon": 2,
        "non_greedy_margin": 1e-6, "reachability_margin": 0., "future_reference": "same_bank_greedy",
        "recompiled_suffix_solver": p1.SUFFIX_SOLVER, "target_delta": [.65, -.65],
        "selection_goal_coordinate": "frozen_encoder_l2", "trained_modules": False,
        "cem": {"horizon": 3, "samples": 128, "steps": 5, "elite": 16,
            "execution": "first packet only, observe, same DLS suffix", "full_compute_matched": False},
        "weights_sha256": _digest(policy / "weights.pt"), "adapter_sha256": _digest(adapter),
        "source_sha256": source_hash, "implementation_sha256": {key: _digest(path) for key, path in implementations.items()},
        "claim_boundary": "Fresh source-episode-disjoint runtime diagnostic, not formal P1, calibration/outcome, or full compute-matched efficacy."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    output.with_suffix(".implementation").mkdir()
    for key, path in implementations.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / f"{key}.py")
    print("Testing fixed model-only progress fallback on every declared new root; CUDA only; not formal P1.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(adapter, "swm/ReacherDMControl-v0")
    print("Verifying complete official source SHA256 and loading finite packet support.", flush=True)
    pool, support = shared.load_packet_pool(dataset, expected_sha256=source_hash)
    model = shared.load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameter_hash = p1._parameter_hash(model)
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
    lower_np, upper_np = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lower_np, device=device), torch.as_tensor(upper_np, device=device)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    calls = samples = 0
    predict = model.predict

    def counted_predict(embeddings, actions):
        nonlocal calls, samples
        if embeddings.device.type != "cuda" or actions.device.type != "cuda":
            raise RuntimeError("CPU fallback forbidden")
        calls += 1
        samples += int(embeddings.shape[0])
        return predict(embeddings, actions)

    model.predict = counted_predict
    rows = []
    try:
        for root in roots:
            seed = root["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            if any(context[key] != root[key] for key in ("source_episode", "source_start_row")):
                raise RuntimeError("live source root differs from pre-inference plan")
            bank, bank_support = shared.proposal_parameters(pool, proposals=128, horizon=1, seed=seed,
                mean=mean, std=std, lower=lower, upper=upper)

            def dispatched(pixels, goal, history, root_seed):
                return dispatch.compile_program_dispatch(model, pixels, goal, history, proposal_bank=bank,
                    reconfiguration_horizon=1, future_horizon=2, action_lower=lower, action_upper=upper,
                    damping=.01, proposals=128, non_greedy_margin=1e-6, reachability_margin=0., seed=root_seed,
                    future_reference="same_bank_greedy", proposal_distribution=SOURCE_SUPPORTED_PROPOSAL_DISTRIBUTION)

            def greedy(pixels, goal, history, _seed):
                return runner._source_matched_greedy_program(model, pixels, goal, history, proposal_bank=bank,
                    action_lower=lower, action_upper=upper)

            def zero(_pixels, _goal, history, _seed):
                return torch.zeros((1, 1, history.shape[-1]), device=device), {"program_type": "zero_prefix_control", "backward_vjp_rows": 0}

            def cem(pixels, goal, history, root_seed):
                params, receipt = p0._cem(model, pixels, goal, history, future_horizon=3,
                    action_lower=lower, action_upper=upper, samples=128, steps=5, elite=16, seed=root_seed)
                receipt.update({"program_type": "three_packet_lookahead_cem_first_packet", "horizon": 3,
                    "model_predict_sample_evaluations": 1920, "full_compute_matched": False})
                return params[:, :1].detach(), receipt

            methods = {}
            for name, program in (("dispatch", dispatched), ("same_bank_greedy", greedy), ("zero", zero), ("lookahead_cem", cem)):
                before_calls, before_samples = calls, samples
                torch.cuda.synchronize()
                method = runner._two_stage(name=name, first_program=program, env=env, seed=seed, context=context,
                    preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                    action_block=5, progress_horizon=2, damping=.01, device=device)
                torch.cuda.synchronize()
                method["actual_compute"] = {"model_predict_calls": calls - before_calls,
                    "model_predict_sample_evaluations": samples - before_samples,
                    "logical_backward_vjp_rows": method["first_receipt"].get("backward_vjp_rows", 0) + method["suffix_receipt"]["backward_vjp_rows"],
                    "full_compute_matched": False}
                methods[name] = method
            row = {**root, "methods": methods, "proposal_support": bank_support, "gates": root_execution_gates(methods)}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "route": methods["dispatch"]["first_receipt"]["dispatch"]["route"],
                "final_distance": {key: value["physical"]["final_qpos_distance"] for key, value in methods.items()},
                "gates": row["gates"]}), flush=True)
            if not all(row["gates"].values()):
                raise RuntimeError("physical dispatch identity/replay contract failed; partial root retained")
    finally:
        model.predict = predict
        env.close()
    gates = {"cuda_only": True, "weights_unchanged": parameter_hash == p1._parameter_hash(model),
        "all_roots_retained": [row["seed"] for row in rows] == seeds,
        "all_source_episodes_disjoint": not ({row["source_episode"] for row in rows} & excluded),
        "implementation_unchanged": all(_digest(path) == sealed["implementation_sha256"][key] for key, path in implementations.items()),
        "all_execution_gates": all(all(row["gates"].values()) for row in rows)}
    result = {"schema": "reacher_d2p_program_dispatch_diagnostic.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": sealed, "gates": gates, "source_packet_support": support,
        "aggregate": {"reconfiguration_count": sum(row["methods"]["dispatch"]["first_receipt"]["dispatch"]["reconfiguration_witness_found"] for row in rows),
            "paired": {name: runner._paired(rows, "dispatch", name, args.seed + 711) for name in ("same_bank_greedy", "zero", "lookahead_cem")}},
        "rows": rows, "claim_boundary": sealed["claim_boundary"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": result["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
