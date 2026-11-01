"""Same-root CUDA audit of raw recurrence and strictly causal innovation.

Replays all four recorded first-action branches on every declared source root.
The unit innovation is computed from predecessor RGB and executed actions
BEFORE accessing any candidate successor. It is a diagnostic alternative,
not a new P1 selector, an efficacy receipt, or a learned model repair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import audit_reacher_d2p_p1_first_chunk_fidelity as fidelity
from experiments.spectral_reachability import reacher_d2p_causal_innovation as innovation_rule
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import reacher_d2p_packet_support as packets
from experiments.spectral_reachability import reacher_d2p_margin_decomposition as decomposition
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


METHODS = ("dispatch", "same_bank_greedy", "zero", "lookahead_cem")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows):
    if not rows:
        raise ValueError("completed root rows required")
    result = {}
    for name, subset in (("all_roots", rows), ("model_eligible_roots", [row for row in rows if row["model_eligible"]])):
        comparisons = [{"raw": row["methods"]["dispatch"]["raw_predicted_goal_cost"] - row["methods"]["zero"]["raw_predicted_goal_cost"],
            "corrected": row["methods"]["dispatch"]["corrected_predicted_goal_cost"] - row["methods"]["zero"]["corrected_predicted_goal_cost"],
            "observed": row["methods"]["dispatch"]["observed_goal_cost"] - row["methods"]["zero"]["observed_goal_cost"]} for row in subset]
        result[name] = {"count": len(subset),
            "raw_strict_order_agreements_vs_zero": sum(row["raw"] * row["observed"] > 0 for row in comparisons),
            "corrected_strict_order_agreements_vs_zero": sum(row["corrected"] * row["observed"] > 0 for row in comparisons),
            "raw_non_greedy_predictions": sum(row["raw"] >= 0 for row in comparisons),
            "corrected_non_greedy_predictions": sum(row["corrected"] >= 0 for row in comparisons),
            "observed_non_greedy": sum(row["observed"] >= 0 for row in comparisons)}
    branches = [method for row in rows for method in row["methods"].values()]
    result["all_branch_prediction_error"] = {"count": len(branches),
        "mean_raw_latent_l2": float(np.mean([cell["raw_latent_l2_error"] for cell in branches])),
        "mean_corrected_latent_l2": float(np.mean([cell["corrected_latent_l2_error"] for cell in branches])),
        "correction_reduces_error_count": sum(cell["corrected_latent_l2_error"] < cell["raw_latent_l2_error"] for cell in branches)}
    return result


def validate_reference(report):
    if report.get("schema") != "reacher_d2p_program_dispatch_diagnostic.v1" or report.get("status") != "completed":
        raise ValueError("completed original dispatch diagnostic required")
    if not report.get("gates") or not all(report["gates"].values()):
        raise ValueError("reference integrity gates failed")
    seeds = [row["seed"] for row in report["rows"]]
    if not seeds or len(set(seeds)) != len(seeds) or sorted(seeds) != sorted(report["sealed_protocol"]["root_seeds"]):
        raise ValueError("complete unique sealed root identities required")
    if any(set(row["methods"]) != set(METHODS) for row in report["rows"]):
        raise ValueError("all four fixed physical branch witnesses required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.reference.read_text())
    validate_reference(report)
    reference = report["sealed_protocol"]
    output = args.output.resolve()
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"), output.with_suffix(".implementation"))):
        raise FileExistsError("new immutable output and sidecars required")
    if digest(args.policy / "weights.pt") != reference["weights_sha256"] or digest(args.adapter_artifact) != reference["adapter_sha256"]:
        raise ValueError("reference weights or action normalizer differ")
    implementations = {"auditor": Path(__file__), "innovation": Path(innovation_rule.__file__), "fidelity": Path(fidelity.__file__),
        "margin_decomposition": Path(decomposition.__file__),
        "context": Path(execution.__file__), "p0": Path(p0.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename)}
    for key in ("context", "p0", "source_context"):
        if digest(implementations[key]) != reference["implementation_sha256"][key]:
            raise ValueError(f"reference execution implementation changed: {key}")
    sealed = {"root_seeds": reference["root_seeds"], "methods": METHODS,
        "reference_sha256": digest(args.reference), "weights_sha256": reference["weights_sha256"],
        "adapter_sha256": reference["adapter_sha256"], "source_sha256": reference["source_sha256"],
        "innovation_rule": innovation_rule.RULE, "coefficient": 1., "coefficient_fitted": False,
        "recorded_batch_cost_atol": 2e-6, "raw_action_bridge_atol": 1e-6,
        "observed_cost_identity": "exact", "observed_successor_rgb_identity": "exact",
        "implementation_sha256": {key: digest(path) for key, path in implementations.items()},
        "scope": "Known diagnostic roots and fixed action branches, no selector tuning, no efficacy, no gate override."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    output.with_suffix(".implementation").mkdir()
    for key, path in implementations.items():
        shutil.copyfile(path, output.with_suffix(".implementation") / f"{key}.py")
    print("Auditing fixed branches: raw replay and past-only unit innovation. No future-informed coefficient, action selection, or training.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    dataset, source_hash = shared.receipt_source(args.adapter_artifact)
    print("Verifying full source dataset SHA256 before state/history rehydration.", flush=True)
    if source_hash != reference["source_sha256"] or packets._sha256(dataset) != source_hash:
        raise ValueError("official source data changed")
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(reference["target_delta"]))
    rows = []
    try:
        for source in sorted(report["rows"], key=lambda row: row["seed"]):
            seed = source["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            identity = source["methods"]["zero"]["execution_identity"]["initial"]
            if context["root_rgb_sha256"] != identity["reset_rgb_sha256"] or context["goal_rgb_sha256"] != identity["goal_rgb_sha256"]:
                raise RuntimeError("source observed root/goal differ from fixed reference")
            if any(context[key] != source[key] for key in ("source_episode", "source_start_row")):
                raise RuntimeError("source episode/history window differs")
            pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], np.float32), device)
            prepared_past = processor._prepare_info({"pixels": np.asarray(context["frames"], np.uint8)[None],
                "goal": context["goal"][None, None], "action": np.asarray(context["history_action_blocks"], np.float32)[None]})
            root_latent = p0._latent_context(model, pixels, history)[0, -1]
            goal_latent = p0._goal_latent(model, goal)
            # This is computed once, before executing ANY branch successor.
            innovation, innovation_receipt = innovation_rule.innovation_from_executed_history(model,
                prepared_past["pixels"].to(device), prepared_past["action"].to(device), root_latent)
            methods, latent_vectors = {}, {}
            compiler_receipt = source["methods"]["dispatch"]["first_receipt"]["reconfiguration_compilation"]
            decision = source["methods"]["dispatch"]["first_receipt"]["dispatch"]
            for name in METHODS:
                original = source["methods"][name]
                raw = np.asarray(original["first_action_witness_raw"], np.float32)
                normalized = np.asarray(original["first_action_witness_normalized"], np.float32)
                predicted = fidelity._first_successor_prediction(model, pixels, history,
                    torch.as_tensor(normalized, device=device, dtype=history.dtype))
                corrected = innovation_rule.corrected_prediction(predicted, innovation)
                restored_root, restored_goal = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
                p0._verify_execution_identity(restored_root, restored_goal, context["root_frame"], context["goal"])
                advanced = execution.advance_executed_context(env, context=context, raw_action_block=raw,
                    action_block=5, verified_root_frame=restored_root)
                next_pixels, _, next_history = p0._prepare(processor, advanced["frames"], advanced["root_frame"], advanced["goal"],
                    np.asarray(advanced["history_action_blocks"], np.float32), device)
                observed = p0._latent_context(model, next_pixels, next_history)[0, -1]
                latent_vectors[name] = {"predicted": predicted.detach(), "observed": observed.detach()}
                cost = lambda value: float((value - goal_latent).square().mean())
                bridge_raw = p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)
                expected_index = {"dispatch": decision["selected_program_index"], "same_bank_greedy": decision["same_bank_greedy_index"], "zero": 0}.get(name)
                recorded_model_cost = compiler_receipt["proposal_rows"][expected_index]["immediate_goal_cost"] if expected_index is not None else None
                observed_cost = cost(observed)
                methods[name] = {"raw_predicted_goal_cost": cost(predicted), "corrected_predicted_goal_cost": cost(corrected),
                    "observed_goal_cost": observed_cost, "raw_latent_l2_error": float(torch.linalg.vector_norm(predicted - observed)),
                    "corrected_latent_l2_error": float(torch.linalg.vector_norm(corrected - observed)),
                    "raw_action_bridge_max_error": float(np.max(np.abs(bridge_raw - raw))),
                    "recorded_model_cost_abs_error": None if recorded_model_cost is None else abs(cost(predicted) - recorded_model_cost),
                    "recorded_observed_cost_abs_error": abs(observed_cost - original["observed_first_chunk"]["candidate_goal_cost"]),
                    "successor_rgb_identity_exact": advanced["root_rgb_sha256"] == original["observed_successor_context"]["root_rgb_sha256"],
                    "successor_actions_identity_exact": advanced["advanced_action_sha256"] == original["observed_successor_context"]["advanced_action_sha256"]}
            comparisons = {name: decomposition.decompose_margin(latent_vectors[name]["predicted"],
                latent_vectors["zero"]["predicted"], latent_vectors[name]["observed"],
                latent_vectors["zero"]["observed"], goal_latent) for name in METHODS if name != "zero"}
            row = {"seed": seed, "source_episode": source["source_episode"], "model_eligible": decision["reconfiguration_witness_found"],
                "innovation": innovation_receipt, "innovation_vector": innovation.cpu().tolist(), "methods": methods,
                "margin_decomposition_vs_zero": comparisons,
                "latent_vectors": {name: {key: value.cpu().tolist() for key, value in vectors.items()} for name, vectors in latent_vectors.items()},
                "goal_latent": goal_latent.cpu().tolist()}
            row["gates"] = {"all_recorded_observations_exact": all(cell["successor_rgb_identity_exact"] and cell["successor_actions_identity_exact"] and cell["recorded_observed_cost_abs_error"] == 0 for cell in methods.values()),
                "raw_model_replays_recorded_scores": all(cell["recorded_model_cost_abs_error"] is None or cell["recorded_model_cost_abs_error"] <= sealed["recorded_batch_cost_atol"] for cell in methods.values()),
                "raw_action_bridge_valid": all(cell["raw_action_bridge_max_error"] <= sealed["raw_action_bridge_atol"] for cell in methods.values())}
            row["gates"]["margin_decomposition_identity"] = all(cell["decomposition_abs_error"] <= 1e-12 for cell in comparisons.values())
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            raw_margin = methods["dispatch"]["raw_predicted_goal_cost"] - methods["zero"]["raw_predicted_goal_cost"]
            corrected_margin = methods["dispatch"]["corrected_predicted_goal_cost"] - methods["zero"]["corrected_predicted_goal_cost"]
            observed_margin = methods["dispatch"]["observed_goal_cost"] - methods["zero"]["observed_goal_cost"]
            print(json.dumps({"seed": seed, "eligible": row["model_eligible"], "gates": row["gates"],
                "raw_margin": raw_margin, "corrected_margin": corrected_margin, "observed_margin": observed_margin}), flush=True)
    finally:
        env.close()
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_reference_roots_audited": sorted(row["seed"] for row in rows) == sorted(reference["root_seeds"]),
        "all_replay_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == sealed["implementation_sha256"][key] for key, path in implementations.items())}
    result = {"schema": "reacher_d2p_causal_innovation_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": sealed, "aggregate": summarize(rows), "rows": rows,
        "claim_boundary": "Known-root fixed-action diagnostic. Causal correction is not a raw-model accuracy claim, selector, fresh efficacy, or P1 override."}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": result["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
