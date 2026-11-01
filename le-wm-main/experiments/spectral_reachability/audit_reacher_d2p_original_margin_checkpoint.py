"""Replay ALL six original margin roots across checkpoints and CUDA precision.

Fixed root RGB, goals and actions come from immutable P1 evidence. There is no
new selector, fitted correction, training, threshold search, or P1 admission.
This isolates checkpoint effects and floating-point score/forward effects.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import audit_reacher_d2p_p1_first_chunk_fidelity as fidelity
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability.reacher_d2p_margin_decomposition import decompose_margin
from experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract import digest
from experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity import write_new


def validate_reference(report, prior):
    fidelity._validate_report(report)
    if (report["protocol"].get("warmup_policy") != execution.WARMUP_POLICY
            or report["protocol"].get("context_support", "uniform") != "uniform"
            or fidelity._receipt_qpos_binding(report)["enabled"]):
        raise ValueError("original uniform raw-latent protocol required")
    seeds = [r["seed"] for r in report["rows"]]
    if (len(seeds) != 6 or len(set(seeds)) != 6 or sorted(seeds) != sorted(r["seed"] for r in prior.get("rows", []))
            or len(prior["rows"]) != 6 or prior.get("status") != "completed"
            or not prior.get("gates") or not all(v is True for v in prior["gates"].values())):
        raise ValueError("all six unique roots and strict historical replay gates required")
    for row in report["rows"]:
        method = row["methods"]["reconfiguration"]
        if method["first_receipt"].get("non_greedy_witness_found") is not True:
            raise ValueError("do not silently skip non-witness roots")
        for key in ("first_action_witness_raw", "zero_prefix_action_witness_raw"):
            value = np.asarray(method[key])
            if value.shape != (5, 2) or not np.isfinite(value).all() or np.abs(value).max() > 1:
                raise ValueError("bounded sealed five-step packet required")
    return sorted(seeds)


def summarize(rows):
    if len(rows) != 6 or len({r["seed"] for r in rows}) != 6:
        raise ValueError("all six root identities required")
    result = {}
    for name in ("receipt_checkpoint", "official_checkpoint"):
        for precision in ("float32", "float64"):
            cells = [r["checkpoints"][name][precision] for r in rows]
            result[name + "/" + precision] = {
                "order_agreements": sum(c["decomposition"]["predicted_margin"] * c["decomposition"]["observed_margin"] > 0 for c in cells),
                "reversed_roots": [r["seed"] for r, c in zip(rows, cells) if c["decomposition"]["predicted_margin"] * c["decomposition"]["observed_margin"] < 0],
                "mean_candidate_latent_l2": float(np.mean([c["candidate_latent_l2"] for c in cells])),
                "mean_zero_latent_l2": float(np.mean([c["zero_latent_l2"] for c in cells]))}
    return result


def dtype_preserving_action_embedding(embedder, actions):
    """Same public Embedder algebra, without its unconditional .float().

    The native float32 path must agree BIT-FOR-BIT before this independent
    double-precision diagnostic is accepted. No deployment code is replaced.
    """
    return embedder.embed(embedder.patch_embed(actions.permute(0, 2, 1)).permute(0, 2, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--receipt-policy", type=Path, required=True)
    parser.add_argument("--official-policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The loader treats relative paths as model cache names, not cwd paths.
    args.receipt_policy = args.receipt_policy.resolve()
    args.official_policy = args.official_policy.resolve()
    report, prior = (json.loads(p.read_text()) for p in (args.reference, args.fidelity))
    seeds = validate_reference(report, prior)
    fidelity._validate_runtime_inputs(report, policy=args.receipt_policy, adapter=args.adapter_artifact)
    if prior["source_report_sha256"] != digest(args.reference):
        raise ValueError("historical fidelity must bind this exact source")
    protocol_path = args.output.with_suffix(".protocol.json")
    impl = args.output.with_suffix(".implementation")
    if any(p.exists() for p in (args.output, protocol_path, impl)):
        raise FileExistsError("immutable new outputs required")
    modules = {"auditor": Path(__file__), "fidelity": Path(fidelity.__file__), "context": Path(execution.__file__),
        "rollout": Path(p0.__file__), "loader": Path(shared.load_compatible_pretrained.__code__.co_filename),
        "goal_transaction": Path(shared.ReacherGoalImageWrapper._render_target_and_restore.__code__.co_filename),
        "decomposition": Path(decompose_margin.__code__.co_filename)}
    inputs = {"reference": args.reference, "prior_fidelity": args.fidelity, "adapter": args.adapter_artifact,
        "receipt_weights": args.receipt_policy / "weights.pt", "official_weights": args.official_policy / "weights.pt",
        "receipt_config": args.receipt_policy / "config.json", "official_config": args.official_policy / "config.json"}
    protocol = {"root_seeds": seeds, "target_delta": list(fidelity._target_delta_from_report(report)),
        "inputs": {k: {"path": str(p.resolve()), "sha256": digest(p)} for k, p in inputs.items()},
        "implementation_sha256": {k: digest(p) for k, p in modules.items()},
        "precisions": ["float32", "float64"], "precision_scope": "dynamics-only; identical original float32 encoder/projector chart",
        "action_embedding_reference": "public Embedder Conv1d/MLP algebra minus unconditional float cast; exact FP32 equivalence required",
        "action_selection": "unchanged original sealed witnesses, no new selection",
        "observed_successor_used_for_planning": False, "physical_labels_used_for_planning": False,
        "scope": "Known-root checkpoint/precision diagnostic only, not independent calibration, efficacy, or theory verdict."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(protocol_path, protocol)
    impl.mkdir()
    for key, path in modules.items():
        shutil.copyfile(path, impl / (key + ".py"))
    print("Fixed ALL6 original root/goal/actions; compare original and official checkpoints plus FP64 DYNAMICS on HOST CUDA with the same FP32 encoder chart. No root exclusion, training, threshold change or P1 admission.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(protocol["target_delta"]))
    prepared_roots, rows = [], []
    try:
        for old in sorted(report["rows"], key=lambda r: r["seed"]):
            seed, method = old["seed"], old["methods"]["reconfiguration"]
            context = execution.collect_executed_context(env, seed=seed, history_len=3, action_block=5)
            if any(context[k] != old["executed_context"][k] for k in ("root_rgb_sha256", "goal_rgb_sha256", "warmup_actions_sha256")):
                raise RuntimeError("original RGB/action root changed")
            pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), torch.device("cuda"))
            observed, normalized, identities = {}, {}, {}
            for name, prefix in (("candidate", "first"), ("zero", "zero_prefix")):
                raw = np.asarray(method[prefix + "_action_witness_raw"], np.float32)
                norm = np.asarray(method[prefix + "_action_witness_normalized"], np.float32)
                if np.max(np.abs(p0._normalized_blocks_to_raw_actions(norm, mean, std, action_dim=2) - raw)) > 1e-6:
                    raise RuntimeError("recorded action normalization no longer inverts")
                root_frame, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
                p0._verify_execution_identity(root_frame, goal_rgb, context["root_frame"], context["goal"])
                advanced = execution.advance_executed_context(env, context=context, raw_action_block=raw, action_block=5)
                expected = method["observed_successor_context" if name == "candidate" else "zero_prefix_successor_context"]
                if any(advanced[k] != v for k, v in expected.items()):
                    raise RuntimeError("original physical successor RGB no longer exact")
                observed[name] = p0._prepare(processor, advanced["frames"], advanced["root_frame"], advanced["goal"], np.asarray(advanced["history_action_blocks"], np.float32), torch.device("cuda"))
                normalized[name] = torch.as_tensor(norm, device="cuda", dtype=history.dtype)
                identities[name] = {k: advanced[k] for k in expected}
            prepared_roots.append((old, pixels, goal, history, normalized, observed))
            rows.append({"seed": seed, "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"],
                "successor_identities": identities, "checkpoints": {}})
    finally:
        env.close()
    samples, model_gates, historical_gates, embedder_gates = 0, [], [], []
    for name, policy in (("receipt_checkpoint", args.receipt_policy), ("official_checkpoint", args.official_policy)):
        model = shared.load_compatible_pretrained(policy, device="cuda").eval()
        model.requires_grad_(False)
        original_predict = model.predict

        def counted_predict(emb, act):
            nonlocal samples
            if emb.device.type != "cuda" or act.device.type != "cuda":
                raise RuntimeError("CPU neural fallback forbidden")
            samples += int(emb.shape[0])
            return original_predict(emb, act)

        model.predict = counted_predict
        for precision, dtype in (("float32", torch.float32), ("float64", torch.float64)):
            # LeWM.encode explicitly casts pixels to float32. Keep that exact
            # encoder chart and vary ONLY action encoder/predictor/pred_proj;
            # do not claim end-to-end double-precision visual inference.
            for module in (model.action_encoder, model.predictor, model.pred_proj):
                module.to(dtype=dtype)
            before = shared._parameter_hash(model)
            for row, (old, pixels, goal, history, normalized, observed) in zip(rows, prepared_roots):
                goal_latent = p0._goal_latent(model, goal).to(dtype=dtype)
                with torch.no_grad():
                    encoded = model.encode({"pixels": pixels})["emb"].detach().to(dtype=dtype)
                predictions, actual = {}, {}
                for branch in ("candidate", "zero"):
                    action_context = torch.cat([history[:, 1:], normalized[branch][None]], dim=1).to(dtype=dtype)
                    with torch.no_grad():
                        act_emb = dtype_preserving_action_embedding(model.action_encoder, action_context)
                        if precision == "float32":
                            exact = torch.equal(act_emb, model.action_encoder(action_context))
                            embedder_gates.append(exact)
                            if not exact:
                                raise RuntimeError("dtype-preserving action reference differs from original FP32 path")
                        predictions[branch] = model.predict(encoded, act_emb)[0, -1].detach()
                    obs_pixels, _, obs_history = observed[branch]
                    with torch.no_grad():
                        actual[branch] = model.encode({"pixels": obs_pixels})["emb"][0, -1].detach().to(dtype=dtype)
                vectors = {"prediction": predictions["candidate"], "prediction_zero": predictions["zero"],
                    "observed": actual["candidate"], "observed_zero": actual["zero"], "goal": goal_latent}
                costs = {key: float((value - goal_latent).square().mean()) for key, value in vectors.items() if key != "goal"}
                decomposition = decompose_margin(*vectors.values())
                if not np.isfinite(list(costs.values())).all() or decomposition["decomposition_abs_error"] > 1e-11:
                    raise RuntimeError("nonfinite costs or invalid algebraic decomposition")
                cell = {"costs": costs, "vectors": {k: v.tolist() for k, v in vectors.items()}, "decomposition": decomposition,
                    "candidate_latent_l2": float((predictions["candidate"] - actual["candidate"]).norm()),
                    "zero_latent_l2": float((predictions["zero"] - actual["zero"]).norm())}
                row["checkpoints"].setdefault(name, {})[precision] = cell
                if name == "receipt_checkpoint" and precision == "float32":
                    expected = next(r for r in prior["rows"] if r["seed"] == row["seed"])
                    replay = all(abs(costs[k] - expected[b][f]) <= 2e-6 for k, b, f in (
                        ("prediction", "candidate", "predicted_goal_cost"), ("prediction_zero", "zero_prefix", "predicted_goal_cost"),
                        ("observed", "candidate", "observed_encoded_goal_cost"), ("observed_zero", "zero_prefix", "observed_encoded_goal_cost")))
                    historical_gates.append(replay)
                    if not replay:
                        raise RuntimeError("original checkpoint's four costs do not replay")
                print(json.dumps({"seed": row["seed"], "checkpoint": name, "precision": precision,
                    "model_margin": decomposition["predicted_margin"], "actual_margin": decomposition["observed_margin"]}), flush=True)
            model_gates.append(before == shared._parameter_hash(model) and not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters()))
        model.predict = original_predict
        del model
        torch.cuda.empty_cache()
    summary = summarize(rows)
    gates = {"cuda_only": True, "all6_fixed_roots": [r["seed"] for r in rows] == seeds,
        "all12_physical_packets_exact_replay": True, "all_original_four_costs_replay": len(historical_gates) == 6 and all(historical_gates),
        "all_checkpoints_frozen_eval": all(model_gates), "forward_sample_accounting": samples == 48,
        "action_embedding_reference_fp32_bit_exact": len(embedder_gates) == 24 and all(embedder_gates),
        "inputs_unchanged": all(digest(p) == protocol["inputs"][k]["sha256"] for k, p in inputs.items()),
        "implementation_unchanged": all(digest(p) == protocol["implementation_sha256"][k] for k, p in modules.items())}
    write_new(args.output, {"schema": "reacher_d2p_original_margin_checkpoint.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "sealed_protocol": protocol, "protocol_sha256": digest(protocol_path),
        "gates": gates, "rows": rows, "summary": summary, "actual_model_predict_samples": samples})
    print(json.dumps({"gates": gates, "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
