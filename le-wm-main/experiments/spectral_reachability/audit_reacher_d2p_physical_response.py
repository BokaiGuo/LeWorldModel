"""CUDA-only all-root model/physical first-packet response comparison.

Fixed coordinate probes, not an action selector. Two predeclared scales expose
finite-difference curvature and RGB rasterization sensitivity. Neither the
physical Jacobian nor evaluator qpos may enter deployment planning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import audit_reacher_d2p_causal_innovation as replay
from experiments.spectral_reachability import audit_reacher_d2p_rgb_compiler_alignment as shared
from experiments.spectral_reachability import reacher_d2p_context as execution
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0

SCALES = (.1, .25)


def response_metrics(analytic, model_fd, physical_fd, goal_residual):
    if analytic.ndim != 2 or any(matrix.shape != analytic.shape for matrix in (model_fd, physical_fd)):
        raise ValueError("matching output-by-action response matrices required")
    if goal_residual.shape != analytic.shape[:1] or any(not torch.isfinite(value).all()
            for value in (analytic, model_fd, physical_fd, goal_residual)):
        raise ValueError("finite responses and matching goal residual required")
    column_cosine = torch.nn.functional.cosine_similarity(analytic, physical_fd, dim=0, eps=1e-12)
    predicted_slopes = goal_residual @ analytic
    physical_slopes = goal_residual @ physical_fd
    return {"model_autograd_vs_model_fd_relative_frobenius_error": float((analytic - model_fd).norm() / analytic.norm().clamp_min(1e-12)),
        "model_vs_physical_fd_relative_frobenius_error": float((analytic - physical_fd).norm() / physical_fd.norm().clamp_min(1e-12)),
        "model_physical_column_cosines": column_cosine.tolist(),
        "model_singular_values": torch.linalg.svdvals(analytic).tolist(),
        "physical_fd_singular_values": torch.linalg.svdvals(physical_fd).tolist(),
        "common_observed_goal_residual_slope_sign_agreements": int((predicted_slopes * physical_slopes > 0).sum()),
        "action_dimension": analytic.shape[1],
        "slope_scope": "Same observed-zero goal residual projects both responses; isolates response error from common prediction bias."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.reference.read_text())
    replay.validate_reference(report)
    reference = report["sealed_protocol"]
    digest = replay.digest
    if digest(args.policy / "weights.pt") != reference["weights_sha256"] or digest(args.adapter_artifact) != reference["adapter_sha256"]:
        raise ValueError("reference checkpoint/action stats changed")
    implementations = {"auditor": Path(__file__), "context": Path(execution.__file__),
        "p0": Path(p0.__file__), "source_context": Path(shared._source_state_context.__code__.co_filename)}
    for key in ("context", "p0", "source_context"):
        if digest(implementations[key]) != reference["implementation_sha256"][key]:
            raise ValueError("reference execution implementation changed")
    output = args.output.resolve()
    if any(path.exists() for path in (output, output.with_suffix(".protocol.json"), output.with_suffix(".roots"))):
        raise FileExistsError("new immutable output required")
    protocol = {"root_seeds": reference["root_seeds"], "reference_sha256": digest(args.reference),
        "weights_sha256": reference["weights_sha256"], "adapter_sha256": reference["adapter_sha256"],
        "source_sha256": reference["source_sha256"], "parameter_probe_scales": SCALES,
        "root_identity": "sealed_observation_exact_state; original measured root checked against reference, fresh successors always rendered",
        "columns": "all ten tanh-bounded parameter coordinates, first five-step packet, symmetric signs",
        "implementation_sha256": {key: digest(path) for key, path in implementations.items()},
        "scope": "Known-root evaluator-only response diagnosis; no threshold fitting, selection, efficacy or P1 override."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    output.with_suffix(".roots").mkdir()
    print("Comparing analytic CUDA action Jacobians with model and physical differences at both fixed scales, all ten coordinates and all reference roots.", flush=True)
    host = shared.require_cuda_allocation("cuda")
    shared.install_missing_field_compat()
    device = torch.device("cuda")
    mean, std = shared._load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    dataset, source_hash = shared.receipt_source(args.adapter_artifact)
    if source_hash != reference["source_sha256"]:
        raise ValueError("source SHA receipt changed")
    model = shared.load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = shared._parameter_hash(model)
    processor = shared._Preprocessor(shared.ActionNormalizer(mean, std))
    env = shared.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=tuple(reference["target_delta"]))
    low, high = shared._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(low, device=device), torch.as_tensor(high, device=device)
    rows = []
    try:
        for source in sorted(report["rows"], key=lambda row: row["seed"]):
            seed = source["seed"]
            context = shared._source_state_context(env, dataset=dataset, seed=seed)
            identity = source["methods"]["zero"]["execution_identity"]["initial"]
            if any(context[key] != source[key] for key in ("source_episode", "source_start_row")) or context["root_rgb_sha256"] != identity["reset_rgb_sha256"] or context["goal_rgb_sha256"] != identity["goal_rgb_sha256"]:
                raise RuntimeError("reference source root/goal changed")
            context["execution_snapshot"]["observed_root_rgb"] = execution.pack_rgb_observation(context["root_frame"])
            pixels, goal, history = p0._prepare(processor, context["frames"], context["root_frame"], context["goal"], np.asarray(context["history_action_blocks"], np.float32), device)

            def predict(flat):
                return p0._terminal_batch(model, pixels, goal, history, flat.reshape(1, 1, 10), action_lower=lower, action_upper=upper)[0]

            def observe(flat):
                raw, _ = p0._to_raw_actions(flat.reshape(1, 1, 10), mean, std, lower, upper, env.action_space.low, env.action_space.high)
                root, goal_rgb = execution.restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"], root_identity="sealed_observation_exact_state")
                p0._verify_execution_identity(root, goal_rgb, context["root_frame"], context["goal"])
                advanced = execution.advance_executed_context(env, context=context, raw_action_block=raw, action_block=5, verified_root_frame=root)
                next_pixels, _, next_history = p0._prepare(processor, advanced["frames"], advanced["root_frame"], advanced["goal"], np.asarray(advanced["history_action_blocks"], np.float32), device)
                return p0._latent_context(model, next_pixels, next_history)[0, -1], advanced["root_rgb_sha256"]

            origin = torch.zeros(10, device=device, requires_grad=True)
            analytic = torch.autograd.functional.jacobian(predict, origin, vectorize=True).detach().to(torch.float64)
            observed_zero, zero_hash = observe(origin.detach())
            zero_exact = zero_hash == source["methods"]["zero"]["observed_successor_context"]["root_rgb_sha256"]
            residual = (observed_zero - p0._goal_latent(model, goal)).to(torch.float64)
            probes = []
            for scale in SCALES:
                model_columns, physical_columns = [], []
                for column in range(10):
                    plus = origin.detach().clone()
                    plus[column] = scale
                    with torch.no_grad():
                        model_difference = (predict(plus).to(torch.float64) - predict(-plus).to(torch.float64)) / (2 * scale)
                        physical_difference = (observe(plus)[0].to(torch.float64) - observe(-plus)[0].to(torch.float64)) / (2 * scale)
                    model_columns.append(model_difference)
                    physical_columns.append(physical_difference)
                model_fd, physical_fd = torch.stack(model_columns, dim=1), torch.stack(physical_columns, dim=1)
                probes.append({"scale": scale, "metrics": response_metrics(analytic, model_fd, physical_fd, residual),
                    "model_fd": model_fd.cpu().tolist(), "physical_fd": physical_fd.cpu().tolist()})
            row = {"seed": seed, "source_episode": source["source_episode"],
                "model_eligible": source["methods"]["dispatch"]["first_receipt"]["dispatch"]["reconfiguration_witness_found"],
                "analytic_jacobian": analytic.cpu().tolist(), "probes": probes,
                "gates": {"recorded_zero_successor_exact": zero_exact, "finite_full_coordinate_responses": all(torch.isfinite(torch.as_tensor(probe["physical_fd"])).all().item() for probe in probes)}}
            with (output.with_suffix(".roots") / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(row, indent=2, sort_keys=True) + "\n")
            rows.append(row)
            print(json.dumps({"seed": seed, "gates": row["gates"], "metrics": [probe["metrics"] for probe in probes]}), flush=True)
    finally:
        env.close()
    gates = {"cuda_only": True, "weights_unchanged": before == shared._parameter_hash(model),
        "all_reference_roots_retained": [row["seed"] for row in rows] == sorted(reference["root_seeds"]),
        "all_execution_gates": all(all(row["gates"].values()) for row in rows),
        "implementation_unchanged": all(digest(path) == protocol["implementation_sha256"][key] for key, path in implementations.items())}
    aggregate = {str(scale): {"mean_model_autograd_vs_fd_relative_error": float(np.mean([row["probes"][index]["metrics"]["model_autograd_vs_model_fd_relative_frobenius_error"] for row in rows])),
        "mean_model_vs_physical_relative_error": float(np.mean([row["probes"][index]["metrics"]["model_vs_physical_fd_relative_frobenius_error"] for row in rows])),
        "goal_slope_sign_agreements": sum(row["probes"][index]["metrics"]["common_observed_goal_residual_slope_sign_agreements"] for row in rows), "total_columns": len(rows) * 10}
        for index, scale in enumerate(SCALES)}
    result = {"schema": "reacher_d2p_physical_response_audit.v1", "status": "completed" if all(gates.values()) else "fail",
        "cuda_host": host, "gates": gates, "sealed_protocol": protocol, "aggregate": aggregate, "rows": rows,
        "data_verification": "Prior source SHA receipt reused; source root/goal identities checked. No new full-source SHA claimed.",
        "claim_boundary": protocol["scope"]}
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gates": gates, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
