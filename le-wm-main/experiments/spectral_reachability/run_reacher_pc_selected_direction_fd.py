"""CUDA finite-difference audit of previously selected spectral directions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import types

import gymnasium as gym
import numpy as np
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup

EPSILONS = (0.001, 0.01, 0.1, 0.5)


def probe_direction(terminal, origin, selected, epsilons=EPSILONS):
    """Compare autograd Jv with centered differences and full displacements."""
    direction = selected - origin
    radius = direction.norm()
    if origin.ndim != 1 or selected.shape != origin.shape or not torch.isfinite(direction).all() or radius <= 0:
        raise ValueError("invalid selected direction")
    direction = direction / radius
    point = origin.detach().requires_grad_(True)
    base = terminal(point)
    jacobian = torch.autograd.functional.jacobian(terminal, point, vectorize=True)
    if jacobian.ndim != 2 or jacobian.shape[1] != point.numel() or not torch.isfinite(jacobian).all():
        raise ValueError("invalid Jacobian")
    tangent = (jacobian @ direction).detach()
    tangent_norm = float(tangent.norm())
    if tangent_norm < 1e-12:
        raise ValueError("zero selected-direction response")
    samples = []
    with torch.no_grad():
        repeated = [terminal(point) for _ in range(3)]
        repeat_diffs = [(value - base).norm() for value in repeated]
        for epsilon in epsilons:
            if epsilon <= 0:
                raise ValueError("finite-difference step must be positive")
            plus = terminal(point + epsilon * direction)
            minus = terminal(point - epsilon * direction)
            centered = (plus - minus) / (2 * epsilon)
            actual_delta = plus - base
            linear_delta = epsilon * tangent
            samples.append({
                "epsilon": float(epsilon),
                "centered_fd_norm": float(centered.norm()),
                "centered_fd_relative_error": float((centered - tangent).norm() / tangent_norm),
                "forward_displacement_norm": float(actual_delta.norm()),
                "linear_displacement_norm": float(linear_delta.norm()),
                "forward_linear_relative_error": float((actual_delta - linear_delta).norm() /
                                                       actual_delta.norm().clamp_min(1e-12)),
            })
    return {"radius": float(radius), "jvp_norm": tangent_norm,
            "jacobian_output_dimension": jacobian.shape[0],
            "backward_vjp_rows": jacobian.shape[0], "jacobian_primal_rollouts": 1,
            "baseline_and_repeat_primal_rollouts": 4,
            "repeat_baseline_max_l2": float(torch.stack(repeat_diffs).max()),
            "finite_difference_primal_rollouts": 2 * len(samples), "samples": samples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--root-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--double", action="store_true", help="diagnostic float64 arithmetic with the same stored weights")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite output")
    if not torch.cuda.is_available():
        raise RuntimeError("host CUDA required")
    source = json.loads(args.source.read_text())
    roots = json.loads(args.root_source.read_text())
    if source["source_report_sha256"] != setup._sha256_file(args.root_source):
        raise RuntimeError("root source report mismatch")
    policy, stats = Path(roots["protocol"]["policy"]), Path(roots["protocol"]["action_stats"])
    for key, path in (("weights_sha256", policy / "weights.pt"), ("action_stats_sha256", stats)):
        if setup._sha256_file(path) != source[key]:
            raise RuntimeError(f"input mismatch: {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    setup.install_missing_field_compat()
    setup.validate_training_semantic_receipt(stats)
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
    model = setup.load_compatible_pretrained(policy, device="cuda").eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    before = setup._parameter_hash(model)
    if before != source["parameter_hash_before"]:
        raise RuntimeError("model mismatch")
    if args.double:
        model = model.double()
        # The upstream action Embedder.forward unconditionally calls .float().
        # Preserve its exact operations while allowing float64 in this
        # diagnostic process only; no installed package or parameters change.
        def dtype_preserving_embedder(module, value):
            value = value.to(dtype=module.patch_embed.weight.dtype)
            value = module.patch_embed(value.permute(0, 2, 1))
            return module.embed(value.permute(0, 2, 1))
        model.action_encoder.forward = types.MethodType(dtype_preserving_embedder, model.action_encoder)
    processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
    env = setup.ReacherGoalImageWrapper(
        gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65),
    )
    rows = []
    try:
        setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
        low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
        lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
        if args.double:
            lower, upper = lower.double(), upper.double()
        for row in source["rows"]:
            seed = row["seed"]
            packets, _ = setup.sample_history_packets(histories, seed=seed)
            context = setup.collect_executed_context(
                env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history",
            )
            if context["root_rgb_sha256"] != row["root_rgb_sha256"] or context["goal_rgb_sha256"] != row["goal_rgb_sha256"]:
                raise RuntimeError("root or goal mismatch")
            pixels, goal, history = setup.p0._prepare(
                processor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), torch.device("cuda"),
            )
            if args.double:
                pixels, goal, history = pixels.double(), goal.double(), history.double()
            selected = torch.as_tensor(row["selected_params"], device="cuda", dtype=history.dtype).reshape(-1)
            bank_selected = torch.as_tensor(row["bank_params"][row["selection"]["selected_proposal_index"]],
                                            device="cuda", dtype=history.dtype).reshape(-1)
            if not torch.equal(selected, bank_selected):
                raise RuntimeError("selected witness differs from source bank")
            def terminal(x):
                output = setup.p0._terminal_batch(model, pixels, goal, history, x.reshape(1, 1, -1),
                                                 action_lower=lower, action_upper=upper)[0]
                return setup.p0._planning_coordinate(model, output, role="predicted")
            result = probe_direction(terminal, torch.zeros_like(selected), selected)
            rows.append({"seed": seed, "selected_proposal_index": row["selection"]["selected_proposal_index"],
                         "audit": result})
            print(json.dumps({"seed": seed, "audit": result}), flush=True)
    finally:
        env.close()
    if args.double:
        model = model.float()
    after = setup._parameter_hash(model)
    if after != before:
        raise RuntimeError("model parameters changed")
    report = {"scope": "known-root numerical diagnostic, not a physical capability or admission result",
              "arithmetic": "float64 diagnostic" if args.double else "float32",
              "action_embedder": "diagnostic dtype-preserving override" if args.double else "upstream implementation",
              "source_sha256": setup._sha256_file(args.source), "script_sha256": setup._sha256_file(Path(__file__)),
              "parameter_hash_before": before, "parameter_hash_after": after, "rows": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
