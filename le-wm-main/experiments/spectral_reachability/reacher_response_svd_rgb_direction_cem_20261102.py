"""Observed-RGB direction-conditioned suffix CEM on the fixed development bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import reacher_response_svd_cp5_cem_20261102 as base
from . import reacher_response_svd_qcp5_cem_20261102 as q_cem

ORIGINAL_PLAN = base.make_plan
PROTOCOL = Path(__file__).resolve().parents[3] / "docs/PC_REACHER_RGB_DIRECTION_CEM_PROTOCOL_20261102.md"
AUTHORITY = "RESPONSE_SVD_RGB_DIRECTION_CEM_DEVELOPMENT"
WEIGHT = 0.1


def direction_costs(predicted, target, observed):
    if (predicted.ndim != 2 or target.ndim != 1 or observed.shape != target.shape
            or predicted.shape[1] != target.numel() or not torch.isfinite(predicted).all()
            or not torch.isfinite(target).all() or not torch.isfinite(observed).all()):
        raise ValueError("invalid RGB-direction coordinates")
    direction = observed - target
    norm = direction.norm()
    if not torch.isfinite(norm) or float(norm) <= 1e-10:
        raise ValueError("observed-goal direction unavailable")
    direction = direction / norm
    error = predicted - target
    return error.square().mean(dim=1) + WEIGHT * (error @ direction).square()


def make_plan(*inputs):
    plan = ORIGINAL_PLAN(*inputs)
    plan["schema"] = "response_svd_rgb_direction_cem_v1"
    plan["files"].update({str(Path(__file__).resolve()): base.bridge.sha_file(__file__),
                          str(PROTOCOL): base.bridge.sha_file(PROTOCOL),
                          str(Path(q_cem.__file__).resolve()): base.bridge.sha_file(q_cem.__file__)})
    plan["rgb_direction_weight"] = WEIGHT
    plan["additional_encoder_calls_per_suffix"] = 2
    plan["scope"] = "two exposed roots; fixed prefixes; RGB-direction CEM; P1 admission false"
    return plan


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for key in ("source-report", "physical-plan", "preflight", "output"):
        prep.add_argument("--" + key, required=True)
    run = sub.add_parser("run")
    for key in ("plan", "sha256", "output", "authority"):
        run.add_argument("--" + key, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        base.host.write_new(args.output, make_plan(args.source_report, args.physical_plan, args.preflight))
        print(json.dumps({"plan": args.output, "sha256": base.bridge.sha_file(args.output)}))
        return
    base.host.require(args.authority == AUTHORITY, "RGB-direction CEM authority")
    from . import run_reacher_d2p_p0_goal_progress as p0

    previous_cem, previous_plan = p0._cem, base.make_plan

    def rgb_direction_cem(model, pixels, goal, history, *, future_horizon, action_lower, action_upper,
                          samples, steps, elite, seed):
        with torch.no_grad():
            target = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()
            observed = p0._planning_coordinate(
                model, p0._latent_context(model, pixels, history)[0, -1], role="observed").detach()
        # Check the observed direction before spending any candidate evaluations.
        direction_costs(target.unsqueeze(0), target, observed)

        def evaluate(candidates):
            terminal = p0._terminal_batch(model, pixels, goal, history, candidates,
                                          action_lower=action_lower, action_upper=action_upper)
            coordinates = p0._planning_coordinate(model, terminal, role="predicted")
            return direction_costs(coordinates, target, observed)

        selected, receipt = q_cem.cem_search(
            evaluate, dimension=history.shape[-1], device=history.device,
            future_horizon=future_horizon, samples=samples, steps=steps, elite=elite, seed=seed)
        receipt.update(objective="observed_rgb_direction_plus_l2", direction_weight=WEIGHT,
                       additional_encoder_calls=2)
        return selected, receipt

    try:
        p0._cem = rgb_direction_cem
        base.make_plan = make_plan
        print(json.dumps(base.run(args.plan, args.sha256, args.output, base.AUTHORITY)))
    finally:
        p0._cem = previous_cem
        base.make_plan = previous_plan


if __name__ == "__main__":
    main()
