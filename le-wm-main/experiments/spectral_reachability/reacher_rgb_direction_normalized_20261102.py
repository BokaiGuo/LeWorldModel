"""Measured dimension-normalized direction CEM on the fixed development bank."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import reacher_response_svd_cp5_cem_20261102 as base
from . import reacher_response_svd_qcp5_cem_20261102 as cem

ORIGINAL_PLAN = base.make_plan
PROTOCOL = Path(__file__).resolve().parents[3] / "docs/PC_REACHER_RGB_DIRECTION_NORMALIZED_PROTOCOL_20261102.md"
AUTHORITY = "RESPONSE_SVD_RGB_DIRECTION_NORMALIZED_DEVELOPMENT"
WEIGHT = 0.1


def direction_costs(predicted, target, observed):
    if (predicted.ndim != 2 or target.ndim != 1 or observed.shape != target.shape
            or target.numel() == 0 or predicted.shape[1] != target.numel()
            or not torch.isfinite(predicted).all() or not torch.isfinite(target).all()
            or not torch.isfinite(observed).all()):
        raise ValueError("invalid RGB-direction coordinates")
    direction = observed - target
    norm = direction.norm()
    if not torch.isfinite(norm) or float(norm) <= 1e-10:
        raise ValueError("observed-goal direction unavailable")
    error = predicted - target
    return (error.square().sum(dim=1) + WEIGHT * (error @ (direction / norm)).square()) / target.numel()


def make_plan(*inputs):
    plan = ORIGINAL_PLAN(*inputs)
    plan["schema"] = "response_svd_rgb_direction_normalized_v1"
    plan["files"].update({str(Path(__file__).resolve()): base.bridge.sha_file(__file__),
                          str(PROTOCOL): base.bridge.sha_file(PROTOCOL),
                          str(Path(cem.__file__).resolve()): base.bridge.sha_file(cem.__file__)})
    plan["direction_relative_weight"] = WEIGHT
    plan["scope"] = "two exposed roots; fixed prefixes; normalized direction CEM; P1 admission false"
    return plan


def measured_cem(model, pixels, goal, history, *, future_horizon, action_lower, action_upper,
                 samples, steps, elite, seed):
    from . import run_reacher_d2p_p0_goal_progress as p0

    if pixels.device.type != "cuda" or history.device.type != "cuda":
        raise ValueError("host CUDA required")
    original_encode, original_predict = model.encode, model.predict
    encode_override, predict_override = "encode" in vars(model), "predict" in vars(model)
    encode_attr, predict_attr = vars(model).get("encode"), vars(model).get("predict")
    counts = {"encode_calls": 0, "predict_calls": 0, "predict_samples": 0}

    def encode(*args, **kwargs):
        counts["encode_calls"] += 1
        return original_encode(*args, **kwargs)

    def predict(embedding, action, *args, **kwargs):
        counts["predict_calls"] += 1
        counts["predict_samples"] += int(embedding.shape[0])
        return original_predict(embedding, action, *args, **kwargs)

    model.encode, model.predict = encode, predict
    try:
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.no_grad():
            target = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()
            observed = p0._planning_coordinate(
                model, p0._latent_context(model, pixels, history)[0, -1], role="observed").detach()
        direction_costs(target.unsqueeze(0), target, observed)

        def evaluate(candidates):
            terminal = p0._terminal_batch(model, pixels, goal, history, candidates,
                                          action_lower=action_lower, action_upper=action_upper)
            coordinates = p0._planning_coordinate(model, terminal, role="predicted")
            return direction_costs(coordinates, target, observed)

        selected, receipt = cem.cem_search(
            evaluate, dimension=history.shape[-1], device=history.device,
            future_horizon=future_horizon, samples=samples, steps=steps, elite=elite, seed=seed)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        receipt.update(objective="observed_rgb_direction_normalized", direction_relative_weight=WEIGHT,
                       measured_suffix_calls=counts, measured_suffix_wall_seconds=elapsed)
        return selected, receipt
    finally:
        if encode_override:
            model.encode = encode_attr
        else:
            delattr(model, "encode")
        if predict_override:
            model.predict = predict_attr
        else:
            delattr(model, "predict")


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
        print(json.dumps({"sha256": base.bridge.sha_file(args.output)}))
        return
    base.host.require(args.authority == AUTHORITY, "normalized direction authority")
    from . import run_reacher_d2p_p0_goal_progress as p0

    original_cem, original_plan = p0._cem, base.make_plan
    try:
        p0._cem, base.make_plan = measured_cem, make_plan
        print(json.dumps(base.run(args.plan, args.sha256, args.output, base.AUTHORITY)))
    finally:
        p0._cem, base.make_plan = original_cem, original_plan


if __name__ == "__main__":
    main()
