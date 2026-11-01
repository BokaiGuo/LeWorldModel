"""Compare fixed tail predictions with the saved physical RGB at both packets.

This is a post-outcome development diagnostic; it cannot select actions or
provide independent P1 evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_dev303_v3_four_arm_cp5_v13_model_20261102_exclusive as model_run
from . import p1_dev303_v3_four_arm_cp5_v13_physical_saved_20261102_exclusive as saved

ROOT = model_run.ROOT
base = model_run.base
require = model_run.require
PHYS_PREP = ROOT / "results/p1_dev303_v3_four_arm_cp5_v13_physical_20261102_exclusive_PREP_01.json"
PHYS_SHA = "76a71b0a61e3a6e2e12fb084f7ff64c2a3195f0caacd5139f00781419b7b194f"
RUN = ROOT / "results/p1_dev303_v3_four_arm_cp5_v13_physical_20261102_exclusive_RUN_01"
REPORT_SHA = "88fe063efd872ad2e3837489975c841efec19097d818709b08a596d7d20b7507"
AUTHORITY = "DEV303_FIXED_TAIL_PREDICTION_AUDIT_CUDA_ONLY"


def run(output, authority):
    require(authority == AUTHORITY, "explicit diagnostic authority")
    output = Path(output)
    require(not output.exists(), "exclusive output")
    verified = saved.verify(PHYS_PREP, PHYS_SHA, RUN, REPORT_SHA)
    require(verified["run_status"] == "recorded_development" and not verified["errors"],
            "complete saved physical chain")
    contract, report = base.read(PHYS_PREP), base.read(RUN / "report.json")
    source = base.read(model_run.PREP)
    task = source["task"]

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    device = torch.device("cuda:0")
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(source["policy"], device=device).eval().requires_grad_(False)
    before = model_run.engine.base.blind.host.signal.model_hash(model)
    require(before["original_state_dict"] == source["model_sha256"], "checkpoint identity")
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
    goal = tensor(task["goal"])[None]
    rows = []

    def encode_observation(row, phase, step):
        matches = [item for item in row["observations"]
                   if item["phase"] == phase and item["step"] == step]
        require(len(matches) == 1, "unique observation")
        pixels = model_run.inputs.rgb._frame(RUN, matches[0])
        image = model_run.engine.base.p0._image_tensor(
            pixels, transform=model_run.engine.base.p0.image_transform(), device=device)
        latent = model.encode({"pixels": image})["emb"][0, 0]
        require(bool(torch.isfinite(latent).all()), "finite encoded observation")
        return latent

    def predict(z, history, action):
        packet = ((action - mean) / std).reshape(1, 1, 10)
        full_history = torch.cat((history, packet), dim=1)
        predicted = model.predict(z, model.action_encoder(full_history))
        require(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()),
                "finite predicted context")
        return predicted[:, -1:], full_history[:, -2:]

    def mse(a, b):
        return float((a - b).square().mean().item())

    with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
        for role, row in zip(contract["roles"], report["rows"], strict=True):
            require(role["arm"] == row["arm"] and row["status"] == "fixed_horizon_completed",
                    "matching completed arm")
            cp5 = encode_observation(row, "prefix", 5)
            real1 = encode_observation(row, "tail", 5)[None, None]
            real2 = encode_observation(row, "tail", 10)[None, None]
            states, actions = model_run.encode.contexts(task, role["prefix"], cp5.cpu().numpy())
            z, past = tensor(states)[None], tensor(actions)[None]
            tail = tensor(np.asarray(role["tail"], np.float32).reshape(2, 5, 2))
            first, past1 = predict(z, past, tail[0])
            open_z = torch.cat((z, first), dim=1)[:, -3:]
            open2, _ = predict(open_z, past1, tail[1])
            refresh_z = torch.cat((z, real1), dim=1)[:, -3:]
            refresh2, _ = predict(refresh_z, past1, tail[1])
            rows.append({
                "arm": role["arm"], "tail_sha256": role["tail_sha256"],
                "first_prediction_mse": mse(first, real1),
                "open_second_prediction_mse": mse(open2, real2),
                "refreshed_second_prediction_mse": mse(refresh2, real2),
                "goal_distance": {
                    "actual_first": mse(real1, goal),
                    "predicted_first": mse(first, goal),
                    "actual_second": mse(real2, goal),
                    "predicted_open_second": mse(open2, goal),
                    "predicted_refreshed_second": mse(refresh2, goal),
                },
                "latent_vectors": {
                    "actual_first": real1.flatten().cpu().tolist(),
                    "predicted_first": first.flatten().cpu().tolist(),
                    "actual_second": real2.flatten().cpu().tolist(),
                    "predicted_open_second": open2.flatten().cpu().tolist(),
                    "predicted_refreshed_second": refresh2.flatten().cpu().tolist(),
                },
            })
    torch.cuda.synchronize(device)
    after = model_run.engine.base.blind.host.signal.model_hash(model)
    require(before == after, "model parameters changed")
    result = {"schema": "p1_dev303_four_arm_tail_prediction_audit.v1",
              "physical_report_sha256": REPORT_SHA, "physical_prepare_sha256": PHYS_SHA,
              "source_sha256": base.sha(__file__), "host": host,
              "model_before": before, "model_after": after, "rows": rows,
              "classification": "post_outcome_development_diagnostic",
              "independent_confirmation": False, "P1_admission": False}
    return model_run.inputs.physical.prior.write_new(output, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--authority", required=True)
    args = parser.parse_args()
    with model_run.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        print(json.dumps({"sha256": run(args.output, args.authority)}))


if __name__ == "__main__":
    main()
