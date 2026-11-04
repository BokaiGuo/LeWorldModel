"""Rescore physically executed suffix pairs from their shared real cp5 RGB."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import audit_reacher_rgb_direction_normalized_20261102 as paired
from . import reacher_cem_context_sensitivity_v2_20261102 as sensitivity
from . import reacher_rgb_direction_normalized_20261102 as metric

host, setup, ROOT = sensitivity.host, sensitivity.setup, sensitivity.ROOT
PLAN = ROOT / "results/reacher_rgb_direction_normalized_PLAN_20261102_v1.json"
PLAN_SHA = "d9e505458e5a825c90b3c1c619ca3a2530723f3e828f881afa9ba887dcea2bbe"
JOURNAL = ROOT / "results/reacher_response_svd_cp5_rgb_direction_normalized_RUN_20261102_v1/events.jsonl"
JOURNAL_SHA = "2a69e3047366a19914a5f6c59a672a40bfe1569f0cc3346c8741f86a83d3b15b"
RECEIPT = ROOT / "results/reacher_rgb_direction_normalized_SAVED_20261102_v1.json"
RECEIPT_SHA = "38347e5f7719bc25e45a0ff337a16ac5dead98517308137d0fb54098a89f655a"
PROTOCOL = ROOT.parent / "docs/PC_REACHER_RGB_DIRECTION_SAME_ACTION_PROTOCOL_20261102.md"
AUTH = "REACHER_RGB_DIRECTION_SAME_ACTION_DEVELOPMENT"


def direct_terminal(model, pixels, history, future):
    """Existing root-anchored rollout with already bounded actions."""
    p0 = setup.p0
    if (future.ndim != 3 or future.shape[1:] != (2, 10) or history.shape != (1, 3, 10)
            or pixels.shape[:2] != history.shape[:2] or not torch.isfinite(future).all()):
        raise ValueError("invalid normalized action or RGB history")
    encoded = p0._latent_context(model, pixels, history).expand(future.shape[0], -1, -1)
    actions = torch.cat([history[:, 1:].expand(future.shape[0], -1, -1), future[:, :1]], dim=1)
    for step in range(2):
        terminal = model.predict(encoded[:, -3:], model.action_encoder(actions[:, -3:]))[:, -1]
        if step == 0:
            encoded = torch.cat([encoded, terminal[:, None]], dim=1)
            actions = torch.cat([actions, future[:, 1:2]], dim=1)
    return terminal


def prepare():
    host.require(setup._sha256_file(RECEIPT) == RECEIPT_SHA, "receipt hash")
    saved = json.loads(RECEIPT.read_text())
    actual = paired.consume(PLAN, PLAN_SHA, JOURNAL, JOURNAL_SHA)
    host.require(saved == actual, "source receipt")
    plan = json.loads(PLAN.read_text())
    files = [Path(__file__), PROTOCOL, PLAN, JOURNAL, RECEIPT, paired.prior.L2_JOURNAL,
             paired.prior.L2_RECEIPT, Path(metric.__file__), Path(setup.p0.__file__),
             Path(plan["weights"]), Path(plan["stats"])]
    return {"schema": "reacher_rgb_direction_same_action_v1",
            "files": {str(p): setup._sha256_file(p) for p in files},
            "model_hash": plan["model_parameter_hash"], "weights": plan["weights"], "stats": plan["stats"],
            "roots": [20263160, 20263161], "pairs": 26, "physical_steps": 0,
            "P1_admission": False, "independent_confirmation": False}


def run(plan_path, digest, output, authority):
    from experiments.gpu_preflight import require_cuda_allocation

    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID, "CUDA authority")
    plan_path, output = Path(plan_path), Path(output)
    host.require(setup._sha256_file(plan_path) == digest and json.loads(plan_path.read_text()) == prepare(),
                 "plan/source drift")
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_rgb_direction_same_action_RUN_")
                 and not os.path.lexists(output), "exclusive output")
    source_plan = json.loads(PLAN.read_text())
    event_sets = [paired.audit.saved.verify_journal(p) for p in (paired.prior.L2_JOURNAL, JOURNAL)]
    ctx = paired.audit.context_lib
    with host.lock.gpu_lock(host.LOCK):
        allocation = require_cuda_allocation("cuda:0")
        host.require(torch.cuda.device_count() == 1 and host.canonical_gpu_uuid(
            torch.cuda.get_device_properties(0).uuid) == host.canonical_gpu_uuid(host.UUID), "GPU identity")
        mean, std = setup._load_action_stats(Path(source_plan["stats"]), "swm/ReacherDMControl-v0")
        setup.validate_training_semantic_receipt(Path(source_plan["stats"]))
        model = setup.load_compatible_pretrained(Path(source_plan["weights"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        before = setup._parameter_hash(model)
        host.require(before == source_plan["model_parameter_hash"], "model hash")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        rows = []
        for root in source_plan["roots"]:
            for arm in root["arms"]:
                found = [[e for e in events if e.get("seed") == root["seed"] and e.get("arm") == arm["arm"]
                          and e["kind"] == "arm_returned"] for events in event_sets]
                contexts = [[e for e in events if e.get("seed") == root["seed"] and e.get("arm") == arm["arm"]
                             and e["kind"] == "context" and e["role"] == "successor"] for events in event_sets]
                host.require(all(len(v) == 1 for v in found + contexts), "arm/context identity")
                actual = [entry[0]["result"] for entry in found]
                value = contexts[0][0]["value"]
                frames = [ctx.unpack_rgb_observation(f) for f in value["frames"]]
                goal = ctx.unpack_rgb_observation(value["goal"])
                history = np.asarray(value["history"], np.float32)
                pixels, target_pixels, actions = sensitivity.prepare_saved_frames(
                    setup.p0._prepare, processor, frames, goal, history, torch.device("cuda"))
                suffixes, residuals = [], []
                for result in actual:
                    normalized = np.asarray(result["action_witness_normalized"], np.float32)[1:]
                    raw = np.asarray(result["action_witness_raw"], np.float32)[5:]
                    host.require(normalized.shape == (2, 10) and raw.shape == (10, 2), "witness shape")
                    rebuilt = setup.p0._normalized_blocks_to_raw_actions(normalized, mean, std, action_dim=2)
                    residual = float(np.max(np.abs(rebuilt - raw)))
                    host.require(np.isfinite(residual) and residual <= 1e-6, "normalized/raw witness mismatch")
                    suffixes.append(normalized)
                    residuals.append(residual)
                futures = torch.as_tensor(np.stack(suffixes), device="cuda")
                with torch.no_grad():
                    target = setup.p0._planning_coordinate(
                        model, setup.p0._goal_latent(model, target_pixels), role="goal")
                    observed = setup.p0._planning_coordinate(
                        model, setup.p0._latent_context(model, pixels, actions)[0, -1], role="observed")
                    prediction = setup.p0._planning_coordinate(
                        model, direct_terminal(model, pixels, actions, futures), role="predicted")
                    l2 = (prediction - target).square().mean(dim=1)
                    direction = metric.direction_costs(prediction, target, observed)
                rows.append({"seed": root["seed"], "prefix": arm["proposal_index"],
                             "cp5_rgb_sha256": ctx.rgb_digest(frames[-1]), "goal_sha256": ctx.rgb_digest(goal),
                             "physical_final": [r["physical"]["final_qpos_distance"] for r in actual],
                             "l2_model_cost": l2.cpu().tolist(), "direction_model_cost": direction.cpu().tolist(),
                             "normalized_raw_max_abs": residuals, "normalized_actions": np.stack(suffixes).tolist()})
        host.require(len(rows) == 26 and setup._parameter_hash(model) == before
                     and json.loads(plan_path.read_text()) == prepare(), "postcheck")
        host.write_new(output, {"schema": "reacher_rgb_direction_same_action_result_v1", "plan_sha256": digest,
                                "cuda_allocation": allocation, "model_before": before,
                                "model_after": before, "rows": rows,
                                "predict_calls": 52, "predict_samples": 104, "P1_admission": False})
        print(json.dumps({"sha256": setup._sha256_file(output), "pairs": len(rows)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    for name in ("plan", "sha256", "output", "authority"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.command == "prepare":
        host.write_new(args.plan, prepare())
        print(json.dumps({"sha256": setup._sha256_file(Path(args.plan))}))
    else:
        run(args.plan, args.sha256, args.output, args.authority)


if __name__ == "__main__":
    main()
