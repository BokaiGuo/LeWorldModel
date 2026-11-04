"""Encode all fixed replay outcomes and separate the two ranking links."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import audit_reacher_fixed_pair_rgb_replay_20261102 as audit
from . import reacher_fixed_pair_rgb_replay_20261102 as replay
from . import reacher_rgb_direction_same_action_20261102 as source

host, setup, ROOT = source.host, source.setup, source.ROOT
PLAN = ROOT / "results/reacher_fixed_pair_rgb_replay_PLAN_20261102_v1.json"
PLAN_SHA = "5657983996890506536def63c9cef3114a8a200bacba7638bb97285d9c65dede"
JOURNAL = ROOT / "results/reacher_fixed_pair_rgb_replay_RUN_20261102_v2/events.jsonl"
JOURNAL_SHA = "98569bcddc661cb776d35e1c297041421bcfc5f31e58e59aa0ea8e21f09f0bdc"
RECEIPT = ROOT / "results/reacher_fixed_pair_rgb_replay_SAVED_20261102_v1.json"
RECEIPT_SHA = "15bca8f27a774b4583b7db7762c913a8220b1ea2e5c9b22827601ac01446b043"
PRED = ROOT / "results/reacher_rgb_direction_same_action_RUN_20261102_v1.json"
PRED_SHA = "4be88439b65bb7222f0e796d6fa80cbb9da3fbe6eb7321e9800c1dc15ffea001"
PROTOCOL = ROOT.parent / "docs/PC_REACHER_PAIR_RGB_ATTRIBUTION_PROTOCOL_20261102.md"
AUTH = "REACHER_FIXED_PAIR_RGB_ATTRIBUTION_DEVELOPMENT"


def prepare():
    for path, digest in ((PLAN, PLAN_SHA), (JOURNAL, JOURNAL_SHA), (RECEIPT, RECEIPT_SHA),
                         (PRED, PRED_SHA)):
        host.require(setup._sha256_file(path) == digest, "input hash")
    plan = json.loads(PLAN.read_text())
    host.require(plan == replay.prepare(), "replay source drift")
    actual = audit.validate(plan, replay.pair.audit.saved.verify_journal(JOURNAL), PLAN_SHA)
    saved = json.loads(RECEIPT.read_text())
    for key in ("schema", "status", "branches", "rows", "P1_admission", "limits"):
        host.require(actual[key] == saved[key], "replay receipt mismatch")
    original = json.loads(source.PLAN.read_text())
    files = [Path(__file__), PROTOCOL, Path(source.sensitivity.__file__), Path(setup.p0.__file__),
             Path(original["weights"]), Path(original["stats"])]
    return {"schema": "reacher_fixed_pair_rgb_attribution_v1",
            "files": {str(p): setup._sha256_file(p) for p in files},
            "replay_plan_sha256": PLAN_SHA, "replay_journal_sha256": JOURNAL_SHA,
            "replay_receipt_sha256": RECEIPT_SHA, "prediction_sha256": PRED_SHA,
            "model_hash": original["model_parameter_hash"], "weights": original["weights"],
            "stats": original["stats"], "pairs": 26, "P1_admission": False}


def run(plan_path, digest, output, authority):
    from experiments.gpu_preflight import require_cuda_allocation

    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID, "CUDA authority")
    plan_path, output = Path(plan_path), Path(output)
    host.require(setup._sha256_file(plan_path) == digest and json.loads(plan_path.read_text()) == prepare(),
                 "plan drift")
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_fixed_pair_rgb_attribution_RUN_")
                 and not os.path.lexists(output), "exclusive output")
    replay_plan = json.loads(PLAN.read_text())
    events = replay.pair.audit.saved.verify_journal(JOURNAL)
    source_events = replay.pair.audit.saved.verify_journal(replay.pair.prior.L2_JOURNAL)
    predictions = json.loads(PRED.read_text())["rows"]
    with host.lock.gpu_lock(host.LOCK):
        allocation = require_cuda_allocation("cuda:0")
        host.require(torch.cuda.device_count() == 1 and host.canonical_gpu_uuid(
            torch.cuda.get_device_properties(0).uuid) == host.canonical_gpu_uuid(host.UUID), "GPU identity")
        original = json.loads(source.PLAN.read_text())
        stats = Path(original["stats"])
        setup.validate_training_semantic_receipt(stats)
        mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        model = setup.load_compatible_pretrained(Path(original["weights"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        before = setup._parameter_hash(model)
        host.require(before == original["model_parameter_hash"], "model hash")
        rows = []
        for root in replay_plan["roots"]:
            for index in range(0, len(root["branches"]), 2):
                arms = root["branches"][index:index+2]
                host.require([a["label"] for a in arms] == ["l2", "direction"]
                             and arms[0]["prefix"] == arms[1]["prefix"], "pair order")
                contexts = [e["value"] for e in source_events if e["kind"] == "context"
                            and e.get("seed") == root["seed"] and e.get("arm") == arms[0]["arm"]
                            and e["role"] == "successor"]
                host.require(len(contexts) == 1, "source context")
                value = contexts[0]
                goal = replay.context.unpack_rgb_observation(value["goal"])
                cp5_frames = [replay.context.unpack_rgb_observation(x) for x in value["frames"]]
                cp5_history = np.asarray(value["history"], np.float32)
                pixels, goal_pixels, history = source.sensitivity.prepare_saved_frames(
                    setup.p0._prepare, processor, cp5_frames, goal, cp5_history, torch.device("cuda"))
                with torch.no_grad():
                    target = setup.p0._planning_coordinate(
                        model, setup.p0._goal_latent(model, goal_pixels), role="goal")
                    observed = setup.p0._planning_coordinate(
                        model, setup.p0._latent_context(model, pixels, history)[0, -1], role="observed")
                results = []
                for offset, arm in enumerate(arms):
                    found = [e["outcome"] for e in events if e["kind"] == "branch_returned"
                             and e["seed"] == root["seed"] and e["index"] == index + offset]
                    host.require(len(found) == 1, "replay result")
                    points = found[0]["trajectory"]
                    frames = [replay.context.unpack_rgb_observation(points[i]["rgb"]) for i in (5, 10, 15)]
                    actions = np.asarray(arm["actions"], np.float32).reshape(3, 10)
                    actual_pixels, actual_goal, actual_history = source.sensitivity.prepare_saved_frames(
                        setup.p0._prepare, processor, frames, goal, actions, torch.device("cuda"))
                    host.require(torch.equal(goal_pixels, actual_goal), "goal mismatch")
                    with torch.no_grad():
                        actual = setup.p0._planning_coordinate(
                            model, setup.p0._latent_context(model, actual_pixels, actual_history)[0, -1],
                            role="observed")
                        l2 = (actual - target).square().mean()
                        direction = source.metric.direction_costs(actual[None], target, observed)[0]
                    results.append({"label": arm["label"], "real_l2": float(l2),
                                    "real_direction": float(direction),
                                    "physical_final": points[15]["qpos_distance"],
                                    "final_rgb_sha256": audit.checks.rgb(points[15]["rgb"])})
                pred = [x for x in predictions if x["seed"] == root["seed"]
                        and x["prefix"] == arms[0]["prefix"]]
                host.require(len(pred) == 1 and pred[0]["cp5_rgb_sha256"] == arms[0]["cp5_rgb_sha256"]
                             and np.allclose(pred[0]["physical_final"],
                                             [r["physical_final"] for r in results], atol=1e-6, rtol=0),
                             "prediction binding")
                rows.append({"seed": root["seed"], "prefix": arms[0]["prefix"],
                             "model_l2": pred[0]["l2_model_cost"],
                             "model_direction": pred[0]["direction_model_cost"], "actual": results})
        host.require(len(rows) == 26 and setup._parameter_hash(model) == before
                     and json.loads(plan_path.read_text()) == prepare(), "postcheck")
        host.write_new(output, {"schema": "reacher_fixed_pair_rgb_attribution_result_v1",
                                "plan_sha256": digest, "cuda_allocation": allocation,
                                "model_before": before, "model_after": before, "rows": rows,
                                "P1_admission": False})
        print(json.dumps({"pairs": len(rows), "sha256": setup._sha256_file(output)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    for arg in ("plan", "sha256", "output", "authority"):
        parser.add_argument("--" + arg)
    args = parser.parse_args()
    if args.command == "prepare":
        host.write_new(args.plan, prepare())
        print(json.dumps({"sha256": setup._sha256_file(Path(args.plan))}))
    else:
        run(args.plan, args.sha256, args.output, args.authority)


if __name__ == "__main__":
    main()
