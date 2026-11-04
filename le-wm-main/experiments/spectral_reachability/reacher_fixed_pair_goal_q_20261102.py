"""Score identical predicted and real RGB outcomes with existing goal-Q."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import reacher_fixed_pair_rgb_attribution_20261102 as old
from . import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight
from . import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as q
from .reacher_d2p_goal_encoder_metric import GoalEncoderMetric

host, setup, ROOT = old.host, old.setup, old.ROOT
PROTOCOL = ROOT.parent / "docs/PC_REACHER_FIXED_PAIR_GOAL_Q_PROTOCOL_20261102.md"
AUTH = "REACHER_FIXED_PAIR_GOAL_Q_DEVELOPMENT"


def prepare():
    old.prepare()
    host.require(setup._sha256_file(preflight.Q_PATH) == preflight.Q_SHA, "goal Q hash")
    grams = json.loads(preflight.Q_PATH.read_text())["rows"]
    host.require([r["seed"] for r in grams] == [20263160, 20263161], "goal Q root order")
    original = json.loads(old.source.PLAN.read_text())
    files = [Path(__file__), PROTOCOL, preflight.Q_PATH, Path(q.__file__),
             Path(old.source.__file__), Path(old.__file__), Path(setup.p0.__file__),
             Path(original["weights"]), Path(original["stats"])]
    return {"schema": "reacher_fixed_pair_goal_q_v1",
            "files": {str(p): setup._sha256_file(p) for p in files},
            "replay_plan_sha256": old.PLAN_SHA, "replay_journal_sha256": old.JOURNAL_SHA,
            "replay_receipt_sha256": old.RECEIPT_SHA, "q_sha256": preflight.Q_SHA,
            "same_action_prediction_sha256": old.PRED_SHA, "model_hash": original["model_parameter_hash"],
            "weights": original["weights"], "stats": original["stats"], "pairs": 26,
            "P1_admission": False}


def run(plan_path, digest, output, authority):
    from experiments.gpu_preflight import require_cuda_allocation

    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID, "CUDA authority")
    plan_path, output = Path(plan_path), Path(output)
    host.require(setup._sha256_file(plan_path) == digest and json.loads(plan_path.read_text()) == prepare(),
                 "plan drift")
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_fixed_pair_goal_q_RUN_")
                 and not os.path.lexists(output), "exclusive output")
    replay_plan = json.loads(old.PLAN.read_text())
    events = old.replay.pair.audit.saved.verify_journal(old.JOURNAL)
    source_events = old.replay.pair.audit.saved.verify_journal(old.replay.pair.prior.L2_JOURNAL)
    source_results = json.loads(old.PRED.read_text())["rows"]
    grams = {r["seed"]: r["goal_metric"]["gram"] for r in json.loads(preflight.Q_PATH.read_text())["rows"]}
    with host.lock.gpu_lock(host.LOCK):
        allocation = require_cuda_allocation("cuda:0")
        host.require(torch.cuda.device_count() == 1 and host.canonical_gpu_uuid(
            torch.cuda.get_device_properties(0).uuid) == host.canonical_gpu_uuid(host.UUID), "GPU identity")
        original = json.loads(old.source.PLAN.read_text())
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
            metric = GoalEncoderMetric(torch.as_tensor(grams[root["seed"]], dtype=torch.float64, device="cuda"))
            for index in range(0, len(root["branches"]), 2):
                arms = root["branches"][index:index+2]
                host.require([a["label"] for a in arms] == ["l2", "direction"]
                             and arms[0]["prefix"] == arms[1]["prefix"], "pair order")
                contexts = [e["value"] for e in source_events if e["kind"] == "context"
                            and e.get("seed") == root["seed"] and e.get("arm") == arms[0]["arm"]
                            and e["role"] == "successor"]
                matches = [r for r in source_results if r["seed"] == root["seed"]
                           and r["prefix"] == arms[0]["prefix"]]
                host.require(len(contexts) == len(matches) == 1, "source context/score")
                value, saved = contexts[0], matches[0]
                goal = old.replay.context.unpack_rgb_observation(value["goal"])
                frames = [old.replay.context.unpack_rgb_observation(f) for f in value["frames"]]
                pixels, target_pixels, history = old.source.sensitivity.prepare_saved_frames(
                    setup.p0._prepare, processor, frames, goal, np.asarray(value["history"], np.float32),
                    torch.device("cuda"))
                futures = torch.as_tensor(np.asarray(saved["normalized_actions"], np.float32), device="cuda")
                with torch.no_grad():
                    target = setup.p0._planning_coordinate(
                        model, setup.p0._goal_latent(model, target_pixels), role="goal")
                    predicted = setup.p0._planning_coordinate(
                        model, old.source.direct_terminal(model, pixels, history, futures), role="predicted")
                actual, physical, final_hashes = [], [], []
                for offset, arm in enumerate(arms):
                    found = [e["outcome"] for e in events if e["kind"] == "branch_returned"
                             and e["seed"] == root["seed"] and e["index"] == index + offset]
                    host.require(len(found) == 1, "physical replay result")
                    points = found[0]["trajectory"]
                    actual_frames = [old.replay.context.unpack_rgb_observation(points[i]["rgb"])
                                     for i in (5, 10, 15)]
                    actual_pixels, _, actual_history = old.source.sensitivity.prepare_saved_frames(
                        setup.p0._prepare, processor, actual_frames, goal,
                        np.asarray(arm["actions"], np.float32).reshape(3, 10), torch.device("cuda"))
                    with torch.no_grad():
                        actual.append(setup.p0._planning_coordinate(
                            model, setup.p0._latent_context(model, actual_pixels, actual_history)[0, -1],
                            role="observed"))
                    physical.append(points[15]["qpos_distance"])
                    final_hashes.append(old.audit.checks.rgb(points[15]["rgb"]))
                actual = torch.stack(actual)
                with torch.no_grad():
                    predicted_scores = q.q_costs(metric, predicted, target)
                    actual_scores = q.q_costs(metric, actual, target)
                rows.append({"seed": root["seed"], "prefix": arms[0]["prefix"],
                             "cp5_rgb_sha256": arms[0]["cp5_rgb_sha256"], "final_rgb_sha256": final_hashes,
                             "actions_sha256": [a["actions_sha256"] for a in arms], "physical_final": physical,
                             "target": target.cpu().tolist(), "predicted": predicted.cpu().tolist(),
                             "actual": actual.cpu().tolist(), "predicted_q": predicted_scores.cpu().tolist(),
                             "actual_q": actual_scores.cpu().tolist()})
        host.require(len(rows) == 26 and setup._parameter_hash(model) == before
                     and json.loads(plan_path.read_text()) == prepare(), "postcheck")
        host.write_new(output, {"schema": "reacher_fixed_pair_goal_q_result_v1", "plan_sha256": digest,
                                "cuda_allocation": allocation, "model_before": before, "model_after": before,
                                "rows": rows, "P1_admission": False})
        print(json.dumps({"pairs": len(rows), "sha256": setup._sha256_file(output)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "run"))
    for arg in ("plan", "sha256", "output", "authority"):
        p.add_argument("--" + arg)
    args = p.parse_args()
    if args.command == "prepare":
        host.write_new(args.plan, prepare())
        print(json.dumps({"sha256": setup._sha256_file(Path(args.plan))}))
    else:
        run(args.plan, args.sha256, args.output, args.authority)


if __name__ == "__main__":
    main()
