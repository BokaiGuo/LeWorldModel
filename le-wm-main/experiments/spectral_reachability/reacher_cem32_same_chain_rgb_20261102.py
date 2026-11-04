"""Encode actual final RGB for the sealed CEM32 physical development chains."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import reacher_cem32_same_chain_saved_20261102 as physical
from . import reacher_cem_context_sensitivity_v2_20261102 as sensitivity

exp = physical.experiment
host, setup, ROOT = exp.host, exp.setup, exp.ROOT
PLAN = ROOT / "results/reacher_cem32_same_chain_PLAN_20261102_v1.json"
PLAN_SHA = "16d1479985426b3274fc0ebe9023a4168a2b8fad57fc945e0eb4c59cc309eb36"
JOURNAL = ROOT / "results/reacher_cem32_same_chain_RUN_20261102_v1/events.jsonl"
JOURNAL_SHA = "e82ac38ee16484ee074d35cef11ee8d0efcf9ece50abfb1bb783f8c0df5c7f28"
RECEIPT = ROOT / "results/reacher_cem32_same_chain_SAVED_20261102_v1.json"
RECEIPT_SHA = "488c6a5bb01bf804749655848ab0cbb36fc6ba5d230bff39433ff17998b8666c"
PROTOCOL = ROOT.parent / "docs/PC_REACHER_CEM32_SAME_CHAIN_RGB_PROTOCOL_20261102.md"
AUTH = "REACHER_CEM32_SAME_CHAIN_RGB_DEVELOPMENT"


def bound_goal(source_contexts, root, ctx):
    """Use the root-wide goal from the available verified neutral context."""
    saved_goal = source_contexts[(root["seed"], 0, "root")]["goal"]
    goal = ctx.unpack_rgb_observation(saved_goal)
    host.require(ctx.rgb_digest(goal) == root["goal_rgb_sha256"], "goal identity")
    return goal


def prepare():
    for path, digest in ((PLAN, PLAN_SHA), (JOURNAL, JOURNAL_SHA), (RECEIPT, RECEIPT_SHA)):
        host.require(setup._sha256_file(path) == digest, "input hashes")
    plan = json.loads(PLAN.read_text())
    events = exp.run.saved.verify_journal(JOURNAL)
    recomputed = physical.validate(plan, events, PLAN_SHA)
    receipt = json.loads(RECEIPT.read_text())
    host.require(plan == exp.prepare() and receipt["rows"] == recomputed["rows"]
                 and receipt["journal_sha256"] == JOURNAL_SHA, "physical source binding")
    old = json.loads(exp.run.physical.OLD_PLAN.read_text())
    files = (Path(__file__), PROTOCOL, Path(physical.__file__), Path(sensitivity.__file__),
             Path(setup.__file__), Path(setup.p0.__file__), Path(old["weights"]), Path(old["stats"]))
    return {"schema": "reacher_cem32_same_chain_rgb_v1",
            "files": {str(path): setup._sha256_file(path) for path in files},
            "physical_plan_sha256": PLAN_SHA, "physical_journal_sha256": JOURNAL_SHA,
            "physical_receipt_sha256": RECEIPT_SHA, "model_hash": old["model_parameter_hash"],
            "weights": old["weights"], "stats": old["stats"], "frames": [5, 10, 15],
            "expected_arms": 6, "new_physical_steps": 0, "P1_admission": False}


def run(plan_path, expected, output, authority):
    import torch
    from experiments.gpu_preflight import require_cuda_allocation

    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID, "CUDA authority")
    plan_path, output = Path(plan_path), Path(output)
    host.require(setup._sha256_file(plan_path) == expected and json.loads(plan_path.read_text()) == prepare(),
                 "plan/source drift")
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_cem32_same_chain_rgb_RUN_") and not os.path.lexists(output),
                 "exclusive output")
    physical_plan = json.loads(PLAN.read_text())
    events = exp.run.saved.verify_journal(JOURNAL)
    source_contexts = sensitivity.contexts(exp.run.saved.verify_journal(sensitivity.JOURNAL),
                                            json.loads(exp.run.physical.OLD_PLAN.read_text())["roots"])
    with host.lock.gpu_lock(host.LOCK):
        allocation = require_cuda_allocation("cuda:0")
        host.require(torch.cuda.device_count() == 1 and host.canonical_gpu_uuid(
            torch.cuda.get_device_properties(0).uuid) == host.canonical_gpu_uuid(host.UUID), "GPU identity")
        output.mkdir(mode=0o700)
        journal = host.durable.Journal(output / "events.jsonl")
        try:
            journal.emit({"kind": "start", "plan_sha256": expected, "cuda_allocation": allocation})
            old = json.loads(exp.run.physical.OLD_PLAN.read_text())
            setup.validate_training_semantic_receipt(Path(old["stats"]))
            mean, std = setup._load_action_stats(Path(old["stats"]), "swm/ReacherDMControl-v0")
            model = setup.load_compatible_pretrained(Path(old["weights"]).parent, device="cuda").eval()
            model.interpolate_pos_encoding = True
            model.requires_grad_(False)
            before = setup._parameter_hash(model)
            host.require(before == old["model_parameter_hash"], "model hash")
            processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
            for root in physical_plan["roots"]:
                for index, branch in enumerate(root["branches"]):
                    rows = [e for e in events if e["kind"] == "branch_returned"
                            and e["seed"] == root["seed"] and e["branch_index"] == index]
                    host.require(len(rows) == 1 and rows[0]["outcome"]["status"] == "complete", "branch")
                    points = rows[0]["outcome"]["trajectory"]
                    frames = [exp.run.saved.ctx.unpack_rgb_observation(points[i]["rgb"]) for i in (5, 10, 15)]
                    goal = bound_goal(source_contexts, root, exp.run.saved.ctx)
                    history = np.asarray(branch["raw_actions"], np.float32).reshape(3, 10)
                    pixels, goal_tensor, actions = sensitivity.prepare_saved_frames(
                        setup.p0._prepare, processor, frames, goal, history, torch.device("cuda"))
                    with torch.no_grad():
                        actual = setup.p0._planning_coordinate(
                            model, setup.p0._latent_context(model, pixels, actions)[0, -1], role="observed")
                        target = setup.p0._planning_coordinate(model, setup.p0._goal_latent(model, goal_tensor),
                                                               role="goal")
                        actual_cost = float((actual - target).square().mean())
                    journal.emit({"kind": "arm", "seed": root["seed"], "role": branch["role"],
                                  "prefix": branch["prefix_index"], "actions_sha256": branch["actions_sha256"],
                                  "frame_hashes": [points[i]["rgb"]["sha256"] for i in (5, 10, 15)],
                                  "goal_sha256": root["goal_rgb_sha256"], "raw_history": history.tolist(),
                                  "predicted_cost": branch["predicted_terminal_cost"], "actual_cost": actual_cost,
                                  "actual_coordinate": actual.cpu().tolist(), "goal_coordinate": target.cpu().tolist(),
                                  "physical_final": points[15]["qpos_distance"], "encoder_calls": 2})
            after = setup._parameter_hash(model)
            host.require(before == after and json.loads(plan_path.read_text()) == prepare(), "postcheck drift")
            journal.emit({"kind": "finished", "model_before": before, "model_after": after})
        except BaseException as exc:
            journal.emit({"kind": "run_error", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            raise
        finally:
            journal.close()


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
        print(json.dumps({"sha256": setup._sha256_file(Path(args.output) / "events.jsonl")}))


if __name__ == "__main__":
    main()
