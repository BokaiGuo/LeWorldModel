"""Paired predicted/observed successor CEM on sealed development contexts."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import reacher_response_svd_predicted_cem_20261102 as source
from . import reacher_response_svd_predicted_cem_physical_20261102 as physical
from . import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as saved
from . import reacher_p1_predicted_successor_cem as cem

host, setup = source.host, source.setup
ROOT = source.ROOT
JOURNAL = ROOT / "results/reacher_response_svd_cp5_RUN_20261102_v1/events.jsonl"
JOURNAL_SHA = "e9c933f8d78abc9787198e324240afdb1186475fa08189a2787e39a3b9480f43"
PROTOCOL = ROOT.parent / "docs/PC_REACHER_CEM_CONTEXT_SENSITIVITY_PROTOCOL_20261102.md"
AUTH = "REACHER_CEM_CONTEXT_SENSITIVITY_DEV_ONLY"


def prepare():
    host.require(setup._sha256_file(JOURNAL) == JOURNAL_SHA
                 and setup._sha256_file(physical.OLD_PLAN) == physical.OLD_PLAN_SHA, "source hashes")
    original = json.loads(physical.OLD_PLAN.read_text())
    host.require([r["seed"] for r in original["roots"]] == [20263160, 20263161], "root roster")
    paths = (Path(__file__), PROTOCOL, Path(cem.__file__), Path(saved.__file__),
             Path(setup.__file__), Path(setup.p0.__file__), physical.OLD_PLAN, JOURNAL)
    return {"schema": "reacher_cem_context_sensitivity_development_v1",
            "files": {str(p): setup._sha256_file(p) for p in paths},
            "source_plan_sha256": physical.OLD_PLAN_SHA, "source_journal_sha256": JOURNAL_SHA,
            "model_parameter_hash": original["model_parameter_hash"],
            "weights": original["weights"], "stats": original["stats"],
            "roots": [20263160, 20263161], "prefixes": [0, 12], "seed_offsets": [0, 1, 2],
            "budgets": [[16, 2, 4], [32, 3, 4]], "contexts": ["predicted", "observed"],
            "cem_invocations": 48, "model_predict_samples": 6144,
            "physical_steps": 0, "independent_confirmation": False, "P1_admission": False}


def contexts(events, roots):
    chosen = {}
    for root in roots:
        for index in (0, 12):
            arm = root["arms"][index]
            for role in ("root", "successor"):
                found = [e for e in events if e["kind"] == "context" and e["seed"] == root["seed"]
                         and e["arm"] == f"prefix_{index:02d}" and e["role"] == role]
                host.require(len(found) == 1, "context uniqueness")
                value = found[0]["value"]
                saved.verify_context(value, root, arm, role)
                chosen[(root["seed"], index, role)] = value
    return chosen


def run(plan_path, plan_sha, output, authority):
    import torch
    from experiments.gpu_preflight import require_cuda_allocation
    ctx = saved.ctx

    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID, "CUDA authority")
    plan_path, output = Path(plan_path), Path(output)
    host.require(setup._sha256_file(plan_path) == plan_sha and json.loads(plan_path.read_text()) == prepare(),
                 "plan/source drift")
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_cem_context_sensitivity_RUN_")
                 and not os.path.lexists(output), "exclusive output")
    original = json.loads(physical.OLD_PLAN.read_text())
    selected = contexts(saved.verify_journal(JOURNAL), original["roots"])
    with host.lock.gpu_lock(host.LOCK):
        allocation = require_cuda_allocation("cuda:0")
        host.require(torch.cuda.device_count() == 1 and host.canonical_gpu_uuid(
            torch.cuda.get_device_properties(0).uuid) == host.canonical_gpu_uuid(host.UUID), "GPU identity")
        output.mkdir(mode=0o700)
        journal = host.durable.Journal(output / "events.jsonl")
        try:
            journal.emit({"kind": "start", "plan_sha256": plan_sha, "cuda_allocation": allocation})
            setup.validate_training_semantic_receipt(Path(original["stats"]))
            mean, std = setup._load_action_stats(Path(original["stats"]), "swm/ReacherDMControl-v0")
            model = setup.load_compatible_pretrained(Path(original["weights"]).parent, device="cuda").eval()
            model.interpolate_pos_encoding = True
            model.requires_grad_(False)
            before = setup._parameter_hash(model)
            host.require(before == original["model_parameter_hash"], "model hash")
            setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
            low, high = setup._normalized_action_bounds(mean, std, np.array([-1., -1.], np.float32),
                                                         np.array([1., 1.], np.float32))
            lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
            processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
            for root in original["roots"]:
                for index in (0, 12):
                    values = []
                    for role in ("root", "successor"):
                        value = selected[(root["seed"], index, role)]
                        frames = [ctx.unpack_rgb_observation(x) for x in value["frames"]]
                        goal_rgb = ctx.unpack_rgb_observation(value["goal"])
                        values.append(setup.p0._prepare(processor, frames, frames[-1], goal_rgb,
                                      np.asarray(value["history"], np.float32), torch.device("cuda")))
                    pixels, goal, history = values[0]
                    observed_pixels, observed_goal, observed_history = values[1]
                    host.require(torch.equal(goal, observed_goal), "goal mismatch")
                    prefix = torch.as_tensor(root["arms"][index]["params"], device="cuda",
                                             dtype=history.dtype).unsqueeze(0)
                    with torch.no_grad():
                        pred_emb, pred_actions = setup._state_after_prefix(
                            model, pixels, history, prefix, action_lower=lower, action_upper=upper)
                        obs_emb = setup.p0._latent_context(model, observed_pixels, observed_history)
                        goal_latent = setup.p0._goal_latent(model, goal)
                    history_error = float((pred_actions - observed_history).abs().max())
                    host.require(history_error < 1e-5, "normalized executed history mismatch")
                    discrepancy = float((pred_emb - obs_emb).square().mean())
                    for samples, steps, elite in ((16, 2, 4), (32, 3, 4)):
                        for offset in (0, 1, 2):
                            seed = root["seed"] + 5000 + offset
                            pair = {}
                            for label, emb, actions in (("predicted", pred_emb, pred_actions),
                                                         ("observed", obs_emb, observed_history)):
                                started = time.monotonic()
                                suffix, receipt = cem.cem_from_predicted_successor(
                                    model, goal_latent, emb, actions, future_horizon=2,
                                    action_lower=lower, action_upper=upper, samples=samples,
                                    steps=steps, elite=elite, seed=seed)
                                torch.cuda.synchronize()
                                pair[label] = {"suffix_parameters": suffix[0].cpu().tolist(),
                                               "receipt": receipt, "wall_seconds": time.monotonic() - started}
                            journal.emit({"kind": "pair", "seed": root["seed"], "prefix": index,
                                          "cem_seed": seed, "budget": [samples, steps, elite],
                                          "context_mse": discrepancy, "history_max_abs_error": history_error,
                                          "parameter_max_abs_difference": float(np.max(np.abs(
                                              np.asarray(pair["predicted"]["suffix_parameters"]) -
                                              np.asarray(pair["observed"]["suffix_parameters"])))),
                                          "contexts": pair})
            after = setup._parameter_hash(model)
            host.require(after == before and json.loads(plan_path.read_text()) == prepare(), "postcheck drift")
            journal.emit({"kind": "finished", "model_before": before, "model_after": after})
        except BaseException as exc:
            journal.emit({"kind": "run_error", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            raise
        finally:
            journal.close()


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
