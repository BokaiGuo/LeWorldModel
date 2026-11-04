"""Execute fixed development prefixes, reobserve RGB, and replan CEM online."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import reacher_response_svd_cp5_cem_20261102 as base
from . import reacher_cem_context_sensitivity_v2_20261102 as sensitivity

host, bridge, ROOT = base.host, base.bridge, base.ROOT
OLD_PLAN = ROOT / "results/reacher_response_svd_cp5_PLAN_20261102_v1.json"
OLD_SHA = "f1087157d9cb4f6d4efdfc9900c74fed66a28f33169835e61093033c79077014"
PROTOCOL = ROOT.parent / "docs/PC_REACHER_CEM_LIVE_CP5_PROTOCOL_20261102.md"
AUTH = "REACHER_CEM_LIVE_CP5_DEVELOPMENT"


def prepare():
    host.require(bridge.sha_file(OLD_PLAN) == OLD_SHA, "source plan hash")
    old = json.loads(OLD_PLAN.read_text())
    host.require(old == base.make_plan(*old["inputs"]), "source plan drift")
    roots = []
    for root in old["roots"]:
        arms = [root["arms"][i] for i in (0, 12)]
        host.require([a["proposal_index"] for a in arms] == [0, 12], "prefix order")
        roots.append({**{k: v for k, v in root.items() if k != "arms"}, "arms": arms})
    paths = (Path(__file__), PROTOCOL, Path(base.__file__), Path(sensitivity.__file__))
    return {"schema": "reacher_cem_live_cp5_development_v1", "source_plan_sha256": OLD_SHA,
            "sources": {str(path): bridge.sha_file(path) for path in paths}, "roots": roots,
            "weights": old["weights"], "stats": old["stats"], "model_parameter_hash": old["model_parameter_hash"],
            "cem": {"samples": 32, "steps": 3, "elite": 4, "seed_rule": "root+5000"},
            "expected_arms": 4, "gpu_uuid": host.UUID, "independent_confirmation": False, "P1_admission": False}


def run(plan_path, expected, output, authority):
    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID, "CUDA authority")
    plan_path, output = Path(plan_path), Path(output)
    host.require(bridge.sha_file(plan_path) == expected and json.loads(plan_path.read_text()) == prepare(),
                 "plan/source drift")
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_cem_live_cp5_RUN_") and not os.path.lexists(output),
                 "exclusive output")
    with host.lock.gpu_lock(host.LOCK):
        import gymnasium as gym
        import torch
        from experiments.gpu_preflight import require_cuda_allocation
        from . import preflight_reacher_d2p_p1_reconfiguration as setup
        from . import run_reacher_d2p_p1_reconfiguration as runner

        allocation = require_cuda_allocation("cuda:0")
        host.require(torch.cuda.device_count() == 1 and host.canonical_gpu_uuid(
            torch.cuda.get_device_properties(0).uuid) == host.canonical_gpu_uuid(host.UUID), "GPU identity")
        plan = json.loads(plan_path.read_text())
        output.mkdir(mode=0o700)
        host.write_new(output / "plan.json", plan)
        journal = host.durable.Journal(output / "events.jsonl")
        env = None
        try:
            journal.emit({"kind": "run_started", "plan_sha256": expected, "gpu_uuid": host.UUID,
                          "cuda_allocation": allocation})
            setup.install_missing_field_compat()
            setup.validate_training_semantic_receipt(Path(plan["stats"]))
            mean, std = setup._load_action_stats(Path(plan["stats"]), "swm/ReacherDMControl-v0")
            model = setup.load_compatible_pretrained(Path(plan["weights"]).parent, device="cuda").eval()
            model.interpolate_pos_encoding = True
            model.requires_grad_(False)
            before = setup._parameter_hash(model)
            host.require(before == plan["model_parameter_hash"], "model hash")
            processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
            env = setup.ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"),
                                                 target_delta=(.65, -.65))
            setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
            low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
            lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
            for root in plan["roots"]:
                journal.emit({"kind": "root_started", "seed": root["seed"]})
                try:
                    packets = np.asarray(root["warmup_packets"], np.float32)
                    context = setup.collect_executed_context(
                        env, seed=root["seed"], history_len=3, action_block=5, warmup_packets=packets,
                        warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                    bridge.check_context(context, root)
                    original = host.instrument(runner, journal, root["seed"])
                    try:
                        for arm in root["arms"]:
                            name, index = arm["arm"], arm["proposal_index"]
                            try:
                                params = torch.as_tensor(arm["params"], dtype=torch.float32,
                                                         device="cuda").unsqueeze(0)
                                raw, _ = setup.p0._to_raw_actions(params, mean, std, lower, upper,
                                                                 env.action_space.low, env.action_space.high)
                                host.require(bridge.sha_array(raw) == arm["raw_prefix_sha256"]
                                             and np.array_equal(raw, np.asarray(arm["raw_prefix"], np.float32)),
                                             "prefix decode mismatch")

                                def first(pixels, goal, history, seed, params=params, index=index):
                                    host.require(seed == root["seed"], "seed mismatch")
                                    return params.clone(), {"program_type": "fixed_response_svd",
                                                            "proposal_index": index}

                                result = runner._two_stage(
                                    name=name, first_program=first, env=env, seed=root["seed"], context=context,
                                    preprocessor=processor, model=model, mean=mean, std=std, lower=lower,
                                    upper=upper, action_block=5, progress_horizon=2, damping=0.0,
                                    device=torch.device("cuda"), progress_solver="cem",
                                    cem_samples=32, cem_steps=3, cem_elite=4)
                                host.require(np.array_equal(np.asarray(result["first_action_witness_raw"], np.float32),
                                                            raw)
                                             and result["suffix_receipt"]["forward_rollout_evaluations"] == 96,
                                             "prefix or CEM budget")
                            except Exception as exc:
                                journal.emit({"kind": "arm_final_error", "seed": root["seed"], "arm": name,
                                              "error_type": type(exc).__name__, "error": str(exc)[:1000]})
                    finally:
                        runner._two_stage = original
                    journal.emit({"kind": "root_finished", "seed": root["seed"]})
                except Exception as exc:
                    journal.emit({"kind": "root_error", "seed": root["seed"],
                                  "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            host.require(setup._parameter_hash(model) == before and json.loads(plan_path.read_text()) == prepare(),
                         "postcheck drift")
            journal.emit({"kind": "run_finished", "model_before": before, "model_after": before})
        except Exception as exc:
            journal.emit({"kind": "run_error", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            raise
        finally:
            if env is not None:
                env.close()
            journal.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    for name in ("plan", "sha256", "output", "authority"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.command == "prepare":
        host.write_new(args.plan, prepare())
        print(json.dumps({"sha256": bridge.sha_file(args.plan)}))
    else:
        run(args.plan, args.sha256, args.output, args.authority)
        print(json.dumps({"journal_sha256": bridge.sha_file(Path(args.output) / "events.jsonl")}))


if __name__ == "__main__":
    main()
