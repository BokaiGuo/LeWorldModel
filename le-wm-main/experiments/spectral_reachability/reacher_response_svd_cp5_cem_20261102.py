"""All fixed response-SVD prefixes, actual cp5 RGB, common CEM configuration."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host
from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_cem_bridge_20261102_codexexclusive as bridge
from experiments.spectral_reachability import run_reacher_pc_response_svd_physical_20261102 as source

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_REACHER_RESPONSE_SVD_CP5_CEM_PROTOCOL_20261102.md"
PHYSICAL_PLAN_SHA = "550f2f900a1ce617b0b79a4617875f8350d6b568d9ee5d08c1bad5cdabe65f4c"
AUTHORITY = "RESPONSE_SVD_ALL_PREFIX_CP5_CEM_DEVELOPMENT"


def make_plan(source_report, physical_plan, preflight):
    paths = [Path(x).resolve(strict=True) for x in (source_report, physical_plan, preflight)]
    host.require(bridge.sha_file(paths[0]) == source.SOURCE_SHA
                 and bridge.sha_file(paths[1]) == PHYSICAL_PLAN_SHA, "source hashes")
    report, physical, pf = (json.loads(path.read_text()) for path in paths)
    host.require(physical["source_sha256"] == source.SOURCE_SHA
                 and physical["preflight_sha256"] == bridge.sha_file(paths[2])
                 and report["frozen_parameters_unchanged"] and report["plan_postcheck"], "source lineage")
    host.require([r["seed"] for r in physical["roots"]] == list(source.ROOTS)
                 and [r["seed"] for r in report["rows"]] == list(source.ROOTS), "root roster")
    roots = []
    for root, row in zip(physical["roots"], report["rows"], strict=True):
        host.require(row["status"] == "complete" and row["compiler_receipt"]["strict_selected_index"] is None,
                     "fixed bank receipt")
        params = np.asarray(row["bank_params"], dtype=np.float32)
        prefixes = np.asarray(row["raw_actions"], dtype=np.float32)
        host.require(params.shape == (13, 1, 10) and prefixes.shape == (13, 5, 2), "bank shape")
        arms = []
        for index in range(13):
            branch = root["branches"][index * 5]
            host.require(branch["prefix_index"] == index and branch["tail_index"] == 0
                         and np.array_equal(np.asarray(branch["raw_actions"], dtype=np.float32)[:5], prefixes[index]),
                         "physical-plan prefix mismatch")
            arms.append({"arm": f"prefix_{index:02d}", "proposal_index": index,
                         "params": params[index].tolist(), "raw_prefix": prefixes[index].tolist(),
                         "raw_prefix_sha256": bridge.sha_array(prefixes[index])})
        roots.append({**{k: root[k] for k in ("seed", "root_rgb_sha256", "goal_rgb_sha256", "history_index",
                                               "warmup_packets", "warmup_sha256")}, "arms": arms})
    weights, stats = Path(pf["weights_path"]), Path(pf["action_stats_path"])
    host.require(bridge.sha_file(weights) == pf["weights_sha256"] and bridge.sha_file(stats) == pf["action_stats_sha256"],
                 "weights or stats hash")
    deps = [Path(__file__), PROTOCOL, Path(host.__file__), Path(bridge.__file__),
            Path(source.__file__), Path(host.durable.__file__), Path(host.lock.__file__),
            Path(host.context_lib.__file__)]
    for name in ("preflight_reacher_d2p_p1_reconfiguration.py", "run_reacher_d2p_p1_reconfiguration.py",
                 "run_reacher_d2p_p0_goal_progress.py", "reacher_d2p_packet_support.py"):
        deps.append(Path(__file__).parent / name)
    return {"schema": "response_svd_all_prefix_cp5_cem_v1", "inputs": [str(p) for p in paths],
            "files": {str(p): bridge.sha_file(p) for p in paths + [weights, stats] + deps},
            "roots": roots, "weights": str(weights), "stats": str(stats),
            "model_parameter_hash": pf["model_parameter_hash_expected"],
            "cem": bridge.CEM, "planned_arms": 26, "gpu_uuid": host.UUID,
            "python": platform.python_version(), "source_closure_complete": False,
            "P1_admission": False, "scope": "two exposed roots; all fixed prefixes; model-only CEM budget"}


def run(plan_path, expected, output, authority):
    host.require(authority == AUTHORITY and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID,
                 "coordinated CUDA authority and UUID required")
    host.require(bridge.sha_file(plan_path) == expected, "plan hash mismatch")
    plan = json.loads(Path(plan_path).read_text())
    host.require(plan == make_plan(*plan["inputs"]), "source drift")
    output = Path(output)
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_response_svd_cp5_") and not os.path.lexists(output),
                 "new exclusive output required")
    with host.lock.gpu_lock(host.LOCK):
        import gymnasium as gym
        import torch
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
        from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner

        allocation = require_cuda_allocation("cuda:0")
        host.require(torch.cuda.device_count() == 1
                     and host.canonical_gpu_uuid(torch.cuda.get_device_properties(0).uuid)
                     == host.canonical_gpu_uuid(host.UUID), "CUDA UUID mismatch")
        output.mkdir(mode=0o700)
        host.write_new(output / "plan.json", plan)
        journal = host.durable.Journal(output / "events.jsonl")
        env = None
        try:
            journal.emit({"kind": "run_started", "plan_sha256": expected, "gpu_uuid": host.UUID,
                          "cuda_allocation": allocation})
            setup.install_missing_field_compat()
            stats = Path(plan["stats"])
            setup.validate_training_semantic_receipt(stats)
            mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
            model = setup.load_compatible_pretrained(Path(plan["weights"]).parent, device="cuda").eval()
            model.interpolate_pos_encoding = True
            model.requires_grad_(False)
            before = setup._parameter_hash(model)
            host.require(before == plan["model_parameter_hash"], "model hash")
            processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
            env = setup.ReacherGoalImageWrapper(
                gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
            setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
            low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
            lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
            for root in plan["roots"]:
                journal.emit({"kind": "root_started", "seed": root["seed"]})
                try:
                    packets = np.asarray(root["warmup_packets"], dtype=np.float32)
                    context = setup.collect_executed_context(
                        env, seed=root["seed"], history_len=3, action_block=5, warmup_packets=packets,
                        warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                    bridge.check_context(context, root)
                    original = host.instrument(runner, journal, root["seed"])
                    try:
                        for arm in root["arms"]:
                            name, index = arm["arm"], arm["proposal_index"]
                            try:
                                params = torch.as_tensor(arm["params"], dtype=torch.float32, device="cuda").unsqueeze(0)
                                raw, _ = setup.p0._to_raw_actions(params, mean, std, lower, upper,
                                                                   env.action_space.low, env.action_space.high)
                                host.require(bridge.sha_array(raw) == arm["raw_prefix_sha256"]
                                             and np.array_equal(raw, np.asarray(arm["raw_prefix"], dtype=np.float32)),
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
                                    cem_samples=16, cem_steps=2, cem_elite=4)
                                host.require(np.array_equal(np.asarray(result["first_action_witness_raw"], dtype=np.float32),
                                                            raw), "executed prefix mismatch")
                                host.require(result["suffix_receipt"]["forward_rollout_evaluations"] == 32,
                                             "CEM budget mismatch")
                            except Exception as exc:
                                # The instrumented executor records its own errors; preserve
                                # pre-dispatch errors here as well, without dropping later arms.
                                journal.emit({"kind": "arm_final_error", "seed": root["seed"], "arm": name,
                                              "error_type": type(exc).__name__, "error": str(exc)[:1000]})
                    finally:
                        runner._two_stage = original
                    journal.emit({"kind": "root_finished", "seed": root["seed"]})
                except Exception as exc:
                    journal.emit({"kind": "root_error", "seed": root["seed"],
                                  "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            after = setup._parameter_hash(model)
            host.require(before == after and plan == make_plan(*plan["inputs"]), "postcheck drift")
            journal.emit({"kind": "run_finished", "model_before": before, "model_after": after})
        except Exception as exc:
            journal.emit({"kind": "run_error", "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            raise
        finally:
            if env is not None:
                env.close()
            journal.close()
    return {"output": str(output), "journal_sha256": bridge.sha_file(output / "events.jsonl")}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    for key in ("source-report", "physical-plan", "preflight", "output"):
        prep.add_argument("--" + key, required=True)
    execute = sub.add_parser("run")
    for key in ("plan", "sha256", "output", "authority"):
        execute.add_argument("--" + key, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        host.write_new(args.output, make_plan(args.source_report, args.physical_plan, args.preflight))
        print(json.dumps({"plan": args.output, "sha256": bridge.sha_file(args.output)}))
    else:
        print(json.dumps(run(args.plan, args.sha256, args.output, args.authority)))


if __name__ == "__main__":
    main()
