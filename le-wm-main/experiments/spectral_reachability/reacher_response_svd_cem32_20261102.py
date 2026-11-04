"""Fixed response-SVD prefixes, model-only predicted-successor CEM selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import preflight_reacher_d2p_p1_reconfiguration as setup
from . import reacher_p1_predicted_successor_cem as selector
from . import reacher_pc_p1_common_tail_preflight_20261102_codexexclusive as preflight
from . import run_reacher_pc_common_tangent_cuda as verify
from . import reacher_pc_p1_fd_cp5_host_20261102_codexexclusive as host

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/reacher_pc_response_svd_cuda_20261102_v2/report.json"
SOURCE_SHA = "b3f4e0cab1c63c0c1f3d7c6cc43309a2d81e6644ea8748885569f38b097479ca"
PREFLIGHT = ROOT / "results/reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json"
PREFLIGHT_SHA = "b15a5e0de27e25004e86b7537c9c1761cd33c3d7e0cdfe10088fe8c993def2c3"
PROTOCOL = ROOT.parent / "docs/PC_REACHER_SVD_CEM32_SELECTION_PROTOCOL_20261102.md"
AUTH = "RESPONSE_SVD_CEM32_KNOWN_ROOT_DEVELOPMENT"


def make_plan():
    host.require(setup._sha256_file(SOURCE) == SOURCE_SHA
                 and setup._sha256_file(PREFLIGHT) == PREFLIGHT_SHA, "source hash")
    previous = json.loads(SOURCE.read_text())
    pf = json.loads(PREFLIGHT.read_text())
    verify.verify_preflight(pf, preflight.material())
    host.require(previous["frozen_parameters_unchanged"] and previous["plan_postcheck"]
                 and [r["seed"] for r in previous["rows"]] == [20263160, 20263161]
                 and all(r["status"] == "complete" and len(r["bank_params"]) == 13
                         for r in previous["rows"]), "source completion")
    paths = [Path(__file__), Path(selector.__file__), PROTOCOL, SOURCE, PREFLIGHT]
    return {"schema": "response_svd_predicted_cem32_v1", "files": {str(p): setup._sha256_file(p) for p in paths},
            "model_hash": previous["parameter_hash_before"], "roots": pf["roots"],
            "samples": 32, "iterations": 3, "elites": 4, "future_horizon": 2,
            "non_greedy_margin": 1e-6, "future_margin": 1e-6,
            "scope": "known roots and opened physical outcomes; no independent outcome or P1 admission"}


def run(plan_path, expected, output, authority):
    import gymnasium as gym
    import numpy as np
    import torch

    plan_path = Path(plan_path)
    host.require(authority == AUTH and os.environ.get("CUDA_VISIBLE_DEVICES") == host.UUID, "CUDA authority")
    host.require(setup._sha256_file(plan_path) == expected and json.loads(Path(plan_path).read_text()) == make_plan(),
                 "sealed plan")
    output = Path(output)
    host.require(output.is_absolute() and output.parent == ROOT / "results"
                 and output.name.startswith("reacher_response_svd_cem32_RUN_")
                 and not os.path.lexists(output), "exclusive output")
    with host.lock.gpu_lock(host.LOCK):
        from experiments.gpu_preflight import require_cuda_allocation
        allocation = require_cuda_allocation("cuda:0")
        host.require(torch.cuda.device_count() == 1 and host.canonical_gpu_uuid(
            torch.cuda.get_device_properties(0).uuid) == host.canonical_gpu_uuid(host.UUID), "GPU identity")
        pf, source = json.loads(PREFLIGHT.read_text()), json.loads(SOURCE.read_text())
        setup.install_missing_field_compat()
        stats = Path(pf["action_stats_path"])
        setup.validate_training_semantic_receipt(stats)
        mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
        _, histories, _, _ = setup.load_runtime_support(stats, history_len=3, action_block=5)
        model = setup.load_compatible_pretrained(Path(pf["weights_path"]).parent, device="cuda").eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        before = setup._parameter_hash(model)
        host.require(before == make_plan()["model_hash"], "model hash")
        processor = setup._Preprocessor(setup.ActionNormalizer(mean, std))
        env = setup.ReacherGoalImageWrapper(
            gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(.65, -.65))
        output.mkdir(mode=0o700)
        report = {"schema": "response_svd_predicted_cem32_result_v1", "plan_sha256": expected,
                  "model_before": before, "cuda_allocation": allocation, "rows": []}
        try:
            setup.p0._validate_reacher_action_protocol(model, mean, std, action_dim=2, action_block=5)
            low, high = setup._normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
            lower, upper = torch.as_tensor(low, device="cuda"), torch.as_tensor(high, device="cuda")
            for fixed in source["rows"]:
                seed = fixed["seed"]
                packets, index = setup.sample_history_packets(histories, seed=seed)
                ctx = setup.collect_executed_context(
                    env, seed=seed, history_len=3, action_block=5, warmup_packets=packets,
                    warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
                host.require(int(index) == fixed["history_index"]
                             and ctx["root_rgb_sha256"] == fixed["root_rgb_sha256"]
                             and ctx["goal_rgb_sha256"] == fixed["goal_rgb_sha256"], "root/goal/history identity")
                pixels, goal, history = setup.p0._prepare(
                    processor, ctx["frames"], ctx["root_frame"], ctx["goal"],
                    np.asarray(ctx["history_action_blocks"], np.float32), torch.device("cuda"))
                bank = torch.as_tensor(fixed["bank_params"], dtype=history.dtype, device="cuda")
                selected, details = selector.select_prefix_with_cem_suffix(
                    model, pixels, goal, history, bank, future_horizon=2, action_lower=lower,
                    action_upper=upper, samples=32, steps=3, elite=4, seed=seed + 5000,
                    non_greedy_margin=1e-6, future_margin=1e-6)
                greedy = min(details["proposal_rows"], key=lambda row: (row["immediate_goal_cost"],
                                                                        row["proposal_index"]))["proposal_index"]
                report["rows"].append({"seed": seed, "selection": details, "same_bank_greedy": greedy,
                                       "selected_params": None if selected is None else selected.cpu().tolist()})
                print(json.dumps({"seed": seed, "selected": details["selected_proposal_index"]}), flush=True)
            report["model_after"] = setup._parameter_hash(model)
            host.require(report["model_after"] == before and json.loads(Path(plan_path).read_text()) == make_plan(),
                         "postcheck drift")
            report.update(status="complete", independent_outcome=False, P1_admission=False)
        except Exception as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1000],
                          independent_outcome=False, P1_admission=False)
            raise
        finally:
            env.close()
            host.write_new(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    for name in ("plan", "sha256", "output", "authority"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.command == "prepare":
        host.write_new(args.plan, make_plan())
        print(json.dumps({"plan": args.plan, "sha256": setup._sha256_file(Path(args.plan))}))
    else:
        run(args.plan, args.sha256, args.output, args.authority)


if __name__ == "__main__":
    main()
