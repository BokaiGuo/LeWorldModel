"""Float64-preserving replacement for the fixed-prefix bridge, development only."""
import json
from pathlib import Path

import numpy as np

from .reacher_pc_p1_fd_cp5_cem_bridge_20261102_codexexclusive import (
    ROOTS, ARMS, CEM, require, sha_file, sha_array, raw_array, check_context, require_host_cuda,
)
from . import reacher_pc_p1_fd_cp5_cem_bridge_20261102_codexexclusive as original


def prepare(model_path, audit_path, physical_plan_path):
    manifest = original.prepare(model_path, audit_path, physical_plan_path)
    source = json.loads(Path(model_path).read_text())
    for row, root in zip(source["rows"], manifest["roots"], strict=True):
        require(row["seed"] == root["seed"], "source root mismatch")
        bank = np.asarray(row["bank_params"], dtype=np.float64)
        require(bank.shape == (9, 1, 10) and np.isfinite(bank).all(), "invalid float64 bank")
        for arm in root["arms"]:
            arm["params"] = bank[arm["proposal_index"]].tolist()
    manifest["schema"] = "fd_cp5_cem_bridge_f64_development_v2"
    manifest["parameter_dtype"] = "float64"
    return manifest


def decode_arm(arm, *, mean, std, lower, upper, raw_low, raw_high, device):
    """Return parameters only after exact original raw-action/hash checks."""
    import torch
    from . import preflight_reacher_d2p_p1_reconfiguration as setup
    values = np.asarray(arm["params"], dtype=np.float64)
    require(values.shape == (1, 10) and np.isfinite(values).all(), "invalid float64 params")
    params = torch.as_tensor(values, dtype=torch.float64, device=device).unsqueeze(0)
    raw, _ = setup.p0._to_raw_actions(params, mean, std, lower, upper, raw_low, raw_high)
    require(sha_array(raw) == arm["raw_prefix_sha256"]
            and np.array_equal(raw, raw_array(arm["raw_prefix"], (5, 2))), "prefix action mismatch")
    return params, raw


def run_root(root, *, model, env, processor, mean, std, lower, upper, device, parameter_hash_expected):
    require_host_cuda(model, device)
    import torch
    from . import preflight_reacher_d2p_p1_reconfiguration as setup
    from . import run_reacher_d2p_p1_reconfiguration as runner
    require(root["seed"] in ROOTS and [(a["arm"], a["proposal_index"]) for a in root["arms"]] == list(ARMS),
            "fixed arm roster mismatch")
    before = setup._parameter_hash(model)
    require(before == parameter_hash_expected, "frozen model identity mismatch")
    packets = raw_array(root["warmup_packets"], (3, 5, 2))
    require(sha_array(packets) == root["warmup_sha256"], "warmup hash mismatch")
    context = setup.collect_executed_context(
        env, seed=root["seed"], history_len=3, action_block=5, warmup_packets=packets,
        warmup_policy=setup.SOURCE_WARMUP_POLICY + ".contiguous_history")
    check_context(context, root)
    results = []
    for arm in root["arms"]:
        record = {"arm": arm["arm"], "proposal_index": arm["proposal_index"], "status": "started",
                  "suffix_seed": root["seed"] + 5000}
        results.append(record)
        try:
            params, raw = decode_arm(arm, mean=mean, std=std, lower=lower, upper=upper,
                                     raw_low=env.action_space.low, raw_high=env.action_space.high, device=device)

            def fixed_first(pixels, goal, history, root_seed):
                require(root_seed == root["seed"], "unexpected root seed")
                return params.clone(), {"program_type": "sealed_fd_fixed_prefix",
                                        "proposal_index": arm["proposal_index"],
                                        "raw_prefix_sha256": arm["raw_prefix_sha256"]}

            result = runner._two_stage(
                name=arm["arm"], first_program=fixed_first, env=env, seed=root["seed"], context=context,
                preprocessor=processor, model=model, mean=mean, std=std, lower=lower, upper=upper,
                action_block=5, progress_horizon=2, damping=0.0, device=torch.device(device),
                progress_solver="cem", cem_samples=16, cem_steps=2, cem_elite=4)
            require(np.array_equal(raw_array(result["first_action_witness_raw"], (5, 2)), raw),
                    "executed prefix witness mismatch")
            require(result["suffix_receipt"]["forward_rollout_evaluations"] == 32,
                    "CEM candidate budget mismatch")
            record.update(status="complete", result=result)
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__, error=str(exc)[:1000])
    after = setup._parameter_hash(model)
    require(after == before, "model parameter drift")
    return {"seed": root["seed"], "root_rgb_sha256": context["root_rgb_sha256"],
            "goal_rgb_sha256": context["goal_rgb_sha256"], "warmup_sha256": root["warmup_sha256"],
            "model_before": before, "model_after": after, "arms": results, "cem": CEM.copy(),
            "expected_arms": 2, "physical_outcome_used_for_selection": False, "P1_admission": False}
