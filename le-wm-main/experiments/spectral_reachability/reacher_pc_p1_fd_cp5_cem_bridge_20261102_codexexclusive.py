"""FD fixed-prefix to actual-cp5 CEM bridge; no execution on import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_SHA = "b76a55c6b6f8e13e7c4cbba0887e6ec6babc062c1a3245ddabf3991d39b00401"
AUDIT_SHA = "127300623499118458970a433521e5c8a3f2c1e00c55c9d2ba69986c11f7ce9c"
PHYSICAL_PLAN_SHA = "6de6f2ed64c80af6dd66eb27a8db920cffeeaf8d0c3bc82c3274dff6e305456a"
ROOTS = (20263160, 20263161)
ARMS = (("selected", 4), ("parameter_zero", 0))
CEM = {"samples": 16, "steps": 2, "elite": 4, "horizon_packets": 2,
       "seed_rule": "root_seed_plus_5000", "rollout_candidates": 32, "predict_token_samples": 64}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def raw_array(value, shape):
    a = np.asarray(value, dtype=np.float32)
    require(a.shape == shape and np.isfinite(a).all(), "invalid array")
    return a


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def prepare(model_path, audit_path, physical_plan_path):
    """Construct development-only fixed actions from pre-outcome model receipts."""
    for path, expected in ((model_path, MODEL_SHA), (audit_path, AUDIT_SHA),
                           (physical_plan_path, PHYSICAL_PLAN_SHA)):
        require(sha_file(path) == expected, "source SHA mismatch")
    model, audited, physical = [json.loads(Path(p).read_text()) for p in
                                (model_path, audit_path, physical_plan_path)]
    require(physical["model_sha256"] == audited["report_sha256"] == MODEL_SHA
            and physical["audit_sha256"] == AUDIT_SHA
            and audited["status"] == "complete_saved_consistency"
            and audited["P1_admission"] is False, "source lineage mismatch")
    for rows in (model["rows"], audited["rows"], physical["roots"]):
        require(tuple(r["seed"] for r in rows) == ROOTS, "wrong development roots")
    roots = []
    for source, checked, root in zip(model["rows"], audited["rows"], physical["roots"], strict=True):
        receipt = source["compiler_receipt"]
        require(source["status"] == "complete" and checked["status"] == "verified_saved_consistency"
                and receipt["development_selected_index"] == checked["development_selected_index"] == 4
                and receipt["strict_selected_index"] is checked["strict_selected_index"] is None,
                "model selection mismatch")
        for field in ("root_rgb_sha256", "goal_rgb_sha256", "history_index"):
            require(source[field] == root[field], "root identity mismatch")
        bank = raw_array(source["bank_params"], (9, 1, 10))
        prefix = raw_array(source["prefix_raw"], (9, 5, 2))
        warmup = raw_array(root["warmup_packets"], (3, 5, 2))
        require(sha_array(warmup) == root["warmup_sha256"], "warmup hash mismatch")
        branches = []
        for arm, index in ARMS:
            reference = root["branches"][0 if arm == "selected" else 5]
            planned = raw_array(reference["raw_actions"], (15, 2))
            require(reference["arm"] == arm and reference["prefix_index"] == index
                    and np.array_equal(planned[:5], prefix[index]), "sealed prefix mismatch")
            branches.append({"arm": arm, "proposal_index": index,
                             "params": bank[index].tolist(), "raw_prefix": prefix[index].tolist(),
                             "raw_prefix_sha256": sha_array(prefix[index])})
        roots.append({"seed": root["seed"], "root_rgb_sha256": root["root_rgb_sha256"],
                      "goal_rgb_sha256": root["goal_rgb_sha256"], "history_index": root["history_index"],
                      "warmup_packets": warmup.tolist(), "warmup_sha256": root["warmup_sha256"],
                      "arms": branches})
    return {"schema": "fd_cp5_cem_bridge_development_v1", "model_report_sha256": MODEL_SHA,
            "model_audit_sha256": AUDIT_SHA, "physical_action_plan_sha256": PHYSICAL_PLAN_SHA,
            "physical_outcome_used": False, "P1_admission": False, "roots": roots,
            "cem": CEM.copy(), "planned_arms": 4, "parameter_hash_expected": model["parameter_hash_before"]}


def require_host_cuda(model, device):
    import torch
    require(torch.device(device).type == "cuda", "host CUDA required; CPU inference forbidden")
    parameters = list(model.parameters())
    require(not model.training and bool(parameters)
            and all(not p.requires_grad and p.device.type == "cuda" for p in parameters),
            "frozen CUDA model required")


def check_context(context, root):
    require(context["root_rgb_sha256"] == root["root_rgb_sha256"]
            and context["goal_rgb_sha256"] == root["goal_rgb_sha256"]
            and context["warmup_actions_sha256"] == root["warmup_sha256"], "context identity mismatch")
    warmup = raw_array(root["warmup_packets"], (3, 5, 2))
    observed = raw_array(context["warmup_raw_steps"], (15, 2))
    require(np.array_equal(observed, warmup.reshape(15, 2)) and len(context["frames"]) == 3
            and sha_array(observed) == root["warmup_sha256"], "context warmup mismatch")


def run_root(root, *, model, env, processor, mean, std, lower, upper, device, parameter_hash_expected):
    """Run both prescribed arms from newly reconstructed actual RGB/action context.

    Requires a separately reviewed host launcher and GPU lock. This function
    never reads a physical-outcome report and never passes qpos to the planner.
    """
    require_host_cuda(model, device)
    import torch
    from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
    from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner

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
            params_np = raw_array(arm["params"], (1, 10))
            params = torch.as_tensor(params_np, dtype=torch.float32, device=device).unsqueeze(0)
            raw, _ = setup.p0._to_raw_actions(params, mean, std, lower, upper,
                                               env.action_space.low, env.action_space.high)
            require(sha_array(raw) == arm["raw_prefix_sha256"]
                    and np.array_equal(raw, raw_array(arm["raw_prefix"], (5, 2))), "prefix action mismatch")

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
            "model_before": before, "model_after": after, "arms": results,
            "cem": CEM.copy(), "expected_arms": 2,
            "physical_outcome_used_for_selection": False, "P1_admission": False}
