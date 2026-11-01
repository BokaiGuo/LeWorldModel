"""Model-only Reacher P1 prerequisite profile; never opens an efficacy cohort.

The damping grid is recorded before model evaluation. It diagnoses a fixed
root/bank family, not a selected physical-success subset or a new P1 pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0
from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability.audit_reacher_d2p_runtime_support import _source_state_context
from experiments.spectral_reachability.preflight_reacher_d2p_p0 import _normalized_action_bounds, _parameter_hash
from experiments.spectral_reachability.reacher_d2p_packet_support import load_packet_pool, proposal_parameters, receipt_source


def gap_summary(rows: list[dict], *, non_greedy_margin: float = 1e-6) -> dict:
    if not rows or [row["proposal_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("ordered complete proposal bank with zero at index zero required")
    if any(not np.isfinite(row[key]) for row in rows for key in ("immediate_goal_cost", "bounded_dls_suffix_goal_cost")):
        raise ValueError("finite model costs required")
    zero = rows[0]
    greedy = min(rows, key=lambda row: (row["immediate_goal_cost"], row["proposal_index"]))
    non_greedy = [row for row in rows[1:] if row["immediate_goal_cost"] >= zero["immediate_goal_cost"] + non_greedy_margin]
    witnesses = [row for row in non_greedy if row["bounded_dls_suffix_goal_cost"] <= zero["bounded_dls_suffix_goal_cost"]
                 and row["bounded_dls_suffix_goal_cost"] < greedy["bounded_dls_suffix_goal_cost"]]
    selected = min(witnesses, key=lambda row: (row["bounded_dls_suffix_goal_cost"], row["proposal_index"])) if witnesses else None
    best = min(non_greedy, key=lambda row: (row["bounded_dls_suffix_goal_cost"], row["proposal_index"])) if non_greedy else None
    return {"greedy_index": greedy["proposal_index"], "non_greedy_candidate_count": len(non_greedy),
            "strong_future_witness_count": len(witnesses), "selected_index": None if selected is None else selected["proposal_index"],
            "zero_future_cost": zero["bounded_dls_suffix_goal_cost"], "greedy_future_cost": greedy["bounded_dls_suffix_goal_cost"],
            "best_non_greedy_future_cost": None if best is None else best["bounded_dls_suffix_goal_cost"],
            "best_non_greedy_minus_greedy_future_cost": None if best is None else best["bounded_dls_suffix_goal_cost"] - greedy["bounded_dls_suffix_goal_cost"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--rgb-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dampings", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--suffix-step-rule", choices=p1.SUFFIX_STEP_RULES, default=p1.DEFAULT_SUFFIX_STEP_RULE)
    args = parser.parse_args()
    if args.output.exists() or len(set(args.dampings)) != len(args.dampings) or any(not np.isfinite(x) or x <= 0 for x in args.dampings):
        raise ValueError("new output and unique positive finite damping grid required")
    reference = json.loads(args.rgb_reference.read_text())
    if reference["status"] != "completed":
        raise ValueError("completed RGB reference required")
    policy = args.policy.resolve()
    weights = hashlib.sha256((policy / "weights.pt").read_bytes()).hexdigest()
    if not any(item["path"] == str(policy) and item["weights_sha256"] == weights for item in reference["inputs"]["policies"]):
        raise ValueError("RGB reference policy differs")
    if hashlib.sha256(args.adapter_artifact.read_bytes()).hexdigest() != reference["inputs"]["adapter_sha256"]:
        raise ValueError("RGB reference adapter differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {"schema": "reacher_d2p_model_gap_profile_protocol.v1", "dampings": args.dampings,
                "root_seeds": [row["seed"] for row in reference["rows"]], "future_horizon": 2,
                "reference_sha256": hashlib.sha256(args.rgb_reference.read_bytes()).hexdigest(),
                "weights_sha256": weights, "physical_candidates_executed": False,
                "suffix_step_rule": args.suffix_step_rule,
                "scope": "Fixed-root diagnostic only; no physical labels, no automatic damping selection, no efficacy root admission."}
    with args.output.with_suffix(".protocol.json").open("x") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    records = args.output.with_suffix(".roots")
    records.mkdir(exist_ok=False)
    host = require_cuda_allocation("cuda")
    install_missing_field_compat()
    device = torch.device("cuda")
    model = load_compatible_pretrained(policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    before = _parameter_hash(model)
    mean, std = _load_action_stats(args.adapter_artifact, "swm/ReacherDMControl-v0")
    dataset, digest = receipt_source(args.adapter_artifact)
    pool, support = load_packet_pool(dataset, expected_sha256=digest)
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"))
    lower_np, upper_np = _normalized_action_bounds(mean, std, env.action_space.low, env.action_space.high)
    lower, upper = torch.as_tensor(lower_np, device=device), torch.as_tensor(upper_np, device=device)
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    roots = []
    try:
        for source in reference["rows"]:
            seed = source["seed"]
            context = _source_state_context(env, dataset=dataset, seed=seed)
            if any(context[key] != source[key] for key in ("root_rgb_sha256", "goal_rgb_sha256", "source_episode", "source_start_row")):
                raise RuntimeError("reference root identity differs; no root may be dropped")
            count = len(source["candidate_raw_packets"])
            bank, bank_receipt = proposal_parameters(pool, proposals=count, horizon=1, seed=seed, mean=mean, std=std, lower=lower, upper=upper)
            if bank_receipt != source["proposal_support"]:
                raise RuntimeError("reference bank identity differs")
            pixels, goal, history = p0._prepare(preprocessor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], np.float32), device)
            with torch.no_grad():
                goal_z = p0._goal_latent(model, goal)
                immediate = p0._cost_fn(model, pixels, goal, history, bank, action_lower=lower, action_upper=upper)
            rows = []
            for index in range(count):
                with torch.no_grad():
                    state = p1._state_after_prefix(model, pixels, history, bank[index:index + 1], action_lower=lower, action_upper=upper)
                reach = p1._reachability_value(model, goal_z, *state, future_horizon=2, action_lower=lower,
                    action_upper=upper, damping=10.0, diagnostic_dampings=tuple(args.dampings), suffix_step_rule=args.suffix_step_rule)
                rows.append({"proposal_index": index, "immediate_goal_cost": float(immediate[index]), **reach})
            summary = []
            for damping in args.dampings:
                scored = [{"proposal_index": row["proposal_index"], "immediate_goal_cost": row["immediate_goal_cost"],
                           **next(item for item in row["diagnostic_damping_scores"] if item["damping"] == damping)} for row in rows]
                summary.append({"damping": damping, **gap_summary(scored)})
            root = {"seed": seed, "root_rgb_sha256": context["root_rgb_sha256"], "goal_rgb_sha256": context["goal_rgb_sha256"],
                    "summary": summary, "proposal_rows": rows}
            with (records / f"root_{seed}.json").open("x") as handle:
                handle.write(json.dumps(root, indent=2, sort_keys=True) + "\n")
            roots.append(root)
            print(json.dumps({"seed": seed, "zero_gap_fraction": rows[0]["first_order_unreachable_residual_fraction"], "summary": summary}), flush=True)
    finally:
        env.close()
    gates = {"cuda_only": True, "weights_unchanged": before == _parameter_hash(model), "all_reference_roots_profiled": len(roots) == len(reference["rows"]),
             "no_physical_candidate_or_outcome_labels": True}
    report = {"schema": "reacher_d2p_model_gap_profile.v1", "status": "completed" if all(gates.values()) else "fail",
              "protocol": protocol, "cuda_host": host, "gates": gates, "source_support": support,
              "aggregate": [{"damping": damping, "model_witness_root_count": sum(any(item["damping"] == damping and item["strong_future_witness_count"] > 0 for item in root["summary"]) for root in roots)} for damping in args.dampings],
              "roots": roots, "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "selector_sha256": hashlib.sha256(Path(p1.__file__).read_bytes()).hexdigest(),
              "claim_boundary": "Model-only prerequisite diagnostic, neither physical controllability nor P1 admission/efficacy. A positive cell needs a new calibration protocol and independent outcome roots."}
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
