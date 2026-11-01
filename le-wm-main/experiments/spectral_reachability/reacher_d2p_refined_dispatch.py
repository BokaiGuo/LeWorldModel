"""Full-bank, fine8-consistent model-only reconfiguration compiler.

Do not mistake a candidate's advantage under a weak single-step controller
for controllability change. Every model-non-greedy candidate and both fixed
controls receive the identical nonlinear refined SVD suffix. Reobserve and
use this same solver on deployment. This does not guarantee physical ranking.
"""
from __future__ import annotations

import math

import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import reacher_d2p_iterated_suffix as solver
from experiments.spectral_reachability import reacher_d2p_suffix_step_rescue as rescue
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


POLICY = "full_bank_common_fine8_suffix_reconfiguration_else_greedy_v1"


def select_refined_bank(rows, *, non_greedy_margin=1e-6):
    if len(rows) < 2 or [row["proposal_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("complete ordered proposal bank required")
    if not math.isfinite(non_greedy_margin) or non_greedy_margin < 0:
        raise ValueError("finite nonnegative margin required")
    if any(not math.isfinite(float(row["immediate_goal_cost"])) for row in rows):
        raise ValueError("finite immediate costs required")
    greedy = min(range(len(rows)), key=lambda index: (rows[index]["immediate_goal_cost"], index))
    eligible = [index for index in range(1, len(rows))
        if rows[index]["immediate_goal_cost"] >= rows[0]["immediate_goal_cost"] + non_greedy_margin]
    required = set(eligible) | {0, greedy}
    if any(rows[index]["refined_goal_cost"] is None or not math.isfinite(float(rows[index]["refined_goal_cost"])) for index in required):
        raise ValueError("all non-greedy candidates and both controls must be refined")
    winners = [index for index in eligible if rows[index]["refined_goal_cost"] < rows[0]["refined_goal_cost"]
        and rows[index]["refined_goal_cost"] < rows[greedy]["refined_goal_cost"]]
    chosen = min(winners, key=lambda index: (rows[index]["refined_goal_cost"], index)) if winners else greedy
    return {"policy": POLICY, "route": "reconfiguration" if winners else "same_bank_goal_progress",
        "selected_program_index": chosen, "same_bank_greedy_index": greedy,
        "reconfiguration_witness_found": bool(winners), "physical_labels_used": False,
        "non_greedy_candidate_count": len(eligible), "refined_proposal_indices": sorted(required),
        "refined_greedy_gain": rows[greedy]["refined_goal_cost"] - rows[chosen]["refined_goal_cost"]}


def compile_refined_dispatch(model, pixels, goal, history, *, proposal_bank, action_lower,
                             action_upper, damping=.01, non_greedy_margin=1e-6, progress=None):
    bank = proposal_bank.detach().to(device=history.device, dtype=history.dtype)
    if bank.ndim != 3 or bank.shape[0] < 2 or bank.shape[1] != 1 or bank.shape[2] != history.shape[-1] or not torch.isfinite(bank).all() or torch.count_nonzero(bank[0]):
        raise ValueError("finite one-packet full bank with exact zero control required")
    with torch.no_grad():
        immediate = p0._cost_fn(model, pixels, goal, history, bank, action_lower=action_lower, action_upper=action_upper)
    if not torch.isfinite(immediate).all():
        raise RuntimeError("non-finite prefix scores")
    goal_latent = p0._goal_latent(model, goal)
    greedy = int(immediate.argmin())
    candidates = {index for index in range(1, len(bank)) if float(immediate[index]) >= float(immediate[0]) + non_greedy_margin}
    required = sorted(candidates | {0, greedy})
    rows = [{"proposal_index": index, "immediate_goal_cost": float(immediate[index]), "refined_goal_cost": None}
        for index in range(len(bank))]
    backward_rows = predict_samples = 0
    for completed, index in enumerate(required, start=1):
        emb, actions = p1._state_after_prefix(model, pixels, history, bank[index:index + 1],
            action_lower=action_lower, action_upper=action_upper)
        suffix, receipt = solver.compile_snapshots(model, goal_latent, emb, actions, future_horizon=2,
            action_lower=action_lower, action_upper=action_upper, damping=damping, checkpoints=(8,),
            accept_update=rescue.accept_with_fine_rescue)[8]
        rows[index].update({"refined_goal_cost": receipt["trace"][-1]["cost_after"], "suffix_receipt": receipt,
            "imagined_successor_latent": emb[0, -1].detach().cpu().tolist(),
            "imagined_suffix_params": suffix[0].cpu().tolist()})
        backward_rows += receipt["backward_vjp_rows"]
        predict_samples += 1 + receipt["model_predict_sample_evaluations"]
        if progress is not None:
            progress(completed, len(required))
    decision = select_refined_bank(rows, non_greedy_margin=non_greedy_margin)
    chosen = decision["selected_program_index"]
    return bank[chosen:chosen + 1], {"program_type": POLICY, "dispatch": decision, "proposal_rows": rows,
        "suffix_solver": "iterated_singleton_svd_dls_fine_stop_rescue_8", "damping": damping,
        "non_greedy_margin": non_greedy_margin, "model_predict_sample_evaluations": len(bank) + predict_samples,
        "backward_vjp_rows": backward_rows,
        "scope": "Full-bank model score/solver consistency, not guaranteed physical non-greedy or P1 admission"}
