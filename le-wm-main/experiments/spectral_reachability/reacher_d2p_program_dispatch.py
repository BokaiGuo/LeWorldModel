"""Model-only dispatch; a failed reconfiguration search is not a stop command.

The legacy neutral-prefix P1 control is unchanged. This is a separately named
runtime policy: use a strong reconfiguration witness, otherwise execute the
same-bank immediate-goal progress program. Physical labels are not inputs.
"""
from __future__ import annotations

import math

import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as compiler

POLICY = "model_only_reconfiguration_else_same_bank_progress_v1"


def dispatch_decision(receipt: dict) -> dict:
    if receipt.get("future_reference") != "same_bank_greedy" or receipt.get("recompiled_suffix_solver") != compiler.SUFFIX_SOLVER:
        raise ValueError("dispatch requires the current strong-reference suffix compiler")
    metric = receipt.get("selection_metric")
    if metric not in ("analytic_v_reach", "bounded_dls_suffix_goal_cost"):
        raise ValueError("dispatch requires an explicit supported model selection metric")
    rows = receipt.get("proposal_rows", [])
    if len(rows) < 2 or [row["proposal_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("complete ordered model proposal bank required")
    if any(not math.isfinite(float(row["immediate_goal_cost"])) for row in rows):
        raise ValueError("finite model costs required")
    greedy = min(range(len(rows)), key=lambda index: (rows[index]["immediate_goal_cost"], index))
    if receipt.get("future_reference_proposal_index") != greedy:
        raise ValueError("same-bank greedy reference identity differs")
    witness = receipt.get("non_greedy_witness_found")
    if not isinstance(witness, bool):
        raise ValueError("explicit model witness boolean required")
    selected = receipt.get("selected_proposal_index")
    if witness:
        if type(selected) is not int or not 0 < selected < len(rows):
            raise ValueError("model witness has invalid action identity")
        score = "value" if metric == "analytic_v_reach" else "bounded_dls_suffix_goal_cost"
        gain = float(rows[greedy][score]) - float(rows[selected][score])
        if not math.isfinite(gain) or gain <= 0 or not math.isclose(
            gain, float(receipt["future_reference_score_improvement"]), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("model witness has no bound strong-reference gain")
    elif selected is not None:
        raise ValueError("abstention must not carry a selected reconfiguration")
    return {"policy": POLICY, "route": "reconfiguration" if witness else "same_bank_goal_progress",
        "selected_program_index": selected if witness else greedy, "same_bank_greedy_index": greedy,
        "reconfiguration_witness_found": witness, "physical_labels_used": False}


def compile_program_dispatch(model, pixels, goal, history, *, proposal_bank, **kwargs):
    prefix, receipt = compiler._compile_reconfiguration(
        model, pixels, goal, history, proposal_bank=proposal_bank, **kwargs)
    decision = dispatch_decision(receipt)
    index = decision["selected_program_index"]
    chosen = proposal_bank[index:index + 1].to(device=history.device, dtype=history.dtype).detach()
    if prefix is not None and not torch.equal(prefix, chosen):
        raise RuntimeError("dispatcher changed the compiler action witness")
    return chosen, {"program_type": "dynamics_native_program_dispatch", "dispatch": decision,
        "reconfiguration_compilation": receipt,
        **{key: receipt[key] for key in (
            "forward_rollout_evaluations", "model_predict_sample_evaluations", "backward_vjp_rows")}}
