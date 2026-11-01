"""Model-only action-word dispatch under the deployed numerical suffix contract.

This is a separately named intervention, not a mutation of historical words.
It retains every original fine8 score and adds the same cached-state optimizer
to all seventeen words, carrying the exact selected bounded command plan.
"""
from __future__ import annotations

import torch

from experiments.spectral_reachability import reacher_d2p_action_words as words
from experiments.spectral_reachability import reacher_d2p_state_suffix as native


POLICY = "all17_state_native_two_chart_cem_exact_command_words_v1"


def compile_words(model, pixels, goal, history, *, lower, upper, seed, progress=None):
    p0, p1, solver, rescue = words.p0, words.p1, words.solver, words.rescue
    emb, target = p0._latent_context(model, pixels, history), p0._goal_latent(model, goal)

    def terminal(flat):
        value = p1._terminal_from_state(model, emb, history, flat.reshape(1, 1, history.shape[-1]),
            action_lower=lower, action_upper=upper)[0]
        return p0._planning_coordinate(model, value, role="predicted")

    origin = torch.zeros(history.shape[-1], device=history.device, dtype=history.dtype, requires_grad=True)
    response = torch.autograd.functional.jacobian(terminal, origin, vectorize=True)
    bank, support = words.build_words(response.detach())
    samples, vjp = 1, response.shape[0]
    score_rows, original_rows, word_records, tails = [], [], [], {}
    for index in range(len(bank)):
        prefix = bank[index:index + 1]
        with torch.no_grad():
            immediate = float(p0._cost_fn(model, pixels, goal, history, prefix, action_lower=lower, action_upper=upper)[0])
        next_emb, next_history = p1._state_after_prefix(model, pixels, history, prefix, action_lower=lower, action_upper=upper)
        reference, original = solver.compile_snapshots(model, target, next_emb, next_history, future_horizon=2,
            action_lower=lower, action_upper=upper, damping=.01, checkpoints=(8,), accept_update=rescue.accept_with_fine_rescue)[8]
        variants, receipt = native.compile_state_candidates(model, target, next_emb, next_history, reference_params=reference,
            action_lower=lower, action_upper=upper, seed=seed + 70000)
        chosen, original_cost = variants[receipt["selected"]], original["trace"][-1]["cost_after"]
        with torch.no_grad():
            complete = native.box.terminal_commands(model, emb, history, torch.cat((prefix.tanh(), chosen), dim=1),
                action_lower=lower, action_upper=upper)
            split = native.box.terminal_commands(model, next_emb, next_history, chosen, action_lower=lower, action_upper=upper)
            partition = float((complete - split).abs().max())
        if partition > 2e-6 or receipt["model_costs"][receipt["selected"]] > original_cost:
            raise RuntimeError("native imagined suffix changed dynamics or lost its incumbent")
        original_rows.append({"proposal_index": index, "immediate_goal_cost": immediate, "refined_goal_cost": original_cost,
            "suffix_receipt": original})
        score_rows.append({"proposal_index": index, "immediate_goal_cost": immediate,
            "refined_goal_cost": receipt["model_costs"][receipt["selected"]]})
        word_records.append({"index": index, "original_fine8_cost": original_cost, "native_receipt": receipt,
            "planned_commands": chosen.tolist(), "partition_max_abs_error": partition})
        tails[index] = chosen
        samples += 4 + 4 + original["model_predict_sample_evaluations"] + receipt["model_predict_sample_evaluations"] + 8
        vjp += original["backward_vjp_rows"] + receipt["backward_vjp_rows"]
        if progress is not None:
            progress(index + 1, len(bank))
    old = words.refinement.select_refined_bank(original_rows, non_greedy_margin=1e-6)
    decision = words.refinement.select_refined_bank(score_rows, non_greedy_margin=1e-6)
    return bank, tails, {"policy": POLICY, "historical_decision": old, "native_decision": decision,
        "score_rows": score_rows, "word_records": word_records, "original_rows": original_rows, "word_support": support,
        "model_predict_sample_evaluations": samples, "backward_vjp_rows": int(vjp), "physical_labels_used": False,
        "trained_modules": False, "partition_max_abs_error": max(w["partition_max_abs_error"] for w in word_records),
        "scope": "Numerical score/execution alignment; no P1 admission or physical ordering guarantee."}
