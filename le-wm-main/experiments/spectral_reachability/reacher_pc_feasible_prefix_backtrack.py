"""Feasible interpolation of nonlinear-refined spectral prefix proposals."""
from __future__ import annotations

import math
import hashlib
from pathlib import Path

import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
from experiments.spectral_reachability import reacher_pc_prefix_refinement as prior
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0


ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0)


def choose(initial, refined, evaluate, threshold, *, alphas=ALPHAS):
    """Select a feasible interpolation without using physical outcomes."""
    if (initial.ndim != 1 or initial.shape != refined.shape
            or not torch.isfinite(initial).all() or not torch.isfinite(refined).all()
            or not math.isfinite(float(threshold)) or not alphas
            or any(not 0 <= a <= 1 for a in alphas)):
        raise ValueError("invalid interpolation contract")
    trials = []
    for alpha in alphas:
        point = initial + alpha * (refined - initial)
        immediate, future = evaluate(point)
        if (immediate.ndim != 0 or future.ndim != 1 or future.numel() == 0
                or not torch.isfinite(immediate) or not torch.isfinite(future).all()):
            raise ValueError("invalid predicted costs")
        score = float((-0.01 * torch.logsumexp(-future / 0.01, dim=0)).detach())
        trials.append({"alpha": float(alpha), "immediate": float(immediate),
                       "future_costs": future.detach().cpu().tolist(), "score": score,
                       "feasible": bool(float(immediate) >= float(threshold))})
    eligible = [i for i, trial in enumerate(trials) if trial["feasible"]]
    if not eligible:
        return None, {"trials": trials, "chosen_alpha": None}
    winner = min(eligible, key=lambda index: trials[index]["score"])
    alpha = alphas[winner]
    return (initial + alpha * (refined - initial)).detach(), {
        "trials": trials, "chosen_alpha": float(alpha)}


def augment(model, pixels, goal, history, bank, *, lower, upper, maximum=4):
    refined_bank, prior_receipt = prior.augment(
        model, pixels, goal, history, bank, lower=lower, upper=upper, maximum=maximum)
    goal_coordinate = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()
    probes = bank.new_tensor([[0., 0.], [0.3, 0.], [-0.3, 0.],
                              [0., 0.3], [0., -0.3]]).repeat(1, 5).unsqueeze(1)

    def evaluate(point):
        embeddings, actions = p1._state_after_prefix(
            model, pixels, history, point.reshape(1, 1, 10), action_lower=lower, action_upper=upper)
        immediate = p0._planning_coordinate(model, embeddings[:, -1], role="predicted")
        terminal = p1._terminal_from_state(
            model, embeddings.expand(5, -1, -1), actions.expand(5, -1, -1), probes,
            action_lower=lower, action_upper=upper)
        future = p0._planning_coordinate(model, terminal, role="predicted")
        return ((immediate - goal_coordinate).square().mean(),
                (future - goal_coordinate).square().mean(dim=1))

    threshold = prior_receipt["neutral_immediate_cost"] + 1e-6
    retained, trials = [], []
    with torch.no_grad():
        for index, record in enumerate(prior_receipt["refinements"]):
            source = record["source_index"]
            candidate, receipt = choose(bank[source, 0], refined_bank[len(bank) + index, 0],
                                        evaluate, threshold)
            receipt["source_index"] = source
            receipt["appended"] = candidate is not None
            trials.append(receipt)
            if candidate is not None:
                retained.append(candidate.reshape(1, 10))
    combined = torch.cat([bank.detach(), torch.stack(retained)], dim=0) if retained else bank.detach()
    return combined, {
        "prior_refinement": prior_receipt, "backtracking": trials, "threshold": threshold,
        "prior_refinement_sha256": hashlib.sha256(Path(prior.__file__).read_bytes()).hexdigest(),
        "backtrack_forward_prefix_samples": len(trials) * len(ALPHAS),
        "backtrack_forward_suffix_samples": len(trials) * len(ALPHAS) * len(probes),
        "scope": "model predictor samples; excludes encoder and wall time",
    }
