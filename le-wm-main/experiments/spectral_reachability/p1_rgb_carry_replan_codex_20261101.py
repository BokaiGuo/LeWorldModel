"""Planner-only equal-horizon RGB feedback selection; no physical launcher.

The caller supplies the current RGB-encoded context and raw model-generated
plans.  Scores are recomputed through the existing metered leaf adapter.
"""
import math

import torch

from . import codex_four_arm_raw_candidates_20261030 as raw_model


def select(leaf, *, states, history, goal, mean, std, new_candidates,
           previous_selected=None, executed_packets=1):
    """Compare fresh plans and the unexecuted tail using current context.

    previous_selected is the plan selected before execution, not actions from
    a physical replay. The caller must bind states to the current RGB and
    history to commands actually issued. This function cannot verify origin.
    """
    tensors = (states, history, goal, mean, std, new_candidates)
    if not all(isinstance(x, torch.Tensor) and x.is_floating_point() for x in tensors):
        raise ValueError("floating tensors required")
    if not all(x.device == states.device and x.dtype == states.dtype and
               bool(torch.isfinite(x).all()) for x in tensors):
        raise ValueError("finite aligned tensors required")
    if (states.ndim != 3 or states.shape[0] != 1 or states.shape[1] < 1 or
        history.shape != (1, states.shape[1] - 1, 10) or
        goal.shape != (1, states.shape[2]) or mean.shape != (2,) or
        std.shape != (2,) or not bool((std > 0).all()) or
        new_candidates.ndim != 4 or new_candidates.shape[0] < 1 or
        new_candidates.shape[1] < 1 or new_candidates.shape[2:] != (5, 2) or
        not bool((new_candidates.abs() <= 1).all())):
        raise ValueError("invalid context or candidate shape/bounds")
    if type(executed_packets) is not int or executed_packets < 1:
        raise ValueError("positive executed packet count required")

    carry = None
    if previous_selected is not None:
        p = previous_selected
        if (not isinstance(p, torch.Tensor) or p.device != states.device or
            p.dtype != states.dtype or p.ndim != 3 or p.shape[1:] != (5, 2) or
            p.shape[0] <= executed_packets or not bool(torch.isfinite(p).all()) or
            not bool((p.abs() <= 1).all())):
            raise ValueError("invalid previous selected plan or exhausted tail")
        carry = p[executed_packets:].detach().clone()
    horizon = carry.shape[0] if carry is not None else new_candidates.shape[1]
    if new_candidates.shape[1] < horizon:
        raise ValueError("fresh candidates shorter than carried tail")
    # All candidates, including carry, have identical horizon and scoring path.
    plans = [x.detach().clone() for x in new_candidates[:, :horizon]]
    if carry is not None:
        plans.append(carry)
    scores = []
    with torch.no_grad():
        for plan in plans:
            leaf._dispatch("verification", "replay", {"replay_calls": 1}, lambda: None)
            prediction = raw_model._normal(leaf, "verification", states, history,
                                           plan.unsqueeze(0), mean, std)

            def score():
                value = float((prediction - goal).square().mean().item())
                if not math.isfinite(value):
                    raise ValueError("nonfinite model score")
                return value

            scores.append(leaf._dispatch("verification", "selection",
                                         {"selection_evaluations": 1}, score))
    # Deterministic ties retain the incumbent, otherwise first fresh candidate.
    index = min(range(len(plans)), key=lambda i: (scores[i], i != len(plans) - 1
                                                   if carry is not None else False, i))
    count = len(plans)
    return {"selected_raw": plans[index], "selected_index": index,
            "selected_source": "carry" if carry is not None and index == count - 1 else "new",
            "scores": scores, "horizon_packets": horizon,
            "incremental_cost": {"action_encode_calls": count * horizon,
                                 "predict_calls": count * horizon,
                                 "predict_samples": count * horizon,
                                 "replay_calls": count,
                                 "selection_evaluations": count,
                                 "backward_calls": 0},
            "physical_labels_used": False}
