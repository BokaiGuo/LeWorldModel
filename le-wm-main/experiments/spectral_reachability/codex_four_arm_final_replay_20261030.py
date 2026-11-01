"""Fixed-roster raw-action fresh model replay and deterministic selection.

Only declared leaf calls are metered. No execution or fairness authorization.
"""
import math

import torch

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as suffix

POLICY_SCHEMA = "codex_four_arm_final_replay_20261030.policy.v1"


def replay_and_select(leaf, states, history_actions, candidates, goal, mean, std, policy, contract):
    """Replay every supplied final raw action sequence, score all, then select.

    candidates is [N,T,5,2], with one shared [1,H,D] history and [1,D] goal.
    History actions are already normalized. Final actions must be supplied by the
    caller's separately metered optimizer, not assumed to have been evaluated.
    Lower terminal-latent mean-square error wins; exact ties use declared order.
    """
    spec.require(type(leaf) is suffix.LeafAdapter, "LeafAdapter required")
    try:
        spec.require(spec.digest(contract) == leaf.contract_sha, "contract binding")
        spec.require(type(policy) is dict and set(policy) ==
                     {"schema", "candidate_ids", "horizon", "history_size", "metric", "tie", "replay_batch",
                      "action_chart", "mean", "std"},
                     "policy fields")
        spec.require(spec.digest(policy) == contract["phase_policy_sha256"], "phase policy hash mismatch")
        ids = policy["candidate_ids"]
        spec.require(policy["schema"] == POLICY_SCHEMA and type(ids) is list and ids
                     and all(type(x) is str and x and x.strip() == x for x in ids)
                     and len(set(ids)) == len(ids), "fixed ordered candidate IDs")
        spec.require(type(policy["horizon"]) is int and policy["horizon"] > 0
                     and type(policy["history_size"]) is int and policy["history_size"] > 0
                     and type(policy["replay_batch"]) is int and policy["replay_batch"] == 1
                     and policy["metric"] == "terminal_latent_mean_squared_error"
                     and policy["tie"] == "first_declared_candidate"
                     and policy["action_chart"] == "raw_packet_5x2_box_minus1_to1", "replay policy")
        for key in ("mean", "std"):
            spec.require(type(policy[key]) is list and len(policy[key]) == 2
                         and all(type(v) in (int, float) and math.isfinite(v) for v in policy[key]),
                         "normalizer policy")
        spec.require(all(v > 0 for v in policy["std"]), "positive normalizer std")
        for tensor in (states, history_actions, candidates, goal, mean, std):
            leaf._tensor(tensor)
            spec.require(tensor.is_floating_point() and suffix._finite(tensor), "finite floating inputs")
        spec.require(states.ndim == 3 and states.shape[0] == 1 and states.shape[1] == policy["history_size"]
                     and history_actions.ndim == 3 and history_actions.shape[:2] == (1, states.shape[1] - 1)
                     and candidates.ndim == 4 and candidates.shape[0] == len(ids)
                     and candidates.shape[1] == policy["horizon"] and candidates.shape[2:] == (5, 2)
                     and history_actions.shape[2] == 10 and mean.shape == std.shape == (2,)
                     and goal.ndim == 2 and goal.shape == (1, states.shape[2]),
                     "fixed shared context and candidate denominator")
        spec.require(all(t.device == states.device and t.dtype == states.dtype
                         for t in (history_actions, candidates, goal, mean, std)), "device/dtype mismatch")
        spec.require(bool((std > 0).all().item()) and
                     torch.equal(mean, torch.tensor(policy["mean"], device=mean.device, dtype=mean.dtype)) and
                     torch.equal(std, torch.tensor(policy["std"], device=std.device, dtype=std.dtype)),
                     "normalizer mismatch")
        spec.require(bool((candidates.abs() <= 1).all().item()), "raw action outside physical box")
        updated = candidates.detach().clone()
        scores = []
        with torch.no_grad():
            for index in range(len(ids)):
                actions = updated[index:index+1]
                # This is a charged replay reservation/entry marker, not proof
                # of replay completion; leaf events and final result are separate.
                leaf._dispatch("verification", "replay", {"replay_calls": 1}, lambda: None)
                z, past = states.detach(), history_actions.detach()
                h = policy["history_size"]
                for step in range(policy["horizon"]):
                    normalized = ((actions[:, step] - mean) / std).reshape(1, 1, 10)
                    context = torch.cat((past, normalized), dim=1)
                    encoded = leaf.action_encode(context, phase="verification")
                    prediction = leaf.predict(z, encoded, phase="verification")
                    spec.require(isinstance(prediction, torch.Tensor) and prediction.shape == z.shape
                                 and suffix._finite(prediction), "replay prediction shape/finite")
                    z = torch.cat((z, prediction[:, -1:]), dim=1)[:, -h:]
                    past = context[:, -(h - 1):] if h > 1 else context[:, :0]

                def score():
                    value = (z[:, -1] - goal.detach()).square().mean()
                    spec.require(suffix._finite(value), "nonfinite replay score")
                    result = float(value.item())
                    spec.require(math.isfinite(result), "nonfinite replay score")
                    return result

                scores.append(leaf._dispatch("verification", "selection",
                                             {"selection_evaluations": 1}, score))
        winner = min(range(len(ids)), key=lambda i: (scores[i], i))
        return {"candidate_ids": list(ids), "scores": scores, "selected_index": winner,
                "selected_id": ids[winner], "actions": updated,
                "scope": "model-only post-update replay; no physical outcome",
                "fairness_proven": False, "execution_authorized": False}
    except BaseException:
        leaf._abort()
        raise
