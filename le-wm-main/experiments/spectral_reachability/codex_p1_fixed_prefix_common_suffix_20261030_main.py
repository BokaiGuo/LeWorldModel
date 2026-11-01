"""Model-only fixed-prefix comparison with independently optimized common suffixes.

This is an exploratory planning kernel, not a physical capability intervention,
strong baseline, complete accounting, or P1 admission.
"""
import math
import torch

from . import codex_four_arm_full_cost_20261030 as spec
from . import codex_four_arm_leaf_suffix_20261030 as leafmod
from . import codex_four_arm_raw_suffix_20261030_main as suffix
from . import codex_four_arm_final_replay_20261030 as replay


def required_cost(count, steps):
    spec.require(type(count) is int and count > 0 and type(steps) is int and steps > 0,
                 "positive candidate and iteration counts")
    out = dict.fromkeys(spec.AXES, 0)
    calls = count * (1 + 2 * steps + 3)
    out.update(action_encode_calls=calls, predict_calls=calls, predict_samples=calls,
               backward_calls=count * steps, vjp_rows=count * steps,
               replay_calls=count, selection_evaluations=count)
    return out


def plan(leaf, states, history, prefixes, suffix_initial, goal, mean, std,
         config, policy, contract, *, emit):
    """Evaluate all declared prefixes with identical two-packet suffix optimization.

    Each predicted prefix successor gets its own action/history-aligned suffix
    optimization, starting from the same fixed initialization. Complete plans
    are freshly replayed from the original context before selection. Physical
    feedback must trigger a separate new invocation after actual execution.
    """
    spec.require(type(leaf) is leafmod.LeafAdapter, "leaf adapter required")
    try:
        spec.require(callable(emit) and spec.digest(contract) == leaf.contract_sha and
                     spec.digest(config) == leaf.config_sha, "contract/config binding")
        spec.require(type(config) is dict and set(config) == {"steps", "learning_rate", "history_size"}
                     and type(config["steps"]) is int and config["steps"] > 0
                     and type(config["history_size"]) is int and config["history_size"] >= 2
                     and type(config["learning_rate"]) in (int, float)
                     and math.isfinite(config["learning_rate"]) and config["learning_rate"] > 0,
                     "optimizer configuration")
        for tensor in (states, history, prefixes, suffix_initial, goal, mean, std):
            leaf._tensor(tensor)
            spec.require(tensor.is_floating_point() and leafmod._finite(tensor) and
                         tensor.device == states.device and tensor.dtype == states.dtype,
                         "finite aligned tensors")
        n = prefixes.shape[0] if prefixes.ndim == 4 else 0
        h = config["history_size"]
        spec.require(states.ndim == 3 and states.shape[:2] == (1, h) and h >= 2
                     and history.shape == (1, h - 1, 10) and goal.shape == (1, states.shape[2])
                     and prefixes.shape == (n, 1, 5, 2) and n > 0
                     and suffix_initial.shape == (1, 2, 5, 2)
                     and mean.shape == std.shape == (2,) and bool((std > 0).all().item())
                     and bool((prefixes.abs() <= 1).all().item())
                     and bool((suffix_initial.abs() <= 1).all().item()), "input shapes or bounds")
        spec.require(type(policy) is dict and set(policy) ==
                     {"schema", "candidate_ids", "horizon", "history_size", "metric", "tie",
                      "replay_batch", "action_chart", "mean", "std"}
                     and policy["schema"] == replay.POLICY_SCHEMA and policy["horizon"] == 3
                     and policy["history_size"] == h and policy["replay_batch"] == 1
                     and policy["metric"] == "terminal_latent_mean_squared_error"
                     and policy["tie"] == "first_declared_candidate"
                     and policy["action_chart"] == "raw_packet_5x2_box_minus1_to1"
                     and type(policy["candidate_ids"]) is list and len(policy["candidate_ids"]) == n
                     and all(type(x) is str and x and x.strip() == x for x in policy["candidate_ids"])
                     and len(set(policy["candidate_ids"])) == n
                     and policy["mean"] == mean.detach().cpu().tolist()
                     and policy["std"] == std.detach().cpu().tolist()
                     and spec.digest(policy) == contract["phase_policy_sha256"], "replay roster")
        needed = required_cost(n, config["steps"])
        spec.require(all(leaf.driver.charged[k] + needed[k] <= leaf.driver.caps[k] for k in spec.AXES),
                     "full comparison reserve unavailable")
        emit("inputs", {"prefixes": prefixes.detach().cpu().tolist(),
                        "suffix_initial": suffix_initial.detach().cpu().tolist()})
        plans = []
        for i in range(n):
            prefix = prefixes[i:i + 1].detach()
            normalized = ((prefix[:, 0] - mean) / std).reshape(1, 1, 10)
            action_context = torch.cat((history.detach(), normalized), dim=1)
            with torch.no_grad():
                encoded = leaf.action_encode(action_context, phase="trial")
                prediction = leaf.predict(states.detach(), encoded, phase="trial")
                spec.require(prediction.shape == states.shape and leafmod._finite(prediction),
                             "prefix prediction")
                successor = torch.cat((states.detach(), prediction[:, -1:]), dim=1)[:, -h:]
                successor_history = action_context[:, -(h - 1):]
            optimized = suffix.optimize(leaf, successor, successor_history, suffix_initial,
                                        goal, mean, std, config)
            plans.append(torch.cat((prefix, optimized), dim=1))
            emit("optimized", {"index": i, "successor": successor.cpu().tolist(),
                               "history": successor_history.cpu().tolist(),
                               "suffix": optimized.cpu().tolist()})
        actions = torch.cat(plans, dim=0)
        final = replay.replay_and_select(leaf, states, history, actions, goal, mean, std,
                                         policy, contract)
        emit("selection", {"scores": final["scores"], "index": final["selected_index"],
                           "actions": actions.cpu().tolist()})
        return {"actions": actions.detach(), "scores": final["scores"],
                "selected_index": final["selected_index"], "fairness_proven": False,
                "P1_admission": False, "physical_execution_authorized": False}
    except BaseException:
        leaf._abort()
        raise
