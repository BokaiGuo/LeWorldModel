"""Known-root goal-conditioned damped spectral initializer with common suffix.

This module is not a physical launcher. Calls are cooperatively metered and
do not certify full cost, fairness, or independent confirmation.
"""
import json

import numpy as np
import torch

from . import codex_four_arm_common_suffix_control_20261030 as base
from . import codex_four_arm_full_cost_20261030 as spec

EXTRA_ARMS = ("native", "random")
EXTRA_ROUNDS = 16
DAMPING_RELATIVE = 0.01


def goal_bank(jac, projection, nominal, goal):
    """Use the damped task-residual solution in the same bounded raw chart."""
    jac = np.asarray(jac, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    nominal = np.asarray(nominal, dtype=np.float64).reshape(-1)
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    spec.require(jac.shape == (192, 30) and projection.shape == (192, 192)
                 and nominal.shape == goal.shape == (192,)
                 and all(np.isfinite(x).all() for x in (jac, projection, nominal, goal)),
                 "goal-conditioned spectral inputs")
    weighted_jac = projection @ jac
    residual = projection @ (goal - nominal)
    left, singular, right = np.linalg.svd(weighted_jac, full_matrices=False)
    spec.require(np.isfinite(singular).all() and singular[0] > 0, "degenerate Jacobian")
    damping = DAMPING_RELATIVE * singular[0] ** 2
    delta = right.T @ ((singular / (singular ** 2 + damping)) * (left.T @ residual))
    spec.require(np.isfinite(delta).all(), "invalid damped solution")
    direction = delta / max(1.0, np.max(np.abs(delta)))
    bank = [np.zeros(30)] + [scale * direction for scale in (.25, .5, 1., -.25, -.5, -1.)]
    bank.append(np.clip(delta, -1., 1.))
    return np.stack(bank).astype(np.float32).reshape(8, 3, 5, 2), singular


def cost(arm):
    result = base.cost(arm)
    # Each extra optimization uses two iterations, three model packets and
    # one VJP per iteration.  Selection and replay are already in base.cost.
    n = EXTRA_ROUNDS * int(arm in EXTRA_ARMS)
    for key in ("action_encode_calls", "predict_calls", "predict_samples"):
        result[key] += n * base.CONFIG["steps"] * 3
    for key in ("backward_calls", "vjp_rows"):
        result[key] += n * base.CONFIG["steps"]
    for key in ("action_encode_calls", "predict_calls", "predict_samples"):
        result[key] += int(arm in EXTRA_ARMS) * 3
    result["replay_calls"] += int(arm in EXTRA_ARMS)
    result["selection_evaluations"] += int(arm in EXTRA_ARMS)
    return result


def contract_for(task, model_sha, hardware_sha):
    policy, contract, config = base.contract_for(task, model_sha, hardware_sha)
    initializer = base.spectral.file_sha(__file__)
    for arm in spec.ARMS:
        contract["arms"][arm]["initializer_sha256"] = initializer
    caps = {key: max(cost(arm)[key] for arm in spec.ARMS) for key in spec.AXES}
    contract["levels"] = [{"id": "node", "caps": caps},
                          {"id": "unused_larger", "caps": {k: v + 1 for k, v in caps.items()}}]
    spec.preflight(contract, spec.digest(contract))
    return policy, contract, config


def plan(leaf, arm, task, config, policy, contract, *, device, emit):
    spec.require(type(leaf) is base.leafmod.LeafAdapter, "leaf")
    try:
        spec.require(arm in spec.ARMS and callable(emit) and spec.digest(contract) == leaf.contract_sha,
                     "contract binding")
        expected_policy, expected_contract, expected_config = contract_for(
            task, contract["identity"]["model"], contract["identity"]["hardware"])
        spec.require((policy, contract, config) == (expected_policy, expected_contract, expected_config)
                     and leaf.config_sha == spec.digest(config), "task/config binding")
        with leaf.driver.path.open("rb") as stream:
            header = json.loads(stream.readline(1024 * 1024))["payload"]
        spec.require(header["arm"] == arm and header["level"] == "node", "arm/level binding")
        expected = cost(arm)
        spec.require(all(leaf.driver.charged[k] + expected[k] <= leaf.driver.caps[k] for k in spec.AXES),
                     "full node reserve unavailable")
        tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        states = tensor(task["root_context"]["latent_history"])[None]
        history = tensor(task["root_context"]["action_history"])[None]
        goal = tensor(task["goal"])[None]
        mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
        projection = tensor(task["projection"])
        for value in (states, history, goal, mean, std, projection):
            leaf._tensor(value)
            spec.require(base.leafmod._finite(value), "nonfinite input")
        spec.require(states.shape == (1, 3, 192) and history.shape == (1, 2, 10)
                     and goal.shape == (1, 192) and mean.shape == std.shape == (2,)
                     and bool((std > 0).all().item()) and projection.shape == (192, 192)
                     and torch.equal(projection, torch.eye(192, device=states.device)), "feedback ABI")
        emit("node_header", {"arm": arm, "task_sha256": spec.digest(task),
                             "label": ("goal_conditioned_damped_spectral" if arm == "structured"
                                       else base.LABELS[arm]), "config": config,
                             "extra_policy": "model_winner" if arm in EXTRA_ARMS else "none"})
        if arm == "structured":
            leaf._dispatch("jacobian", "jacobian", {"jacobian_calls": 1}, lambda: None)
            nominal = torch.zeros((1, 3, 5, 2), device=states.device, requires_grad=True)
            pred = base.raw._normal(leaf, "jacobian", states.detach(), history.detach(), nominal, mean, std)
            rows = []
            for i in range(192):
                grad = leaf._dispatch("jacobian", "backward", {"backward_calls": 1, "vjp_rows": 1},
                    lambda i=i: torch.autograd.grad(pred[0, i], nominal, retain_graph=i < 191)[0])
                spec.require(grad.shape == nominal.shape and base.leafmod._finite(grad), "invalid Jacobian")
                rows.append(grad.detach().reshape(-1))
            jac = torch.stack(rows).cpu().numpy()
            emit("spectral_signal", {"nominal": pred.detach().cpu().tolist(), "jacobian": jac.tolist(),
                                     "projection": projection.cpu().tolist()})
            bank, singular = leaf._dispatch("initialization", "svd", {"svd_calls": 1},
                lambda: goal_bank(jac, projection.cpu().numpy(), pred.detach().cpu().numpy(),
                                  goal.detach().cpu().numpy()))
            initial = torch.from_numpy(bank).to(device=states.device, dtype=states.dtype)
            emit("spectral_values", {"singular_values": singular.tolist()})
        else:
            generator = torch.Generator(device="cpu").manual_seed(base.SEEDS[arm])
            n = 16 if arm == "immediate_progress" else 8
            candidates = (.4 * torch.randn((n, 3, 5, 2), generator=generator)).clamp(-1, 1).to(states.device)
            if arm == "native":
                candidates[0] = 0
            emit("generated_candidates", {"raw": candidates.cpu().tolist(), "seed": base.SEEDS[arm]})
            if arm == "immediate_progress":
                scores = []
                with torch.no_grad():
                    for i in range(16):
                        leaf._dispatch("screening", "replay", {"replay_calls": 1}, lambda: None)
                        prediction = base.raw._normal(leaf, "screening", states, history,
                                                      candidates[i:i+1, :1], mean, std)
                        def score():
                            value = float((prediction - goal).square().mean().item())
                            spec.require(base.math.isfinite(value), "nonfinite screen")
                            return value
                        scores.append(leaf._dispatch("screening", "selection",
                                                     {"selection_evaluations": 1}, score))
                        emit("screen_score", {"index": i, "score": scores[-1]})
                order = sorted(range(16), key=lambda i: (scores[i], i))[:8]
                emit("shortlist", {"indices": order})
                initial = candidates[order]
            else:
                initial = candidates
        emit("initial_candidates", {"raw": initial.detach().cpu().tolist()})
        parts = [base.suffix.optimize(leaf, states, history, initial[i:i+1], goal, mean, std, config)
                 for i in range(8)]
        optimized = torch.cat(parts)
        emit("optimized_candidates", {"raw": optimized.detach().cpu().tolist()})
        final = base.replay.replay_and_select(leaf, states, history, optimized, goal, mean, std, policy, contract)
        if arm in EXTRA_ARMS:
            index = final["selected_index"]
            before = final["scores"][index]
            updated = optimized[index:index+1]
            for _ in range(EXTRA_ROUNDS):
                updated = base.suffix.optimize(leaf, states, history, updated,
                                               goal, mean, std, config)
            with torch.no_grad():
                leaf._dispatch("verification", "replay", {"replay_calls": 1}, lambda: None)
                predicted = base.raw._normal(leaf, "verification", states, history, updated, mean, std)
                def score():
                    value = float((predicted - goal).square().mean().item())
                    spec.require(base.math.isfinite(value), "nonfinite updated score")
                    return value
                after = leaf._dispatch("verification", "selection", {"selection_evaluations": 1}, score)
            accepted = after < before
            emit("winner_refinement", {"index": index, "extra_rounds": EXTRA_ROUNDS,
                                      "before_score": before,
                                      "after_score": after, "accepted": accepted,
                                      "before_raw": optimized[index].detach().cpu().tolist(),
                                      "after_raw": updated[0].detach().cpu().tolist()})
            if accepted:
                optimized = optimized.clone()
                optimized[index] = updated[0]
                final["scores"][index] = after
        emit("final_selection", {"scores": final["scores"], "index": final["selected_index"],
                                 "id": final["selected_id"]})
        return {"initial_raw": initial.detach(), "optimized_raw": optimized.detach(),
                "selected_raw": optimized[final["selected_index"]].detach(),
                "scores": final["scores"], "selected_index": final["selected_index"],
                "label": base.LABELS[arm], "fairness_proven": False,
                "physical_execution_authorized": False}
    except BaseException:
        leaf._abort()
        raise
