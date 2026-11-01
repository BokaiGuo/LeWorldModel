"""Explicit shared quadratic objective; native latent dynamics stay unchanged."""
import math
import torch

from . import p1_four_arm_extra32_budget_20261031 as control
from . import p1_carry_binding_20261101 as binding

base, spec = control.base, control.spec
MATRIX = None
GOAL = None
ARTIFACT_SHA = None


def loss(prediction, goal):
    if MATRIX is None or GOAL is None:
        raise ValueError("unbound quadratic objective")
    if goal.detach().cpu().tolist() != [GOAL]:
        raise ValueError("metric bound to a different goal")
    matrix = torch.as_tensor(MATRIX, device=prediction.device, dtype=torch.float64)
    residual = (prediction - goal).double()
    value = (residual @ matrix.T).square().mean()
    if not bool(torch.isfinite(value)):
        raise ValueError("nonfinite Q loss")
    return value


def evaluate(leaf, states, history, plans, goal, mean, std, phase="verification"):
    scores = []
    with torch.no_grad():
        for plan in plans:
            leaf._dispatch(phase, "replay", {"replay_calls": 1}, lambda: None)
            prediction = base.raw._normal(leaf, phase, states, history, plan[None], mean, std)
            scores.append(leaf._dispatch(phase, "selection", {"selection_evaluations": 1},
                                         lambda: float(loss(prediction, goal).item())))
    return scores


def optimize(leaf, states, history, initial, goal, mean, std, config):
    spec.require(type(leaf) is base.leafmod.LeafAdapter and spec.digest(config) == leaf.config_sha,
                 "optimizer contract")
    spec.require(initial.shape == (1, 3, 5, 2) and states.shape == (1, 3, 192)
                 and history.shape == (1, 2, 10) and goal.shape == (1, 192)
                 and bool(torch.isfinite(initial).all()) and bool((initial.abs() <= 1).all())
                 and bool((std > 0).all()), "optimizer input")
    value = initial.detach().clone()
    for _ in range(config["steps"]):
        raw = value.detach().requires_grad_(True)
        z, past = states.detach(), history.detach()
        for step in range(3):
            normalized = ((raw[:, step] - mean) / std).reshape(1, 1, 10)
            context = torch.cat((past, normalized), dim=1)
            prediction = leaf.predict(z, leaf.action_encode(context, phase="common_suffix"),
                                      phase="common_suffix")
            spec.require(prediction.shape == z.shape and bool(torch.isfinite(prediction).all()),
                         "native prediction")
            z, past = torch.cat((z, prediction[:, -1:]), dim=1)[:, -3:], context[:, -2:]
        gradient = leaf.gradient(loss(z[:, -1], goal.detach()), raw, phase="common_suffix")
        spec.require(gradient.shape == raw.shape and bool(torch.isfinite(gradient).all()), "Q gradient")
        value = (raw - config["learning_rate"] * gradient).clamp(-1, 1).detach()
    return value


def select(leaf, *, states, history, goal, mean, std, new_candidates,
           previous_selected=None, executed_packets=1):
    spec.require(new_candidates.ndim == 4 and new_candidates.shape[2:] == (5, 2)
                 and bool(torch.isfinite(new_candidates).all())
                 and bool((new_candidates.abs() <= 1).all()), "fresh candidates")
    carry = None
    if previous_selected is not None:
        spec.require(previous_selected.ndim == 3 and previous_selected.shape[1:] == (5, 2)
                     and previous_selected.shape[0] > executed_packets
                     and bool(torch.isfinite(previous_selected).all())
                     and bool((previous_selected.abs() <= 1).all()), "carry source")
        carry = previous_selected[executed_packets:].detach().clone()
    horizon = carry.shape[0] if carry is not None else new_candidates.shape[1]
    spec.require(new_candidates.shape[1] >= horizon, "equal horizon")
    bank = new_candidates[:, :horizon].detach().clone()
    if carry is not None:
        bank = torch.cat((bank, carry[None]))
    scores = evaluate(leaf, states, history, bank, goal, mean, std)
    index = min(range(len(scores)), key=lambda i: (scores[i], i != len(scores)-1 if carry is not None else False, i))
    return dict(selected_raw=bank[index], selected_index=index,
                selected_source="carry" if carry is not None and index == len(scores)-1 else "new",
                scores=scores, horizon_packets=horizon, physical_labels_used=False,
                incremental_cost=binding.extra_cost(len(new_candidates), horizon, has_carry=carry is not None))


def plan(leaf, arm, task, config, policy, contract, *, device, emit):
    spec.require(type(leaf) is base.leafmod.LeafAdapter, "leaf")
    try:
        spec.require(arm in spec.ARMS and spec.digest(contract) == leaf.contract_sha, "contract")
        expected = control.contract_for(task, contract["identity"]["model"], contract["identity"]["hardware"])
        spec.require((policy, contract, config) == expected and leaf.config_sha == spec.digest(config),
                     "complete Q node config")
        spec.require(policy["metric"] == "goal_encoder_cometric_v1", "explicit Q policy")
        budget = control.cost(arm)
        spec.require(all(leaf.driver.charged[k] + budget[k] <= leaf.driver.caps[k] for k in spec.AXES),
                     "node reservation")
        tensor = lambda x: torch.tensor(x, device=device, dtype=torch.float32)
        states, history = tensor(task["root_context"]["latent_history"])[None], tensor(task["root_context"]["action_history"])[None]
        goal, mean, std = tensor(task["goal"])[None], tensor(task["action_mean"]), tensor(task["action_std"])
        spec.require(states.shape == (1, 3, 192) and history.shape == (1, 2, 10)
                     and task["goal"] == GOAL and bool((std > 0).all()), "native context/goal")
        emit("node_header", dict(arm=arm, task_sha256=spec.digest(task), config=config,
                                 label=base.LABELS[arm], metric_artifact_sha256=ARTIFACT_SHA))
        if arm == "structured":
            leaf._dispatch("jacobian", "jacobian", {"jacobian_calls": 1}, lambda: None)
            nominal = torch.zeros((1, 3, 5, 2), device=device, requires_grad=True)
            prediction = base.raw._normal(leaf, "jacobian", states, history, nominal, mean, std)
            gradients = []
            for i in range(192):
                grad = leaf._dispatch("jacobian", "backward", {"backward_calls": 1, "vjp_rows": 1},
                    lambda i=i: torch.autograd.grad(prediction[0, i], nominal, retain_graph=i < 191)[0])
                gradients.append(grad.detach().reshape(-1))
            jac = torch.stack(gradients).cpu().numpy()
            projection = tensor(task["projection"]).cpu().numpy()
            emit("spectral_signal", dict(nominal=prediction.detach().cpu().tolist(),
                                         jacobian=jac.tolist(), projection=projection.tolist()))
            bank, singular = leaf._dispatch("initialization", "svd", {"svd_calls": 1},
                lambda: base.spectral.spectral_bank(jac, projection))
            initial = torch.from_numpy(bank).to(device=device)
            emit("spectral_values", dict(singular_values=singular.tolist()))
        else:
            generator = torch.Generator(device="cpu").manual_seed(base.SEEDS[arm])
            count = 16 if arm == "immediate_progress" else 8
            bank = (.4 * torch.randn((count, 3, 5, 2), generator=generator)).clamp(-1, 1).to(device)
            if arm == "native":
                bank[0] = 0
            emit("generated_candidates", dict(raw=bank.cpu().tolist(), seed=base.SEEDS[arm]))
            if arm == "immediate_progress":
                scores = evaluate(leaf, states, history, bank[:, :1], goal, mean, std, phase="screening")
                for i, value in enumerate(scores):
                    emit("screen_score", dict(index=i, score=value))
                indices = sorted(range(16), key=lambda i: (scores[i], i))[:8]
                emit("shortlist", dict(indices=indices))
                initial = bank[indices]
            else:
                initial = bank
        emit("initial_candidates", dict(raw=initial.detach().cpu().tolist()))
        optimized = torch.cat([base.suffix.optimize(leaf, states, history, initial[i:i+1],
                                                   goal, mean, std, config) for i in range(8)])
        emit("optimized_candidates", dict(raw=optimized.detach().cpu().tolist()))
        scores = evaluate(leaf, states, history, optimized, goal, mean, std)
        index = min(range(8), key=lambda i: (scores[i], i))
        if arm in control.EXTRA_ARMS:
            updated = optimized[index:index+1]
            for _ in range(control.EXTRA_ROUNDS):
                updated = base.suffix.optimize(leaf, states, history, updated, goal, mean, std, config)
            after = evaluate(leaf, states, history, updated, goal, mean, std)[0]
            accepted = after < scores[index]
            emit("winner_refinement", dict(index=index, extra_rounds=control.EXTRA_ROUNDS,
                                           before_score=scores[index], after_score=after, accepted=accepted,
                                           before_raw=optimized[index].detach().cpu().tolist(),
                                           after_raw=updated[0].detach().cpu().tolist()))
            if accepted:
                optimized = optimized.clone()
                optimized[index], scores[index] = updated[0], after
        emit("final_selection", dict(scores=scores, index=index, id=policy["candidate_ids"][index]))
        return dict(initial_raw=initial.detach(), optimized_raw=optimized.detach(),
                    selected_raw=optimized[index].detach(), selected_index=index, scores=scores,
                    label=base.LABELS[arm], fairness_proven=False, physical_execution_authorized=False)
    except BaseException:
        leaf._abort()
        raise
