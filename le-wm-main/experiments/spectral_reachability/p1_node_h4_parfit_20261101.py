"""Independent H4 Q-planning node for an already-authorized metered leaf.

No environment, model loader, task loader, CUDA launch, or witness input.
The caller is responsible for the physical/RGB bindings and durable contract.
"""
import math

import numpy as np
import torch

from . import p1_h4_parfit_20261101 as h4

SEEDS = {"native": 20261030, "random": 20261031, "immediate_progress": 20261101}


def feedback_mix(bank, boundary):
    """H4 counterpart of the fixed four-response/four-Gaussian feedback bank."""
    value = np.asarray(bank)
    if (value.shape != (8, 4, 5, 2) or value.dtype != np.float32
            or not np.isfinite(value).all() or np.any(np.abs(value) > 1)
            or type(boundary) is not int or boundary not in (0, 5, 10, 15)):
        raise ValueError("invalid feedback initializer")
    if boundary == 0:
        return value.copy()
    generator = np.random.default_rng(20261031 + boundary)
    random = np.clip(.4 * generator.standard_normal((4, 4, 5, 2)), -1, 1).astype(np.float32)
    return np.concatenate((value[[3, 5, 6, 7]], random))


def run_node(leaf, *, arm, boundary, states, history, goal, mean, std, matrix,
             previous=None, issued=None, emit):
    """Run the fixed-cost node through the leaf's real dispatch API.

    ``previous`` and ``issued`` are accepted only after the external physical
    driver verifies episode, returned commands, current RGB and action history.
    All calls are checked against the already-open durable driver's counters.
    """
    horizon, carry = h4.dimensions(boundary)
    expected = h4.node_cost(arm, boundary)
    driver = leaf.driver
    if (not callable(emit) or set(driver.charged) != set(h4.AXES)
            or set(driver.caps) != set(h4.AXES) or driver.failed or driver.closed
            or any(driver.charged[k] + expected[k] > driver.caps[k] for k in h4.AXES)):
        raise ValueError("complete node reservation unavailable")
    before = dict(driver.charged)
    if (states.shape != (1, 3, 192) or history.shape != (1, 2, 10)
            or goal.shape != (1, 192) or mean.shape != (2,) or std.shape != (2,)
            or matrix.shape != (192, 192)
            or any(not isinstance(x, torch.Tensor) or not x.is_floating_point()
                   or x.device != states.device or not bool(torch.isfinite(x).all())
                   for x in (states, history, goal, mean, std, matrix))
            or any(x.dtype != states.dtype for x in (history, goal, mean, std))
            or not bool((std > 0).all())
            or (not carry and (previous is not None or issued is not None))
            or (carry and (previous is None or issued is None))):
        raise ValueError("node input contract")
    if carry and (not isinstance(previous, torch.Tensor) or not isinstance(issued, torch.Tensor)
                  or previous.shape != (horizon + 1, 5, 2) or issued.shape != (5, 2)
                  or previous.device != states.device or issued.device != states.device
                  or previous.dtype != states.dtype or issued.dtype != states.dtype
                  or not bool(torch.isfinite(previous).all())
                  or not bool((previous.abs() <= 1).all())
                  or not torch.equal(previous[0], issued)):
        raise ValueError("previous plan or returned packet mismatch")

    def rollout(raw, phase):
        z, past = states.detach(), history.detach()
        for t in range(raw.shape[1]):
            packet = ((raw[:, t] - mean) / std).reshape(1, 1, 10)
            context = torch.cat((past, packet), dim=1)
            z_next = leaf.predict(z, leaf.action_encode(context, phase=phase), phase=phase)
            if z_next.shape != z.shape or not bool(torch.isfinite(z_next).all()):
                raise ValueError("invalid prediction")
            z = torch.cat((z, z_next[:, -1:]), dim=1)[:, -3:]
            past = context[:, -2:]
        return z[:, -1]

    def objective(pred):
        residual = (pred - goal).double()
        value = (residual @ matrix.double().T).square().mean()
        if not bool(torch.isfinite(value)):
            raise ValueError("nonfinite objective")
        return value

    def score(plan, phase="verification"):
        with torch.no_grad():
            leaf._dispatch(phase, "replay", {"replay_calls": 1}, lambda: None)
            prediction = rollout(plan[None], phase)
            return leaf._dispatch(
                phase, "selection", {"selection_evaluations": 1},
                lambda: float(objective(prediction).item()))

    try:
        if arm == "structured":
            leaf._dispatch("jacobian", "jacobian", {"jacobian_calls": 1}, lambda: None)
            nominal = torch.zeros((1, 4, 5, 2), device=states.device,
                                  dtype=states.dtype, requires_grad=True)
            prediction = rollout(nominal[:, :horizon], "jacobian")
            rows = []
            for i in range(192):
                grad = leaf._dispatch(
                    "jacobian", "backward", {"backward_calls": 1, "vjp_rows": 1},
                    lambda i=i: torch.autograd.grad(prediction[0, i], nominal,
                                                    retain_graph=i < 191)[0])
                rows.append(grad.detach().reshape(-1))
            jac = torch.stack(rows).cpu().numpy()
            bank, singular = leaf._dispatch(
                "initialization", "svd", {"svd_calls": 1},
                lambda: h4.response_bank(jac, matrix.detach().cpu().numpy(),
                    goal.detach().cpu().numpy().reshape(192),
                    prediction.detach().cpu().numpy().reshape(192), horizon))
            bank = feedback_mix(bank, boundary)
            initial = torch.from_numpy(bank).to(device=states.device, dtype=states.dtype)
            emit("spectral_signal", {"jacobian": jac.tolist(),
                                     "nominal": prediction.detach().cpu().tolist(),
                                     "singular_values": singular.tolist()})
        else:
            generator = torch.Generator(device="cpu").manual_seed(SEEDS[arm])
            count = 16 if arm == "immediate_progress" else 8
            initial = (.4 * torch.randn((count, 4, 5, 2), generator=generator,
                                        dtype=states.dtype)).clamp(-1, 1).to(states.device)
            if arm == "native":
                initial[0] = 0
            emit("generated_candidates", {"raw": initial.cpu().tolist(), "seed": SEEDS[arm]})
            if arm == "immediate_progress":
                screening = [score(plan[:1], "screening") for plan in initial]
                order = sorted(range(16), key=lambda i: (screening[i], i))[:8]
                emit("screen_scores", {"scores": screening, "indices": order})
                initial = initial[order]
        emit("initial_candidates", {"raw": initial.detach().cpu().tolist()})
        optimized = []
        for initial_plan in initial:
            value = initial_plan[None].detach().clone()
            for _ in range(6):
                raw = value.detach().requires_grad_(True)
                prediction = rollout(raw[:, :horizon], "common_suffix")
                gradient = leaf.gradient(objective(prediction), raw, phase="common_suffix")
                value = h4.normalized_update(raw, gradient, horizon, boundary)
            optimized.append(value[0])
        bank = torch.stack(optimized)
        emit("optimized_candidates_before_exchange", {"raw": bank.cpu().tolist()})
        initial_scores = [score(plan[:horizon]) for plan in bank]
        if horizon > 1:
            bank, exchange_scores, tails = h4.exchange(bank, horizon, score)
            emit("suffix_exchange", {"scores": exchange_scores, "tail_indices": tails})
            fresh_scores = [exchange_scores[8*i+j] for i, j in enumerate(tails)]
        else:
            fresh_scores = initial_scores
        if carry:
            selected, scores, index = h4.feedback_select(bank, previous, issued, boundary, score)
        else:
            scores = fresh_scores
            index = min(range(8), key=lambda i: (scores[i], i))
            selected = bank[index, :horizon].clone()
        actual = {key: driver.charged[key] - before[key] for key in h4.AXES}
        if actual != expected or not all(math.isfinite(x) for x in scores):
            raise ValueError("actual node charges do not match reservation")
        emit("selected_full_plan", {"raw": selected.detach().cpu().tolist(),
                                    "index": index, "scores": scores,
                                    "charges": actual, "horizon": horizon})
        return {"selected_raw": selected.detach(), "selected_index": index,
                "scores": scores, "optimized_raw": bank.detach(),
                "charges": actual, "horizon": horizon}
    except BaseException:
        leaf._abort()
        raise
