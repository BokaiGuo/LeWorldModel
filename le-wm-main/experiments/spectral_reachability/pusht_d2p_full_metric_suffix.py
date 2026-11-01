"""Full-coordinate bounded suffix refinement with an explicit zero incumbent."""
from __future__ import annotations

import torch


def refine(terminal, goal: torch.Tensor, initial: torch.Tensor, *, damping: float = .01,
           iterations: int = 4) -> tuple[torch.Tensor, dict]:
    """Minimize actual nonlinear SSE + damping * ||atanh(action)||^2.

    The frozen model is only an oracle for action optimization. Every accepted
    iterate is checked in the SAME full latent metric as the returned score.
    The zero program and supplied one-shot solution are both retained.
    """
    if (initial.ndim != 2 or initial.shape[-1] != 2 or goal.ndim != 1
            or not torch.isfinite(initial).all() or not torch.isfinite(goal).all()
            or damping <= 0 or iterations < 0):
        raise ValueError("finite two-coordinate action parameters and positive damping required")
    shape = initial.shape

    def evaluate(flat):
        prediction = terminal(flat.reshape(shape))
        if prediction.shape != goal.shape or not torch.isfinite(prediction).all():
            raise ValueError("terminal/goal full-coordinate contract violated")
        cost = (prediction-goal).square().sum()+damping*flat.square().sum()
        return prediction, cost

    start = initial.detach().reshape(-1).clone()
    with torch.no_grad():
        _, origin_cost = evaluate(torch.zeros_like(start))
        _, initial_cost = evaluate(start)
        best = start if initial_cost < origin_cost else torch.zeros_like(start)
        before = float(torch.minimum(origin_cost, initial_cost))
    history = []
    for step in range(iterations):
        current = best.detach().requires_grad_(True)
        prediction, cost = evaluate(current)
        jac = torch.autograd.functional.jacobian(lambda x: terminal(x.reshape(shape)), current, vectorize=True)
        if not torch.isfinite(jac).all():
            raise ValueError("nonfinite full-coordinate Jacobian")
        gram = jac.T@jac+damping*torch.eye(current.numel(), device=current.device, dtype=current.dtype)
        direction = torch.linalg.solve(gram, jac.T@(goal-prediction)-damping*current)
        candidate_costs = []
        chosen, chosen_cost = current.detach(), float(cost.detach())
        with torch.no_grad():
            for alpha in (1., .5, .25, .125):
                candidate = current.detach()+alpha*direction.detach()
                _, score = evaluate(candidate)
                value = float(score)
                candidate_costs.append(value)
                if value < chosen_cost:
                    chosen, chosen_cost = candidate.detach(), value
        history.append({"iteration": step, "before": float(cost.detach()), "after": chosen_cost,
                        "line_search_costs": candidate_costs})
        best = chosen
    with torch.no_grad():
        prediction, objective = evaluate(best)
    return best.detach().reshape(shape), {
        "metric": "full_latent_SSE_plus_damping_parameter_energy", "damping": damping,
        "initial_objective": float(initial_cost), "zero_objective": float(origin_cost),
        "retained_incumbent_objective": before, "final_objective": float(objective),
        "predicted_goal_cost": float((prediction-goal).square().mean()),
        "regularization_coordinate": "unconstrained_atanh_action_parameter",
        "regularization_center": "absolute_zero_not_current_iterate",
        "final_parameters": best.detach().reshape(shape).cpu().tolist(),
        "final_parameter_energy": float(best.square().sum()),
        "final_terminal_sse": float((prediction-goal).square().sum()),
        "predicted_terminal_vector": prediction.cpu().tolist(), "iterations": history,
    }
