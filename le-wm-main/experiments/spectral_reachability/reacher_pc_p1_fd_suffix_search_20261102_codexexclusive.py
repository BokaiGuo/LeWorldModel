"""Fixed FD model-only proposal search; diagnostic selection is not P1 admission."""
from __future__ import annotations

import torch

from experiments.spectral_reachability import reacher_pc_p1_common_tail_tangent_20261102_codexexclusive as common
from experiments.spectral_reachability import reacher_pc_p1_constrained_common_tail_20261102_codexexclusive as strict

FD_STEP = 0.01
RADII = (0.05, 0.10)
NORMAL_MIX = 0.25
EPS = 1e-12


def generate(origin, evaluate):
    """21 FD evaluations then 8 candidates; evaluator returns only model scores."""
    if (origin.ndim != 1 or origin.numel() != 10 or not origin.is_floating_point()
            or not torch.isfinite(origin).all()):
        raise ValueError("finite 10D origin required")
    origin = origin.detach()
    with torch.no_grad():
        immediate0, final0 = evaluate(origin)
        strict._costs(immediate0, final0)
        immediate0, final0 = immediate0.detach(), final0.detach()
        observations = []
        for axis in range(10):
            offset = torch.zeros_like(origin)
            offset[axis] = FD_STEP
            ip, fp = evaluate(origin + offset)
            im, fm = evaluate(origin - offset)
            strict._costs(ip, fp)
            strict._costs(im, fm)
            observations.append((ip, fp, im, fm))
        grad_i = torch.stack([(p - m) / (2 * FD_STEP) for p, _, m, _ in observations])
        grad_f = torch.stack([(p - m) / (2 * FD_STEP) for _, p, _, m in observations], dim=1)
        if not torch.isfinite(grad_i).all() or not torch.isfinite(grad_f).all():
            raise ValueError("nonfinite finite-difference gradient")
        norm = float(grad_i.norm())
        if norm <= EPS:
            raise ValueError("zero immediate gradient; abort root without fallback")
        normal = grad_i / norm
        baseline_tail = int(torch.argmin(final0).item())
        targets = (grad_f.mean(dim=0), grad_f[baseline_tail], grad_f[0],
                   (grad_f[1:] - grad_f[0]).mean(dim=0))
        directions = []
        direction_valid = []
        for gradient in targets:
            tangent = -gradient + torch.dot(gradient, normal) * normal
            tangent_norm = float(tangent.norm())
            valid = tangent_norm > EPS
            direction_valid.append(valid)
            # Retain a fixed duplicate for degenerate directions, but it cannot win.
            vector = tangent / tangent_norm + NORMAL_MIX * normal if valid else torch.zeros_like(normal)
            directions.append(vector / vector.norm() if valid else vector)
        bank = [origin.clone()]
        rows = [strict.row(0, origin, origin, immediate0, final0, immediate0, final0, "neutral")]
        probes = [{"axis": i, "plus_immediate": float(p), "plus_final": p_f.cpu().tolist(),
                   "minus_immediate": float(m), "minus_final": m_f.cpu().tolist()}
                  for i, (p, p_f, m, m_f) in enumerate(observations)]
        for direction_index, direction in enumerate(directions):
            for radius in RADII:
                point = origin + radius * direction
                immediate, final = evaluate(point)
                strict._costs(immediate, final)
                index = len(bank)
                row = strict.row(index, point, origin, immediate, final, immediate0, final0,
                                 f"objective{direction_index}_radius{radius:g}")
                tail = int(torch.argmin(final).item())
                row["best_tail_index"] = tail
                row["development_eligible"] = bool(
                    direction_valid[direction_index]
                    and float(immediate) >= float(immediate0) + common.MARGIN
                    and float(final[tail]) <= float(final0[baseline_tail]) - common.MARGIN)
                row["direction_valid"] = direction_valid[direction_index]
                rows.append(row)
                bank.append(point.detach().clone())
        eligible = [r for r in rows[1:] if r["development_eligible"]]
        winner = min(eligible, key=lambda r: (min(r["final_costs"]), r["proposal_index"])) if eligible else None
        strict_eligible = [r for r in rows[1:] if r["eligible"]]
        strict_winner = min(strict_eligible, key=lambda r: (sum(r["per_tail_delta"]), r["proposal_index"])) if strict_eligible else None
    return torch.stack(bank), {
        "proposal_rows": rows,
        "fd_probes": probes,
        "baseline_best_tail_index": baseline_tail,
        "direction_valid": direction_valid,
        "development_selected_index": None if winner is None else winner["proposal_index"],
        "development_selected_tail_index": None if winner is None else winner["best_tail_index"],
        "strict_selected_index": None if strict_winner is None else strict_winner["proposal_index"],
        "logical_costs": {"evaluation_calls": 29, "prefix_predict_samples": 29,
                          "suffix_token_predict_samples": 290, "prefix_gradient_vjp_calls": 0},
        "scope": "model-only development; development selection is not strict mechanism or P1 gate",
    }


def compile_reacher(model, pixels, goal, history, metric, *, lower, upper):
    """Frozen CUDA interface; caller authenticates identities and ledgers."""
    if (pixels.device.type != "cuda" or goal.device.type != "cuda"
            or history.device.type != "cuda" or model.training
            or any(p.requires_grad for p in model.parameters())):
        raise ValueError("frozen evaluation model and host CUDA inputs required")
    if history.ndim != 3 or history.shape[0] != 1 or history.shape[-1] != 10:
        raise ValueError("expected one Reacher action history of 10D packets")
    from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as p0

    origin = history.new_zeros(10)
    tails = common.common_tails(origin)
    target = p0._planning_coordinate(model, p0._goal_latent(model, goal), role="goal").detach()

    def evaluate(point):
        embeddings, actions = p1._state_after_prefix(model, pixels, history, point.reshape(1, 1, 10),
                                                      action_lower=lower, action_upper=upper)
        immediate = p0._planning_coordinate(model, embeddings[:, -1], role="predicted")
        final = p1._terminal_from_state(model, embeddings.expand(5, -1, -1),
                                        actions.expand(5, -1, -1), tails,
                                        action_lower=lower, action_upper=upper)
        return common.q_costs(metric, immediate, target)[0], common.q_costs(
            metric, p0._planning_coordinate(model, final, role="predicted"), target)

    bank, receipt = generate(origin, evaluate)
    receipt["common_suffix_parameters"] = tails.detach().cpu().tolist()
    return bank[:, None, :], receipt
