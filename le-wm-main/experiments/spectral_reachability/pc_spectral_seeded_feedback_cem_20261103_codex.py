"""Training-free spectral candidate source for a shared feedback CEM.

This module does not access environments, model weights, observations or
physical outcomes; the caller is responsible for those contracts and for
accounting for actual work in the supplied callbacks.
"""
import torch

AMPLITUDES = (.25, -.25, .5, -.5)


def spectral_candidates(response, action_mean, action_std):
    """Return normalized five-action candidates and the raw-action Jacobian."""
    mean = torch.as_tensor(action_mean)
    std = torch.as_tensor(action_std, device=mean.device, dtype=mean.dtype)
    if (mean.shape != (2,) or std.shape != (2,) or not mean.is_floating_point()
            or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all())
            or not bool((std > 0).all())):
        raise ValueError("invalid action normalization")
    mean10, std10 = mean.repeat(5), std.repeat(5)
    zero = (-mean10 / std10).detach().requires_grad_(True)
    value = response(zero)
    if (not isinstance(value, torch.Tensor) or value.shape != (2,)
            or not bool(torch.isfinite(value).all()) or not value.requires_grad):
        raise ValueError("response must be a differentiable finite 2-vector")
    rows = []
    try:
        for index in range(2):
            derivative, = torch.autograd.grad(value[index], zero, retain_graph=index == 0)
            rows.append(derivative)
    except RuntimeError as exc:
        raise ValueError("response is not differentiable with respect to actions") from exc
    jacobian = torch.stack(rows).detach() / std10.unsqueeze(0)
    if not bool(torch.isfinite(jacobian).all()):
        raise ValueError("nonfinite raw-action response")
    _, singular, vh = torch.linalg.svd(jacobian, full_matrices=False)
    if not bool(torch.isfinite(singular).all()) or float(singular[0]) <= 1e-8:
        raise ValueError("degenerate task response")
    direction = vh[0]
    pivot = int(torch.argmax(direction.abs()))
    if float(direction[pivot]) < 0:
        direction = -direction
    direction = direction / direction.abs().max()
    raw = torch.stack([amplitude * direction for amplitude in AMPLITUDES])
    if not bool(torch.isfinite(raw).all()) or bool((raw.abs() > 1).any()):
        raise ValueError("spectral proposals out of raw action bounds")
    normalized = (raw - mean10) / std10
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError("nonfinite normalized candidates")
    return {"raw": raw.detach(), "normalized": normalized.detach(),
            "jacobian_raw": jacobian, "singular_values": singular.detach(),
            "direction_raw": direction.detach(),
            "structure_counts": {"response_forward_calls": 1, "reverse_vector_calls": 2, "svd_calls": 1}}


def seeded_cem(evaluate, prior, lower, upper, noise, injected=None):
    """Run the historical CEM update with optional first-generation seeds."""
    if (prior.shape != (10,) or lower.shape != (10,) or upper.shape != (10,)
            or noise.shape != (3, 16, 10)
            or any(t.device != prior.device or t.dtype != prior.dtype for t in (lower, upper, noise))
            or not all(bool(torch.isfinite(t).all()) for t in (prior, lower, upper, noise))
            or bool((lower >= upper).any())):
        raise ValueError("invalid CEM inputs")
    if injected is not None:
        if (not isinstance(injected, torch.Tensor) or injected.shape != (4, 10)
                or injected.device != prior.device or injected.dtype != prior.dtype
                or not bool(torch.isfinite(injected).all())
                or bool((injected < lower).any() or (injected > upper).any())):
            raise ValueError("invalid injected candidates")
    mean, std = prior.clone(), torch.full_like(prior, 1 / 3)
    trace, samples, batches = [], 0, 0

    def score(candidates):
        nonlocal samples, batches
        values = evaluate(candidates)
        if (not isinstance(values, torch.Tensor) or values.shape != (len(candidates),)
                or values.device != prior.device or not bool(torch.isfinite(values).all())):
            raise ValueError("invalid CEM scores")
        samples += len(candidates)
        batches += 1
        return values

    for iteration in range(3):
        candidates = (mean + noise[iteration] * std).clamp(lower, upper)
        candidates[0] = mean.clamp(lower, upper)
        if iteration == 0 and injected is not None:
            candidates[1:5] = injected
        scores = score(candidates)
        elite = torch.argsort(scores, stable=True)[:4]
        chosen = candidates[elite]
        mean, std = chosen.mean(0), chosen.std(0, correction=1).clamp_min(1e-5)
        trace.append(tuple(t.detach().clone() for t in (candidates, scores, elite, mean, std)))
    final = torch.stack((prior, mean)).clamp(lower, upper)
    final_scores = score(final)
    winner = int(final_scores[1] < final_scores[0])
    return {"chosen": final[winner].detach().clone(), "trace": trace,
            "final": final.detach().clone(), "final_scores": final_scores.detach().clone(),
            "winner": winner, "cem_counts": {"samples": samples, "batches": batches}}
