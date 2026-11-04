"""AP-compatible CEM update with optional first-generation spectral proposals."""
import torch


def spectral_cem(evaluate, prior, lower, upper, generator, injected=None,
                 *, population=16, iterations=3, elites=4,
                 projected_elite_update=False):
    """Optimize a packet; default preserves the pinned AP-compatible update.

    With projected_elite_update, fit the sampling distribution to the bounded
    packets actually scored. This is a distinct experimental variant, not
    the official baseline or a retroactive change to existing receipts.
    """
    if not isinstance(projected_elite_update, bool):
        raise ValueError("projected_elite_update must be boolean")
    if (prior.shape != (1, 1, 10) or prior.dtype != torch.float32
            or lower.shape != (10,) or upper.shape != (10,)
            or prior.device != lower.device or lower.device != upper.device
            or not torch.isfinite(prior).all() or not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all() or not (lower < upper).all()
            or not ((prior >= lower) & (prior <= upper)).all()):
        raise ValueError("invalid CEM domain")
    if population < 5 or iterations < 1 or not 2 <= elites <= population:
        raise ValueError("invalid CEM budget")
    if injected is not None and (
            injected.shape != (4, 10) or injected.device != prior.device
            or injected.dtype != prior.dtype or not torch.isfinite(injected).all()
            or not ((injected >= lower) & (injected <= upper)).all()):
        raise ValueError("invalid injected normalized packets")

    def project(x):
        return torch.minimum(torch.maximum(x, lower), upper)

    def costs(x):
        y = evaluate(project(x))
        if y.shape != x.shape[:2] or not torch.isfinite(y).all():
            raise ValueError("invalid CEM scores")
        return y

    mean, std = prior.clone(), torch.full_like(prior, 1. / 3.)
    for iteration in range(iterations):
        noise = torch.randn(1, population, 1, 10, generator=generator,
                            device=prior.device, dtype=prior.dtype)
        proposals = noise * std[:, None] + mean[:, None]
        proposals[:, 0] = mean
        if iteration == 0 and injected is not None:
            proposals[0, 1:5, 0] = injected
        values = costs(proposals)
        indices = torch.topk(values, elites, dim=1, largest=False).indices
        update_packets = project(proposals) if projected_elite_update else proposals
        selected = update_packets[torch.arange(1, device=prior.device)[:, None], indices]
        mean, std = selected.mean(dim=1), selected.std(dim=1, correction=1).clamp_min(1e-5)
    final = torch.stack((prior, mean), dim=1)
    final_scores = costs(final)
    refined = bool((final_scores[0, 1] < final_scores[0, 0]).item())
    return (project(final[:, 1] if refined else final[:, 0]).detach(),
            float(final_scores[0, 0]), float(final_scores[0, 1]), refined)
