"""AP-compatible CEM for one or more normalized five-step action packets."""
import torch


def packet_cem(evaluate, prior, lower, upper, generator, *, population=16, iterations=3, elites=4):
    if (prior.ndim != 3 or prior.shape[:2] != (1, 1) or prior.shape[-1] < 1
            or prior.dtype != torch.float32 or lower.shape != prior.shape[-1:]
            or upper.shape != lower.shape or lower.device != prior.device
            or upper.device != prior.device or lower.dtype != prior.dtype or upper.dtype != prior.dtype
            or not torch.isfinite(prior).all() or not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all() or not (lower < upper).all()
            or not ((prior >= lower) & (prior <= upper)).all()):
        raise ValueError("invalid bounded CEM domain")
    if (not all(type(x) is int for x in (population, iterations, elites))
            or population < 5 or iterations < 1 or not 2 <= elites <= population):
        raise ValueError("invalid CEM budget")

    def project(x):
        return torch.minimum(torch.maximum(x, lower), upper)

    def costs(x):
        value = evaluate(project(x))
        if value.shape != x.shape[:2] or not torch.isfinite(value).all():
            raise ValueError("invalid CEM scores")
        return value

    mean, std = prior.clone(), torch.full_like(prior, 1. / 3.)
    for _ in range(iterations):
        noise = torch.randn(1, population, 1, prior.shape[-1], generator=generator,
                            device=prior.device, dtype=prior.dtype)
        proposals = noise * std[:, None] + mean[:, None]
        proposals[:, 0] = mean
        values = costs(proposals)
        indices = torch.topk(values, elites, dim=1, largest=False).indices
        selected = proposals[torch.arange(1, device=prior.device)[:, None], indices]
        mean, std = selected.mean(dim=1), selected.std(dim=1, correction=1).clamp_min(1e-5)
    final = torch.stack((prior, mean), dim=1)
    final_scores = costs(final)
    refined = bool((final_scores[0, 1] < final_scores[0, 0]).item())
    return (project(final[:, 1] if refined else final[:, 0]).detach(),
            float(final_scores[0, 0]), float(final_scores[0, 1]), refined)
