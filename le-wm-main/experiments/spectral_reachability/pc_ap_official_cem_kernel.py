"""AP CEM kernel matching the pinned upstream algorithm, without model bindings.

This is a separately named development implementation. Existing AP receipts
remain bound to their original NumPy implementation. The caller supplies the
generator and model evaluation; production neural callbacks must use CUDA.
"""
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class OfficialCEMResult:
    normalized_block: torch.Tensor
    initial_cost: float
    refined_cost: float
    refined_selected: bool
    prediction_batches: int
    predicted_blocks: int


def official_cem_kernel(
    evaluate: Callable[[torch.Tensor], torch.Tensor],
    prior: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    generator: torch.Generator,
    *,
    population: int = 300,
    iterations: int = 30,
    elites: int = 30,
) -> OfficialCEMResult:
    """Run the upstream float32 CEM update; callback takes [1,N,1,D]."""
    if prior.ndim != 3 or prior.shape[:2] != (1, 1):
        raise ValueError("prior must have shape [1,1,D]")
    if prior.dtype != torch.float32 or prior.numel() == 0:
        raise ValueError("prior must be nonempty float32")
    if lower.shape != prior.shape[-1:] or upper.shape != lower.shape:
        raise ValueError("bounds must have shape [D]")
    if (lower.device != prior.device or upper.device != prior.device
            or lower.dtype != prior.dtype or upper.dtype != prior.dtype):
        raise ValueError("prior and bounds must share device and dtype")
    if (not torch.isfinite(prior).all() or not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all() or not (lower < upper).all()
            or not ((prior >= lower) & (prior <= upper)).all()):
        raise ValueError("invalid bounds or prior")
    if (not all(isinstance(v, int) for v in (population, iterations, elites))
            or iterations < 1 or population < 2 or not 2 <= elites <= population):
        raise ValueError("invalid budget; sample standard deviation needs two elites")

    def project(x: torch.Tensor) -> torch.Tensor:
        return torch.minimum(torch.maximum(x, lower), upper)

    def score(x: torch.Tensor) -> torch.Tensor:
        result = evaluate(project(x))
        if result.shape != x.shape[:2] or not torch.isfinite(result).all():
            raise ValueError("callback must return finite [1,N] costs")
        return result

    mean = prior.clone()
    std = torch.full_like(mean, 1.0 / 3.0)
    for _ in range(iterations):
        noise = torch.randn(
            1, population, 1, prior.shape[-1], generator=generator,
            device=prior.device, dtype=prior.dtype,
        )
        candidates = noise * std[:, None] + mean[:, None]
        candidates[:, 0] = mean
        values = score(candidates)
        indices = torch.topk(values, elites, dim=1, largest=False).indices
        selected = candidates[torch.arange(1, device=prior.device)[:, None], indices]
        mean = selected.mean(dim=1)
        std = selected.std(dim=1, correction=1).clamp_min(1e-5)

    final = torch.stack((prior, mean), dim=1)
    final_cost = score(final)
    refined = bool((final_cost[0, 1] < final_cost[0, 0]).item())
    chosen = project(final[:, 1] if refined else final[:, 0])
    return OfficialCEMResult(
        normalized_block=chosen.detach().clone(),
        initial_cost=float(final_cost[0, 0]),
        refined_cost=float(final_cost[0, 1]),
        refined_selected=refined,
        prediction_batches=iterations + 1,
        predicted_blocks=population * iterations + 2,
    )
