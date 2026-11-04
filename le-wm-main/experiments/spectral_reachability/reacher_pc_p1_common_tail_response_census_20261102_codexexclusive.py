"""Diagnostic Q-weighted response census; does not change any selection gate."""
import torch


def census(metric, neutral_prefix_terminals, candidate_terminals):
    """Compare the same five predicted tail endpoints for two fixed prefixes.

    Rows are neutral, +axis0, -axis0, +axis1, -axis1, all at identical tail
    parameters. Input is predicted latent coordinates only, never a physical
    outcome. A ratio is descriptive, not an admission decision.
    """
    a, b = neutral_prefix_terminals, candidate_terminals
    if (a.ndim != 2 or a.shape[0] != 5 or a.shape != b.shape or
            not torch.isfinite(a).all() or not torch.isfinite(b).all() or
            a.device != b.device):
        raise ValueError("expected aligned finite 5-by-latent same-tail endpoints")
    def measure(values):
        transformed = metric.apply((values - values[:1]).double().T)
        if transformed.ndim != 2 or transformed.shape[1] != 5 or not torch.isfinite(transformed).all():
            raise ValueError("invalid Q-transformed response")
        centered = torch.stack(((transformed[:, 1] - transformed[:, 2]) / 2,
                                (transformed[:, 3] - transformed[:, 4]) / 2), dim=1)
        singular = torch.linalg.svdvals(centered)
        return centered, singular
    baseline, base_singular = measure(a)
    candidate, candidate_singular = measure(b)
    base_norm, candidate_norm = float(baseline.norm()), float(candidate.norm())
    return {"neutral_prefix_response_frobenius": base_norm,
            "candidate_response_frobenius": candidate_norm,
            "response_frobenius_ratio": None if base_norm <= 1e-12 else candidate_norm/base_norm,
            "neutral_prefix_singular_values": base_singular.detach().cpu().tolist(),
            "candidate_singular_values": candidate_singular.detach().cpu().tolist(),
            "singular_value_ratios": [None if float(x) <= 1e-12 else float(y/x)
                                      for x, y in zip(base_singular, candidate_singular, strict=True)],
            "tail_order": ["neutral", "+axis0", "-axis0", "+axis1", "-axis1"],
            "changes_existing_selector": False,
            "interpretation": "finite model-response diagnostics only; neither physical capability "
                              "nor nonlinear controllability is certified"}
