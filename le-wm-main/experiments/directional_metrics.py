"""Metrics for latent eigenspace directional audits.

The functions in this module intentionally depend only on NumPy/Torch so they
can be unit-tested without the full stable_worldmodel training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


EPS = 1e-12


@dataclass(frozen=True)
class Eigenspace:
    mean: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    num_samples: int


def as_2d_tensor(x: np.ndarray | torch.Tensor, *, dtype=torch.float32) -> torch.Tensor:
    tensor = x if torch.is_tensor(x) else torch.as_tensor(x)
    tensor = tensor.detach().to(dtype=dtype)
    if tensor.ndim != 2:
        raise ValueError(f"expected a 2D array/tensor, got shape {tuple(tensor.shape)}")
    return tensor


def covariance(x: np.ndarray | torch.Tensor, center: bool = True) -> torch.Tensor:
    x = as_2d_tensor(x)
    if x.size(0) < 2:
        raise ValueError("covariance requires at least two samples")
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    return x.T @ x / max(x.size(0) - 1, 1)


def compute_eigenspace(x: np.ndarray | torch.Tensor) -> Eigenspace:
    x = as_2d_tensor(x)
    mean = x.mean(dim=0)
    cov = covariance(x)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    return Eigenspace(
        mean=mean,
        eigenvalues=eigvals[order].clamp_min(0),
        eigenvectors=eigvecs[:, order],
        num_samples=x.size(0),
    )


def spectral_descriptors(
    eigenvalues: np.ndarray | torch.Tensor,
    tail_ks: Iterable[int] = (8, 32, 64, 128),
) -> dict[str, object]:
    """Return compact spectrum shape descriptors for a covariance spectrum."""

    vals = torch.as_tensor(eigenvalues, dtype=torch.float32).detach().flatten().clamp_min(0)
    if vals.numel() == 0:
        raise ValueError("spectral_descriptors requires at least one eigenvalue")
    total = vals.sum().clamp_min(EPS)
    sq_total = vals.square().sum().clamp_min(EPS)
    probs = vals / total
    entropy = -(probs * probs.clamp_min(EPS).log()).sum()
    dim = vals.numel()
    return {
        "dim": int(dim),
        "trace": float(total.item()),
        "lambda_max": float(vals[0].item()),
        "stable_rank": float((total / vals[0].clamp_min(EPS)).item()),
        "participation_ratio": float((total.square() / sq_total).item()),
        "normalized_entropy": float((entropy / torch.log(torch.tensor(float(dim)))).item()) if dim > 1 else 0.0,
        "tail_mass": {
            str(k): float((vals[min(int(k), dim) :].sum() / total).item()) for k in tail_ks
        },
    }


def top_band_energy(
    eigenvectors: np.ndarray | torch.Tensor,
    tangents: np.ndarray | torch.Tensor,
    k: int,
) -> torch.Tensor:
    u = as_2d_tensor(eigenvectors)
    t = as_2d_tensor(tangents)
    k = min(k, u.size(1))
    if t.size(1) != u.size(0):
        raise ValueError(
            f"tangent dim {t.size(1)} does not match eigenspace dim {u.size(0)}"
        )
    proj = t @ u[:, :k]
    return proj.square().sum(dim=1).mean()


def total_energy(tangents: np.ndarray | torch.Tensor) -> torch.Tensor:
    t = as_2d_tensor(tangents)
    return t.square().sum(dim=1).mean()


def top_band_lift(
    eigenvectors: np.ndarray | torch.Tensor,
    tangents: np.ndarray | torch.Tensor,
    k: int,
) -> float:
    u = as_2d_tensor(eigenvectors)
    t = as_2d_tensor(tangents)
    d = u.size(0)
    k = min(k, d)
    denom = (k / d) * total_energy(t).clamp_min(EPS)
    return float((top_band_energy(u, t, k) / denom).item())


def lift_curve(
    eigenvectors: np.ndarray | torch.Tensor,
    tangents: np.ndarray | torch.Tensor,
    ks: Iterable[int],
) -> dict[str, float]:
    return {str(k): top_band_lift(eigenvectors, tangents, int(k)) for k in ks}


def band_energy_profile(
    eigenvectors: np.ndarray | torch.Tensor,
    tangents: np.ndarray | torch.Tensor,
    bands: Iterable[tuple[int, int]],
) -> dict[str, float]:
    """Return normalized tangent energy in inclusive-exclusive rank bands."""

    u = as_2d_tensor(eigenvectors)
    t = as_2d_tensor(tangents)
    if t.size(1) != u.size(0):
        raise ValueError(
            f"tangent dim {t.size(1)} does not match eigenspace dim {u.size(0)}"
        )
    total = total_energy(t).clamp_min(EPS)
    out: dict[str, float] = {}
    for lo, hi in bands:
        lo = max(0, int(lo))
        hi = min(int(hi), u.size(1))
        if hi <= lo:
            continue
        proj = t @ u[:, lo:hi]
        out[f"{lo}:{hi}"] = float((proj.square().sum(dim=1).mean() / total).item())
    return out


def tangent_cosine(
    a: np.ndarray | torch.Tensor,
    b: np.ndarray | torch.Tensor,
) -> float:
    a_t = as_2d_tensor(a)
    b_t = as_2d_tensor(b)
    if a_t.shape != b_t.shape:
        raise ValueError(
            f"tangent shapes must match, got {tuple(a_t.shape)} and {tuple(b_t.shape)}"
        )
    num = (a_t * b_t).sum(dim=1)
    denom = a_t.norm(dim=1).clamp_min(EPS) * b_t.norm(dim=1).clamp_min(EPS)
    return float((num / denom).mean().item())


def feature_space_adapter(
    target_tangents: np.ndarray | torch.Tensor,
    strength: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Build A=(I+lambda Z^T Z)^-1 for selective tangent suppression."""

    z = as_2d_tensor(target_tangents)
    gram = z.T @ z
    if normalize:
        gram = gram / max(z.size(0), 1)
    eye = torch.eye(z.size(1), dtype=z.dtype, device=z.device)
    return torch.linalg.solve(eye + float(strength) * gram, eye)


def apply_feature_adapter(
    adapter: np.ndarray | torch.Tensor,
    tangents: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    a = as_2d_tensor(adapter)
    t = as_2d_tensor(tangents)
    if a.size(0) != a.size(1):
        raise ValueError(f"adapter must be square, got shape {tuple(a.shape)}")
    if t.size(1) != a.size(0):
        raise ValueError(f"tangent dim {t.size(1)} does not match adapter dim {a.size(0)}")
    return t @ a.T


def tangent_retention(
    adapter: np.ndarray | torch.Tensor,
    tangents: np.ndarray | torch.Tensor,
) -> float:
    before = total_energy(tangents).clamp_min(EPS)
    after = total_energy(apply_feature_adapter(adapter, tangents))
    return float((after / before).item())


def shuffled_control(tangents: np.ndarray | torch.Tensor, seed: int = 0) -> torch.Tensor:
    t = as_2d_tensor(tangents)
    g = torch.Generator(device=t.device).manual_seed(seed)
    perm = torch.randperm(t.size(0), generator=g, device=t.device)
    return t[perm]


def gaussian_control_like(tangents: np.ndarray | torch.Tensor, seed: int = 0) -> torch.Tensor:
    t = as_2d_tensor(tangents)
    g = torch.Generator(device=t.device).manual_seed(seed)
    noise = torch.randn(t.shape, generator=g, device=t.device, dtype=t.dtype)
    return noise * t.std(dim=0, keepdim=True).clamp_min(EPS)


def matched_eigenvalue_random_rotation_like(
    tangents: np.ndarray | torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """Gaussian control with the source covariance spectrum but random eigenvectors."""

    t = as_2d_tensor(tangents)
    if t.size(0) < 2:
        return gaussian_control_like(t, seed=seed)
    eig = compute_eigenspace(t)
    rates = eig.eigenvalues.clamp_min(EPS).sqrt()
    basis = random_orthonormal_basis(t.size(1), seed=seed).to(device=t.device, dtype=t.dtype)
    g = torch.Generator(device=t.device).manual_seed(seed)
    noise = torch.randn(t.shape, generator=g, device=t.device, dtype=t.dtype)
    return (noise * rates.unsqueeze(0)) @ basis.T


def random_orthonormal_basis(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g))
    return q


def identity_basis(dim: int) -> torch.Tensor:
    return torch.eye(dim)


def rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    if len(unique) != len(x):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom <= EPS:
        return float("nan")
    return float((x * y).sum() / denom)


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    return pearsonr(rankdata(np.asarray(x)), rankdata(np.asarray(y)))


def cost_success_alignment(
    latent_cost: np.ndarray,
    success: np.ndarray,
    state_distance: np.ndarray,
    goal_distance: np.ndarray | None = None,
) -> dict[str, float]:
    success = np.asarray(success, dtype=np.float64)
    state_distance = np.asarray(state_distance, dtype=np.float64)
    out = {
        "pearson_cost_success": pearsonr(latent_cost, success),
        "spearman_cost_success": spearmanr(latent_cost, success),
        "pearson_cost_state_distance": pearsonr(latent_cost, state_distance),
        "spearman_cost_state_distance": spearmanr(latent_cost, state_distance),
    }
    if goal_distance is not None:
        out["pearson_cost_goal_distance"] = pearsonr(latent_cost, goal_distance)
        out["spearman_cost_goal_distance"] = spearmanr(latent_cost, goal_distance)
    return out


def cluster_states(states: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    try:
        from sklearn.cluster import MiniBatchKMeans

        return MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=max(1024, 4 * n_clusters),
            n_init=10,
        ).fit_predict(states)
    except Exception:
        rng = np.random.default_rng(seed)
        proj = rng.normal(size=(states.shape[1], min(4, states.shape[1])))
        scores = states @ proj
        bins = [
            np.digitize(scores[:, i], np.quantile(scores[:, i], np.linspace(0, 1, 5)[1:-1]))
            for i in range(scores.shape[1])
        ]
        labels = np.zeros(states.shape[0], dtype=np.int64)
        for i, b in enumerate(bins):
            labels += b.astype(np.int64) * (4**i)
        return labels % n_clusters


def summarize_by_cluster(
    latents: np.ndarray,
    tangents: dict[str, np.ndarray],
    labels: np.ndarray,
    ks: Iterable[int],
    min_cluster_size: int = 256,
) -> dict[str, object]:
    latents = np.asarray(latents, dtype=np.float32)
    labels = np.asarray(labels)
    results: dict[str, object] = {"clusters": {}, "weighted_mean": {}}
    weighted: dict[str, list[tuple[int, dict[str, float]]]] = {}

    for label in sorted(np.unique(labels).tolist()):
        mask = labels == label
        n = int(mask.sum())
        if n < max(min_cluster_size, 2):
            continue
        eig = compute_eigenspace(latents[mask])
        cdict: dict[str, object] = {
            "num_samples": n,
            "eigenvalues": eig.eigenvalues.cpu().numpy().tolist(),
            "spectrum": spectral_descriptors(eig.eigenvalues),
        }
        for name, tangent in tangents.items():
            curve = lift_curve(eig.eigenvectors, tangent[mask], ks)
            cdict[f"{name}_lift"] = curve
            weighted.setdefault(name, []).append((n, curve))
        results["clusters"][str(label)] = cdict

    for name, entries in weighted.items():
        total = sum(n for n, _ in entries)
        results["weighted_mean"][f"{name}_lift"] = {
            str(k): float(sum(n * curve[str(k)] for n, curve in entries) / total)
            for k in ks
        }
    return results



def summarize_action_conditioned(
    latents: np.ndarray,
    tangents: dict[str, np.ndarray],
    actions: np.ndarray,
    ks: Iterable[int],
    n_action_bins: int = 8,
    min_bin_size: int = 256,
    seed: int = 0,
) -> dict[str, object]:
    """Audit tangent localization within action bins or action clusters."""

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        actions = actions.reshape(actions.shape[0], -1)
    if actions.shape[1] == 1 and np.unique(actions).size <= n_action_bins:
        labels = actions[:, 0]
    else:
        labels = cluster_states(actions, n_action_bins, seed=seed)
    return summarize_by_cluster(
        latents,
        tangents,
        labels,
        ks,
        min_cluster_size=min_bin_size,
    )
