"""Empirical spectral audit metrics for LeWorldModel latents.

This module is separate from the theory package.  It implements the proposal's
eigenvector-sensitive diagnostics over collected LeWM arrays:

- DTL: dynamics/action top-band lift.
- RFP: rank-wise transfer fingerprint.
- PSE: prediction spectral error.
- PPTE: planning-projected tangent/covariance energy.
- NCR: nuisance contamination of planner-facing bands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.directional_metrics import (
    EPS,
    apply_feature_adapter,
    band_energy_profile,
    cluster_states,
    compute_eigenspace,
    feature_space_adapter,
    gaussian_control_like,
    lift_curve,
    matched_eigenvalue_random_rotation_like,
    random_orthonormal_basis,
    tangent_cosine,
    top_band_energy,
    top_band_lift,
    total_energy,
)
from experiments.io_utils import to_numpy


DEFAULT_RANK_FRACTIONS = ((0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.0))


@dataclass(frozen=True)
class EmpiricalAuditConfig:
    condition: str = "states"
    clusters: int = 32
    ks: tuple[int, ...] = (4, 8, 16, 32, 64)
    min_cell_size: int = 256
    band_mode: str = "rank"
    horizons: tuple[int, ...] = (1,)
    seed: int = 3072
    include_interventions: bool = True


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {arr.shape}")
    return arr


def _safe_2d(x: np.ndarray, n: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    else:
        arr = arr.reshape(arr.shape[0], -1)
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"array length {arr.shape[0]} does not match expected length {n}")
    return arr


def _unit_direction(direction: np.ndarray) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= EPS:
        raise ValueError("planning direction has near-zero norm")
    return vec / norm


def rank_fraction_bands(
    dim: int,
    fractions: Iterable[tuple[float, float]] = DEFAULT_RANK_FRACTIONS,
) -> list[tuple[int, int]]:
    """Convert rank fractions to non-empty inclusive-exclusive bands."""

    if dim <= 0:
        raise ValueError("dim must be positive")
    bands: list[tuple[int, int]] = []
    last_hi = 0
    for lo_f, hi_f in fractions:
        lo = max(last_hi, int(np.floor(float(lo_f) * dim)))
        hi = int(np.ceil(float(hi_f) * dim))
        hi = min(dim, max(hi, lo + 1))
        if hi > lo:
            bands.append((lo, hi))
            last_hi = hi
    if not bands or bands[-1][1] < dim:
        bands.append((bands[-1][1] if bands else 0, dim))
    return bands


def cumulative_energy_bands(
    eigenvalues: np.ndarray,
    cuts: Iterable[float] = (0.2, 0.8),
) -> list[tuple[int, int]]:
    """Build bands by cumulative latent covariance energy."""

    vals = np.asarray(eigenvalues, dtype=np.float64).clip(min=0)
    if vals.ndim != 1 or vals.size == 0:
        raise ValueError("eigenvalues must be a non-empty 1D array")
    total = vals.sum()
    if total <= EPS:
        return [(0, vals.size)]
    cumulative = np.cumsum(vals) / total
    edges = [0]
    for cut in cuts:
        idx = int(np.searchsorted(cumulative, float(cut), side="left") + 1)
        idx = min(vals.size, max(edges[-1] + 1, idx))
        edges.append(idx)
    if edges[-1] < vals.size:
        edges.append(vals.size)
    return [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]


def rank_wise_transfer_fingerprint(
    eigenvectors: np.ndarray,
    tangents: np.ndarray,
    bands: Iterable[tuple[int, int]],
) -> dict[str, float]:
    """Normalized tangent energy profile over rank bands."""

    return band_energy_profile(eigenvectors, tangents, bands)


def rank_energy_vector(eigenvectors: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    basis = _as_2d(eigenvectors)
    tangent = _as_2d(tangents)
    if tangent.shape[1] != basis.shape[0]:
        raise ValueError(f"tangent dim {tangent.shape[1]} does not match basis dim {basis.shape[0]}")
    energy = np.square(tangent @ basis).mean(axis=0)
    total = float(energy.sum())
    return energy / max(total, EPS)


def prediction_spectral_error(
    eigenvectors: np.ndarray,
    true_tangent: np.ndarray,
    predicted_tangent: np.ndarray,
    bands: Iterable[tuple[int, int]],
) -> dict[str, object]:
    """Decompose prediction tangent error into eigenspace bands."""

    true = _as_2d(true_tangent)
    pred = _as_2d(predicted_tangent)
    if true.shape != pred.shape:
        raise ValueError(f"true and predicted tangents must match, got {true.shape} and {pred.shape}")
    residual = pred - true
    true_profile = rank_wise_transfer_fingerprint(eigenvectors, true, bands)
    pred_profile = rank_wise_transfer_fingerprint(eigenvectors, pred, bands)
    residual_profile = rank_wise_transfer_fingerprint(eigenvectors, residual, bands)
    profile_l1 = sum(abs(true_profile[k] - pred_profile.get(k, 0.0)) for k in true_profile)
    return {
        "mse": float(np.square(residual).mean()),
        "mean_cosine": tangent_cosine(true, pred),
        "profile_l1": float(profile_l1),
        "true_fingerprint": true_profile,
        "predicted_fingerprint": pred_profile,
        "residual_fingerprint": residual_profile,
    }


def planning_projected_tangent_energy(
    tangents: np.ndarray,
    planning_direction: np.ndarray,
) -> dict[str, float]:
    """Energy of tangent displacement along a planning margin direction."""

    t = _as_2d(tangents)
    u = _unit_direction(planning_direction)
    if t.shape[1] != u.shape[0]:
        raise ValueError(f"tangent dim {t.shape[1]} does not match direction dim {u.shape[0]}")
    projected = np.square(t @ u).mean()
    total = float(to_numpy(total_energy(t)))
    share = projected / max(total, EPS)
    lift = share / max(1.0 / t.shape[1], EPS)
    return {
        "energy": float(projected),
        "share": float(share),
        "lift": float(lift),
    }


def projected_covariance_lift(
    samples: np.ndarray,
    planning_direction: np.ndarray,
) -> dict[str, float]:
    """PPTE_lift for latent covariance along a planning direction."""

    x = _as_2d(samples)
    u = _unit_direction(planning_direction)
    if x.shape[1] != u.shape[0]:
        raise ValueError(f"sample dim {x.shape[1]} does not match direction dim {u.shape[0]}")
    centered = x - x.mean(axis=0, keepdims=True)
    projected = float(np.square(centered @ u).mean())
    trace = float(np.square(centered).sum(axis=1).mean())
    share = projected / max(trace, EPS)
    lift = share / max(1.0 / x.shape[1], EPS)
    return {
        "energy": projected,
        "share": float(share),
        "lift": float(lift),
    }


def nuisance_contamination_ratio(
    eigenvectors: np.ndarray,
    planning_tangent: np.ndarray,
    nuisance_tangent: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Compare planning/action energy and nuisance energy in a top-k subspace."""

    plan = _as_2d(planning_tangent)
    nuis = _as_2d(nuisance_tangent)
    basis = _as_2d(eigenvectors)
    k = min(int(k), basis.shape[1])
    plan_share = float(to_numpy(top_band_energy(basis, plan, k) / total_energy(plan).clamp_min(EPS)))
    nuis_share = float(to_numpy(top_band_energy(basis, nuis, k) / total_energy(nuis).clamp_min(EPS)))
    return {
        "k": int(k),
        "planning_share": plan_share,
        "nuisance_share": nuis_share,
        "nuisance_over_planning": float(nuis_share / max(plan_share, EPS)),
        "planning_purity": float(plan_share / max(nuis_share, EPS)),
        "planning_lift": top_band_lift(basis, plan, k),
        "nuisance_lift": top_band_lift(basis, nuis, k),
    }


def norm_matched_random_tangents(tangents: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random directions with the same per-row norms as the source tangents."""

    t = _as_2d(tangents)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=t.shape).astype(np.float32)
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    noise = noise / np.maximum(noise_norm, EPS)
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    return noise * norms


def isotropic_gaussian_tangents_like(tangents: np.ndarray, seed: int = 0) -> np.ndarray:
    t = _as_2d(tangents)
    rng = np.random.default_rng(seed)
    scale = float(np.sqrt(np.square(t).sum(axis=1).mean() / max(t.shape[1], 1)))
    return rng.normal(scale=scale, size=t.shape).astype(np.float32)


def matched_eigenvalue_random_rotation(dim: int, seed: int = 0) -> np.ndarray:
    return to_numpy(random_orthonormal_basis(dim, seed=seed))


def load_latent_arrays(path: str | Path, max_samples: int = 0, seed: int = 3072) -> dict[str, np.ndarray]:
    """Load a latent NPZ, optionally with deterministic row subsampling.

    Compressed `.npz` files cannot be memory-mapped.  Loading keys one at a
    time keeps peak memory bounded by the largest source array plus the sampled
    output arrays, instead of materializing every full array at once.
    """

    with np.load(path) as data:
        if "latents" not in data.files:
            raise ValueError(f"{path} does not contain required array 'latents'")
        n = int(data["latents"].shape[0])
        rows = None
        if max_samples and n > int(max_samples):
            rng = np.random.default_rng(seed)
            rows = np.sort(rng.choice(n, size=int(max_samples), replace=False))
        arrays = {}
        for key in data.files:
            value = data[key]
            if rows is not None and value.shape[:1] == (n,):
                value = value[rows]
            else:
                value = np.array(value)
            arrays[key] = value
    if rows is not None:
        arrays["_sample_index"] = rows.astype(np.int64)
    return arrays


def _same_shape_tangent(arrays: dict[str, np.ndarray], key: str, latents: np.ndarray) -> np.ndarray | None:
    if key not in arrays:
        return None
    value = _safe_2d(arrays[key], n=latents.shape[0])
    if value.shape[1] != latents.shape[1]:
        return None
    return value.astype(np.float32, copy=False)


def build_tangent_families(
    arrays: dict[str, np.ndarray],
    horizons: Iterable[int] = (1,),
) -> dict[str, np.ndarray]:
    """Collect base, multi-horizon, physical, and counterfactual tangent families."""

    latents = _as_2d(arrays["latents"])
    tangents: dict[str, np.ndarray] = {}
    for key in ["dynamics", "planning", "nuisance", "predicted_dynamics", "prediction_residual"]:
        value = _same_shape_tangent(arrays, key, latents)
        if value is not None:
            tangents[key] = value
    if "prediction_residual" not in tangents and {"dynamics", "predicted_dynamics"}.issubset(tangents):
        tangents["prediction_residual"] = tangents["dynamics"] - tangents["predicted_dynamics"]

    for horizon in horizons:
        h = int(horizon)
        suffix = "" if h == 1 else f"_h{h}"
        dyn_key = "dynamics" if h == 1 else f"dynamics_h{h}"
        pred_key = "predicted_dynamics" if h == 1 else f"predicted_dynamics_h{h}"
        res_key = "prediction_residual" if h == 1 else f"prediction_residual_h{h}"
        dyn = _same_shape_tangent(arrays, dyn_key, latents)
        pred = _same_shape_tangent(arrays, pred_key, latents)
        res = _same_shape_tangent(arrays, res_key, latents)
        if dyn is not None:
            tangents[f"dynamics{suffix}"] = dyn
        if pred is not None:
            tangents[f"predicted_dynamics{suffix}"] = pred
        if res is not None:
            tangents[f"prediction_residual{suffix}"] = res
        elif dyn is not None and pred is not None:
            tangents[f"prediction_residual{suffix}"] = dyn - pred

    for key in sorted(arrays):
        if key.startswith(("physical_", "counterfactual_")):
            value = _same_shape_tangent(arrays, key, latents)
            if value is not None:
                tangents[key] = value
    return tangents


def _quantile_labels(values: np.ndarray, bins: int) -> np.ndarray:
    flat = _safe_2d(values)
    if flat.shape[1] > 1:
        score = flat[:, 0]
    else:
        score = flat.reshape(-1)
    cuts = np.unique(np.quantile(score, np.linspace(0, 1, bins + 1)[1:-1]))
    if cuts.size == 0:
        return np.zeros(score.shape[0], dtype=np.int64)
    return np.digitize(score, cuts).astype(np.int64)


def build_condition_labels(arrays: dict[str, np.ndarray], config: EmpiricalAuditConfig) -> np.ndarray:
    latents = _as_2d(arrays["latents"])
    condition = config.condition
    if condition == "global":
        return np.zeros(latents.shape[0], dtype=np.int64)
    if condition == "actions":
        if "actions" not in arrays:
            raise ValueError("condition='actions' requires an 'actions' array")
        return cluster_states(_safe_2d(arrays["actions"], latents.shape[0]), config.clusters, seed=config.seed)
    if condition == "latents":
        return cluster_states(latents, config.clusters, seed=config.seed)
    if condition == "states":
        states = _safe_2d(arrays.get("states", latents), latents.shape[0])
        return cluster_states(states, config.clusters, seed=config.seed)
    if condition == "goals":
        if "goals" not in arrays:
            raise ValueError("condition='goals' requires a 'goals' array")
        return cluster_states(_safe_2d(arrays["goals"], latents.shape[0]), config.clusters, seed=config.seed)
    if condition == "episode":
        if "episode_id" not in arrays:
            raise ValueError("condition='episode' requires an 'episode_id' array")
        return _safe_2d(arrays["episode_id"], latents.shape[0]).reshape(-1).astype(np.int64)
    if condition == "phase":
        if "timestep" not in arrays:
            raise ValueError("condition='phase' requires a 'timestep' array")
        return _quantile_labels(arrays["timestep"], config.clusters)
    raise ValueError(f"unknown condition {condition!r}")


def _weighted_curve_mean(entries: list[tuple[int, dict[str, float]]], ks: Iterable[int]) -> dict[str, float]:
    total = sum(n for n, _ in entries)
    return {
        str(k): float(sum(n * curve[str(k)] for n, curve in entries) / max(total, 1))
        for k in ks
    }


def _mean_scalar(entries: list[tuple[int, float]]) -> float:
    total = sum(n for n, _ in entries)
    return float(sum(n * value for n, value in entries) / max(total, 1))


def _cell_audit(
    latents: np.ndarray,
    tangents: dict[str, np.ndarray],
    ks: Iterable[int],
    band_mode: str,
    seed: int,
    planning_direction: np.ndarray | None,
) -> dict[str, object]:
    eig = compute_eigenspace(latents)
    eigvecs = to_numpy(eig.eigenvectors)
    eigvals = to_numpy(eig.eigenvalues)
    bands = cumulative_energy_bands(eigvals) if band_mode == "energy" else rank_fraction_bands(latents.shape[1])
    out: dict[str, object] = {
        "num_samples": int(latents.shape[0]),
        "bands": [f"{lo}:{hi}" for lo, hi in bands],
        "eigenvalues": eigvals.tolist(),
        "lifts": {},
        "fingerprints": {},
    }
    for name, tangent in tangents.items():
        out["lifts"][name] = lift_curve(eigvecs, tangent, ks)
        out["fingerprints"][name] = rank_wise_transfer_fingerprint(eigvecs, tangent, bands)

    if "dynamics" in tangents:
        random_basis = matched_eigenvalue_random_rotation(latents.shape[1], seed=seed)
        random_tangent = norm_matched_random_tangents(tangents["dynamics"], seed=seed)
        random_rotation_tangent = matched_eigenvalue_random_rotation_like(tangents["dynamics"], seed=seed + 31)
        isotropic = isotropic_gaussian_tangents_like(tangents["dynamics"], seed=seed + 17)
        out["controls"] = {
            "random_basis_dynamics_lift": lift_curve(random_basis, tangents["dynamics"], ks),
            "matched_eigenvalue_random_rotation_dynamics_lift": lift_curve(eigvecs, random_rotation_tangent, ks),
            "norm_matched_random_dynamics_lift": lift_curve(eigvecs, random_tangent, ks),
            "diagonal_gaussian_dynamics_lift": lift_curve(eigvecs, to_numpy(gaussian_control_like(tangents["dynamics"], seed=seed)), ks),
            "isotropic_gaussian_dynamics_lift": lift_curve(eigvecs, isotropic, ks),
        }

    if {"dynamics", "predicted_dynamics"}.issubset(tangents):
        out["prediction_spectral_error"] = prediction_spectral_error(
            eigvecs, tangents["dynamics"], tangents["predicted_dynamics"], bands
        )

    if planning_direction is not None:
        out["ppte_latent"] = projected_covariance_lift(latents, planning_direction)
        out["ppte_tangents"] = {
            name: planning_projected_tangent_energy(tangent, planning_direction)
            for name, tangent in tangents.items()
            if name in {"dynamics", "planning", "predicted_dynamics", "prediction_residual", "nuisance"}
        }

    if {"planning", "nuisance"}.issubset(tangents):
        out["nuisance_contamination"] = {
            str(k): nuisance_contamination_ratio(eigvecs, tangents["planning"], tangents["nuisance"], int(k))
            for k in ks
        }
    return out


def summarize_conditioned_latent_audit(
    latents: np.ndarray,
    tangents: dict[str, np.ndarray],
    labels: np.ndarray,
    ks: Iterable[int],
    min_cell_size: int = 256,
    band_mode: str = "rank",
    seed: int = 0,
    planning_direction: np.ndarray | None = None,
) -> dict[str, object]:
    """Run the empirical audit independently per condition cell."""

    x = _as_2d(latents)
    labels = np.asarray(labels)
    if labels.shape[0] != x.shape[0]:
        raise ValueError(f"labels length {labels.shape[0]} does not match latents length {x.shape[0]}")
    clean_tangents = {name: _as_2d(value) for name, value in tangents.items()}
    for name, value in clean_tangents.items():
        if value.shape != x.shape:
            raise ValueError(f"tangent {name} has shape {value.shape}, expected {x.shape}")

    ks = [int(k) for k in ks]
    results: dict[str, object] = {"cells": {}, "weighted_mean": {}, "config": {}}
    weighted_lifts: dict[str, list[tuple[int, dict[str, float]]]] = {}
    weighted_controls: dict[str, list[tuple[int, dict[str, float]]]] = {}
    weighted_pse: dict[str, list[tuple[int, float]]] = {}
    weighted_ppte: dict[str, list[tuple[int, float]]] = {}
    weighted_ncr: dict[str, list[tuple[int, float]]] = {}

    for label in sorted(np.unique(labels).tolist()):
        mask = labels == label
        n = int(mask.sum())
        if n < max(int(min_cell_size), 2):
            continue
        local_tangents = {name: value[mask] for name, value in clean_tangents.items()}
        local_direction = planning_direction
        if local_direction is None and "planning" in local_tangents:
            mean_plan = local_tangents["planning"].mean(axis=0)
            if np.linalg.norm(mean_plan) > EPS:
                local_direction = mean_plan
        label_seed = sum(ord(ch) for ch in str(label))
        cell = _cell_audit(
            x[mask],
            local_tangents,
            ks,
            band_mode=band_mode,
            seed=seed + label_seed,
            planning_direction=local_direction,
        )
        results["cells"][str(label)] = cell
        for name, curve in cell["lifts"].items():
            weighted_lifts.setdefault(name, []).append((n, curve))
        for name, curve in cell.get("controls", {}).items():
            weighted_controls.setdefault(name, []).append((n, curve))
        pse = cell.get("prediction_spectral_error")
        if isinstance(pse, dict):
            weighted_pse.setdefault("mse", []).append((n, float(pse["mse"])))
            weighted_pse.setdefault("mean_cosine", []).append((n, float(pse["mean_cosine"])))
            weighted_pse.setdefault("profile_l1", []).append((n, float(pse["profile_l1"])))
        ppte_latent = cell.get("ppte_latent")
        if isinstance(ppte_latent, dict):
            weighted_ppte.setdefault("latent_lift", []).append((n, float(ppte_latent["lift"])))
        if isinstance(cell.get("nuisance_contamination"), dict):
            for k, value in cell["nuisance_contamination"].items():
                weighted_ncr.setdefault(f"{k}_nuisance_over_planning", []).append(
                    (n, float(value["nuisance_over_planning"]))
                )
                weighted_ncr.setdefault(f"{k}_planning_purity", []).append((n, float(value["planning_purity"])))

    results["weighted_mean"]["lifts"] = {
        name: _weighted_curve_mean(entries, ks) for name, entries in weighted_lifts.items()
    }
    results["weighted_mean"]["controls"] = {
        name: _weighted_curve_mean(entries, ks) for name, entries in weighted_controls.items()
    }
    results["weighted_mean"]["prediction_spectral_error"] = {
        name: _mean_scalar(entries) for name, entries in weighted_pse.items()
    }
    results["weighted_mean"]["ppte"] = {
        name: _mean_scalar(entries) for name, entries in weighted_ppte.items()
    }
    results["weighted_mean"]["ncr"] = {
        name: _mean_scalar(entries) for name, entries in weighted_ncr.items()
    }
    results["config"] = {
        "ks": ks,
        "min_cell_size": int(min_cell_size),
        "band_mode": band_mode,
        "num_cells": len(results["cells"]),
        "num_samples": int(x.shape[0]),
    }
    return results


def _permuted_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return labels[rng.permutation(labels.shape[0])]


def _permuted_tangents(tangents: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    if not tangents:
        return {}
    n = next(iter(tangents.values())).shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return {name: value[perm] for name, value in tangents.items()}


def _summarize_control(
    latents: np.ndarray,
    tangents: dict[str, np.ndarray],
    labels: np.ndarray,
    config: EmpiricalAuditConfig,
    name: str,
) -> dict[str, object]:
    summary = summarize_conditioned_latent_audit(
        latents,
        tangents,
        labels,
        config.ks,
        min_cell_size=config.min_cell_size,
        band_mode=config.band_mode,
        seed=config.seed + 1009,
    )
    return {
        "name": name,
        "num_cells": summary["config"]["num_cells"],
        "weighted_mean": summary["weighted_mean"],
    }


def multi_horizon_report(
    latents: np.ndarray,
    tangents: dict[str, np.ndarray],
    ks: Iterable[int],
    seed: int = 0,
) -> dict[str, object]:
    eig = compute_eigenspace(latents)
    eigvecs = to_numpy(eig.eigenvectors)
    out = {}
    for name, tangent in sorted(tangents.items()):
        if name.startswith("dynamics_h") or name == "dynamics":
            out[name] = {
                "lift": lift_curve(eigvecs, tangent, ks),
                "norm": float(np.sqrt(np.square(tangent).sum(axis=1).mean())),
            }
            pred_name = name.replace("dynamics", "predicted_dynamics", 1)
            if pred_name in tangents:
                out[name]["predicted_cosine"] = tangent_cosine(tangent, tangents[pred_name])
    if len(out) >= 2:
        first_key = sorted(out)[0]
        first_lift = out[first_key]["lift"]
        out["horizon_collapse"] = {
            key: {str(k): float(value["lift"][str(k)] / max(first_lift[str(k)], EPS)) for k in ks}
            for key, value in out.items()
            if isinstance(value, dict) and "lift" in value
        }
    return out


def controllability_spectrum_report(
    latents: np.ndarray,
    tangents: dict[str, np.ndarray],
    actions: np.ndarray | None,
    ks: Iterable[int],
    labels: np.ndarray | None = None,
    min_cell_size: int = 256,
) -> dict[str, object]:
    if actions is None or "dynamics" not in tangents:
        return {}
    ks = tuple(int(k) for k in ks)
    if not ks:
        raise ValueError("ks must contain at least one rank")
    latents = _as_2d(latents)
    actions = _safe_2d(actions, latents.shape[0])
    dynamics = _safe_2d(tangents["dynamics"], latents.shape[0])

    def _fit_slice(mask: np.ndarray) -> dict[str, object]:
        source = latents[mask]
        x = actions[mask]
        y = dynamics[mask]
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        controllable = x @ coef
        residual = y - controllable
        eigvecs = to_numpy(compute_eigenspace(source).eigenvectors)
        ss_res = float(np.square(residual).sum())
        ss_tot = float(np.square(y - y.mean(axis=0, keepdims=True)).sum())
        dynamics_lift = lift_curve(eigvecs, y, ks)
        controllable_lift = lift_curve(eigvecs, controllable, ks)
        unexplained_lift = lift_curve(eigvecs, residual, ks)
        return {
            "n_samples": int(mask.sum()),
            "action_to_dynamics_r2": float(1.0 - ss_res / max(ss_tot, EPS)),
            "dynamics_lift": dynamics_lift,
            "controllable_lift": controllable_lift,
            "unexplained_lift": unexplained_lift,
            "dynamics_minus_controllable_lift": {
                str(k): float(dynamics_lift[str(k)] - controllable_lift[str(k)]) for k in ks
            },
        }

    result = _fit_slice(np.ones(latents.shape[0], dtype=bool))
    # Keep the original top-level fields stable for downstream summaries.
    result["controllability_conditioning"] = "global"
    if labels is None:
        return {
            "action_to_dynamics_r2": result["action_to_dynamics_r2"],
            "controllable_lift": result["controllable_lift"],
            "unexplained_lift": result["unexplained_lift"],
            "conditioning": result["controllability_conditioning"],
        }

    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != latents.shape[0]:
        raise ValueError("controllability labels must match the latent row count")
    cells = []
    for label in sorted(np.unique(labels).tolist()):
        mask = labels == label
        if int(mask.sum()) < max(int(min_cell_size), 2):
            continue
        cell = _fit_slice(mask)
        cell["label"] = str(label)
        cells.append(cell)
    result["conditioning"] = "provided_labels"
    result["min_cell_size"] = int(min_cell_size)
    result["cells"] = cells
    result["num_cells"] = len(cells)
    if cells:
        result["cell_summary"] = {
            "r2_min": float(min(float(cell["action_to_dynamics_r2"]) for cell in cells)),
            "r2_max": float(max(float(cell["action_to_dynamics_r2"]) for cell in cells)),
            "r2_mean": float(np.mean([float(cell["action_to_dynamics_r2"]) for cell in cells])),
            "high_variance_low_controllability_cells": int(
                sum(
                    float(cell["dynamics_minus_controllable_lift"].get(str(max(ks)), 0.0)) > 0.0
                    and float(cell["action_to_dynamics_r2"]) < float(result["action_to_dynamics_r2"])
                    for cell in cells
                )
            ),
        }
    return result


def observability_probe_report(
    latents: np.ndarray,
    arrays: dict[str, np.ndarray],
    ks: Iterable[int],
) -> dict[str, object]:
    physical = {key: _safe_2d(value, latents.shape[0]) for key, value in arrays.items() if key.startswith("physical_")}
    if not physical:
        return {}
    eig = compute_eigenspace(latents)
    eigvecs = to_numpy(eig.eigenvectors)
    out = {}
    centered_targets = {name: value - value.mean(axis=0, keepdims=True) for name, value in physical.items()}
    for name, target in centered_targets.items():
        target_ss = float(np.square(target).sum())
        out[name] = {}
        for k in ks:
            basis = eigvecs[:, : min(int(k), eigvecs.shape[1])]
            features = latents @ basis
            coef, *_ = np.linalg.lstsq(features, target, rcond=None)
            pred = features @ coef
            r2 = 1.0 - float(np.square(target - pred).sum()) / max(target_ss, EPS)
            out[name][str(k)] = float(r2)
    return out


def eigenspace_transport_report(
    latents: np.ndarray,
    labels: np.ndarray,
    k: int,
    min_cell_size: int,
) -> dict[str, object]:
    bases = []
    cell_ids = []
    for label in sorted(np.unique(labels).tolist()):
        mask = labels == label
        if int(mask.sum()) < max(min_cell_size, 2):
            continue
        eig = compute_eigenspace(latents[mask])
        bases.append(to_numpy(eig.eigenvectors[:, : min(k, eig.eigenvectors.shape[1])]))
        cell_ids.append(str(label))
    transitions = {}
    for idx in range(len(bases) - 1):
        left = bases[idx]
        right = bases[idx + 1]
        kk = min(left.shape[1], right.shape[1])
        overlap = np.square(left[:, :kk].T @ right[:, :kk]).sum() / max(kk, 1)
        transitions[f"{cell_ids[idx]}->{cell_ids[idx + 1]}"] = float(overlap)
    return {"k": int(k), "transitions": transitions}


def uncertainty_residual_report(
    latents: np.ndarray,
    tangents: dict[str, np.ndarray],
    ks: Iterable[int],
) -> dict[str, object]:
    residuals = {name: value for name, value in tangents.items() if name.startswith("prediction_residual")}
    if not residuals:
        return {}
    eig = compute_eigenspace(latents)
    eigvecs = to_numpy(eig.eigenvectors)
    bands = rank_fraction_bands(latents.shape[1])
    out = {}
    for name, residual in residuals.items():
        norms = np.linalg.norm(residual, axis=1)
        out[name] = {
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "norm_p90": float(np.quantile(norms, 0.9)),
            "lift": lift_curve(eigvecs, residual, ks),
            "residual_band_fingerprint": rank_wise_transfer_fingerprint(eigvecs, residual, bands),
        }
    return out


def spectral_planner_weights(
    eigenvectors: np.ndarray,
    reachability_tangent: np.ndarray,
    controllability_tangent: np.ndarray,
    nuisance_tangent: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    reach = rank_energy_vector(eigenvectors, reachability_tangent)
    ctrl = rank_energy_vector(eigenvectors, controllability_tangent)
    if nuisance_tangent is None:
        nuisance = np.zeros_like(reach)
    else:
        nuisance = rank_energy_vector(eigenvectors, nuisance_tangent)
    weights = (reach + ctrl + float(eps)) / (nuisance + float(eps))
    return weights / max(float(weights.mean()), EPS)


def spectral_planner_distance(
    diff: np.ndarray,
    eigenvectors: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    delta = _as_2d(diff)
    basis = _as_2d(eigenvectors)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if delta.shape[1] != basis.shape[0]:
        raise ValueError(f"diff dim {delta.shape[1]} does not match basis dim {basis.shape[0]}")
    w = w[: basis.shape[1]]
    proj = delta @ basis[:, : w.shape[0]]
    return np.square(proj) @ w


def spectral_planner_metric_report(
    latents: np.ndarray,
    tangents: dict[str, np.ndarray],
) -> dict[str, object]:
    if "planning" not in tangents and "dynamics" not in tangents:
        return {}
    eig = compute_eigenspace(latents)
    eigvecs = to_numpy(eig.eigenvectors)
    reach = tangents.get("planning", tangents.get("dynamics"))
    ctrl = tangents.get("dynamics", reach)
    nuis = tangents.get("nuisance")
    weights = spectral_planner_weights(eigvecs, reach, ctrl, nuis)
    return {
        "formula": "d_spec(z,g)=(z-g)^T U diag(w) U^T (z-g)",
        "weight_rule": "w_j proportional to (reachability_j + controllability_j) / (nuisance_j + eps)",
        "weights": weights.tolist(),
        "weight_summary": {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
            "top_weight_mean": float(weights[: max(1, weights.size // 8)].mean()),
            "tail_weight_mean": float(weights[max(1, weights.size // 2) :].mean()),
        },
    }


def spectral_exploration_report(
    latents: np.ndarray,
    tangents: dict[str, np.ndarray],
    k: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> dict[str, object]:
    if "dynamics" not in tangents:
        return {}
    eig = compute_eigenspace(latents)
    eigvecs = to_numpy(eig.eigenvectors)
    k = min(int(k), eigvecs.shape[1])
    basis = eigvecs[:, :k]
    dyn = tangents["dynamics"]
    dyn_total = np.square(dyn).sum(axis=1)
    dyn_top = np.square(dyn @ basis).sum(axis=1)
    sci = dyn_top / np.maximum(dyn_total, EPS)
    residual = tangents.get("prediction_residual")
    unc_plan = np.zeros(dyn.shape[0], dtype=np.float32)
    if residual is not None:
        unc_plan = np.square(residual @ basis).sum(axis=1)
        unc_plan = unc_plan / max(float(np.quantile(unc_plan, 0.9)), EPS)
    planning_direction = tangents.get("planning", dyn).mean(axis=0)
    if np.linalg.norm(planning_direction) > EPS:
        ppte_row = np.square(dyn @ _unit_direction(planning_direction)) / np.maximum(dyn_total, EPS)
    else:
        ppte_row = np.zeros_like(sci)
    nuisance = tangents.get("nuisance")
    ncr_row = np.zeros_like(sci)
    if nuisance is not None:
        nuis_total = np.square(nuisance).sum(axis=1)
        ncr_row = np.square(nuisance @ basis).sum(axis=1) / np.maximum(nuis_total, EPS)
    score = alpha * unc_plan + beta * (1.0 - sci) + gamma * (1.0 - ppte_row) + delta * ncr_row
    return {
        "formula": "S=alpha*Unc_plan + beta*(1-SCI) + gamma*(1-PPTE) + delta*NCR",
        "k": int(k),
        "score_mean": float(score.mean()),
        "score_p90": float(np.quantile(score, 0.9)),
        "components": {
            "unc_plan_mean": float(unc_plan.mean()),
            "sci_mean": float(sci.mean()),
            "ppte_mean": float(ppte_row.mean()),
            "ncr_mean": float(ncr_row.mean()),
        },
    }


def spectral_shift_report(source_latents: np.ndarray, target_latents: np.ndarray, k: int = 8) -> dict[str, object]:
    source = compute_eigenspace(source_latents)
    target = compute_eigenspace(target_latents)
    src_vals = to_numpy(source.eigenvalues)
    tgt_vals = to_numpy(target.eigenvalues)
    src_profile = src_vals / max(float(src_vals.sum()), EPS)
    tgt_profile = tgt_vals / max(float(tgt_vals.sum()), EPS)
    dim = min(src_profile.size, tgt_profile.size)
    kk = min(int(k), to_numpy(source.eigenvectors).shape[1], to_numpy(target.eigenvectors).shape[1])
    overlap = np.square(to_numpy(source.eigenvectors[:, :kk]).T @ to_numpy(target.eigenvectors[:, :kk])).sum() / max(kk, 1)
    return {
        "spectrum_l1": float(np.abs(src_profile[:dim] - tgt_profile[:dim]).sum()),
        "topk_subspace_overlap": float(overlap),
        "source_stable_rank": float(src_vals.sum() / max(float(src_vals[0]), EPS)),
        "target_stable_rank": float(tgt_vals.sum() / max(float(tgt_vals[0]), EPS)),
    }


def audit_warnings(arrays: dict[str, np.ndarray], tangents: dict[str, np.ndarray]) -> list[str]:
    warnings = []
    optional_groups = {
        "goal-conditioned audit disabled; missing goals": ["goals"],
        "success/failure PPTE margin disabled; missing success": ["success"],
        "episode-conditioned audit disabled; missing episode_id": ["episode_id"],
        "phase-conditioned audit disabled; missing timestep": ["timestep"],
        "observability probes disabled; missing physical_* arrays": [],
        "counterfactual impossible-transition control disabled; missing counterfactual_* arrays": [],
        "multi-horizon audit limited to h=1; missing dynamics_h* arrays": [],
    }
    for message, keys in optional_groups.items():
        if keys and not all(key in arrays for key in keys):
            warnings.append(message)
    if not any(key.startswith("physical_") for key in arrays):
        warnings.append("observability probes disabled; missing physical_* arrays")
    if not any(key.startswith("counterfactual_") for key in tangents):
        warnings.append("counterfactual impossible-transition control disabled; missing counterfactual_* arrays")
    if not any(key.startswith("dynamics_h") for key in tangents):
        warnings.append("multi-horizon audit limited to h=1; missing dynamics_h* arrays")
    return warnings


def run_intervention_audit(
    arrays: dict[str, np.ndarray],
    config: EmpiricalAuditConfig,
) -> dict[str, object]:
    latents = _as_2d(arrays["latents"])
    tangents = build_tangent_families(arrays, config.horizons)
    if not tangents:
        return {}
    eig = compute_eigenspace(latents)
    eigvecs = to_numpy(eig.eigenvectors)
    k = min(config.ks[0], eigvecs.shape[1]) if config.ks else min(8, eigvecs.shape[1])
    basis = eigvecs[:, :k]
    out: dict[str, object] = {"top_k": int(k), "band_dropout": {}, "band_noise": {}, "factor_suppression": {}}
    rng = np.random.default_rng(config.seed)
    for name, tangent in tangents.items():
        projected = (tangent @ basis) @ basis.T
        dropped = tangent - projected
        out["band_dropout"][name] = {
            "retention": float(np.square(dropped).sum(axis=1).mean() / max(np.square(tangent).sum(axis=1).mean(), EPS))
        }
        noise = rng.normal(size=(tangent.shape[0], k)).astype(np.float32)
        noise = noise * np.sqrt(max(np.square(tangent @ basis).mean(), EPS))
        noised = tangent + noise @ basis.T
        out["band_noise"][name] = {
            "energy_ratio": float(np.square(noised).sum(axis=1).mean() / max(np.square(tangent).sum(axis=1).mean(), EPS))
        }
    target = "nuisance" if "nuisance" in tangents else next(iter(tangents))
    adapter = feature_space_adapter(tangents[target], strength=1.0)
    out["factor_suppression"]["target"] = target
    out["factor_suppression"]["retention"] = {
        name: float(
            np.square(to_numpy(apply_feature_adapter(adapter, tangent))).sum(axis=1).mean()
            / max(np.square(tangent).sum(axis=1).mean(), EPS)
        )
        for name, tangent in tangents.items()
    }
    if {"planning", "nuisance"}.issubset(tangents):
        amplified = tangents["planning"] * 2.0
        rotated = tangents["planning"].copy()
        rotated[:, :k] = tangents["nuisance"][:, :k]
        out["factor_amplification"] = {
            "planning_energy_gain": float(
                np.square(amplified).sum(axis=1).mean() / max(np.square(tangents["planning"]).sum(axis=1).mean(), EPS)
            )
        }
        out["factor_rotation"] = {
            "planning_to_nuisance_cosine": tangent_cosine(rotated, tangents["nuisance"]),
        }
    return out


def run_empirical_audit(
    arrays: dict[str, np.ndarray],
    config: EmpiricalAuditConfig,
) -> dict[str, object]:
    latents = _as_2d(arrays["latents"])
    tangents = build_tangent_families(arrays, config.horizons)
    if not tangents:
        raise ValueError("arrays must contain at least one tangent family")
    labels = build_condition_labels(arrays, config)
    base = summarize_conditioned_latent_audit(
        latents,
        tangents,
        labels,
        config.ks,
        min_cell_size=config.min_cell_size,
        band_mode=config.band_mode,
        seed=config.seed,
    )
    state_source = _safe_2d(arrays.get("states", latents), latents.shape[0])
    state_labels = cluster_states(state_source, config.clusters, seed=config.seed)
    controls = {
        "state_cluster_shuffle": _summarize_control(
            latents, tangents, _permuted_labels(state_labels, config.seed + 1), config, "state_cluster_shuffle"
        ),
        "time_shuffle": _summarize_control(
            latents, _permuted_tangents(tangents, config.seed + 2), labels, config, "time_shuffle"
        ),
    }
    if "actions" in arrays:
        action_labels = cluster_states(_safe_2d(arrays["actions"], latents.shape[0]), config.clusters, seed=config.seed)
        controls["action_shuffle"] = _summarize_control(
            latents,
            tangents,
            _permuted_labels(action_labels, config.seed + 3),
            config,
            "action_shuffle",
        )
    if any(name.startswith("counterfactual_") for name in tangents):
        controls["counterfactual_impossible_transition"] = {
            name: lift_curve(to_numpy(compute_eigenspace(latents).eigenvectors), tangent, config.ks)
            for name, tangent in tangents.items()
            if name.startswith("counterfactual_")
        }
    report = {
        **base,
        "schema": {
            "required": ["latents"],
            "available": sorted(arrays),
            "available_tangents": sorted(tangents),
            "optional_prefixes": ["dynamics_h", "predicted_dynamics_h", "physical_", "counterfactual_"],
        },
        "condition": config.condition,
        "config": {**base["config"], **asdict(config), "ks": list(config.ks), "horizons": list(config.horizons)},
        "label_controls": controls,
        "multi_horizon": multi_horizon_report(latents, tangents, config.ks, seed=config.seed),
        "controllability": controllability_spectrum_report(
            latents,
            tangents,
            _safe_2d(arrays["actions"], latents.shape[0]) if "actions" in arrays else None,
            config.ks,
            labels=labels,
            min_cell_size=config.min_cell_size,
        ),
        "observability": observability_probe_report(latents, arrays, config.ks),
        "eigenspace_transport": eigenspace_transport_report(
            latents, labels, k=min(config.ks[0], latents.shape[1]), min_cell_size=config.min_cell_size
        ),
        "uncertainty_residual": uncertainty_residual_report(latents, tangents, config.ks),
        "spectral_planner_metric": spectral_planner_metric_report(latents, tangents),
        "spectral_exploration": spectral_exploration_report(latents, tangents, k=min(config.ks[0], latents.shape[1])),
        "warnings": audit_warnings(arrays, tangents),
    }
    if "ood_latents" in arrays:
        report["ood_spectral_shift"] = spectral_shift_report(latents, _safe_2d(arrays["ood_latents"]), k=min(config.ks[0], latents.shape[1]))
    if config.include_interventions:
        report["planner_interventions"] = run_intervention_audit(arrays, config)
    return report


def empirical_spectral_report(
    arrays: dict[str, np.ndarray],
    condition: str = "states",
    clusters: int = 32,
    ks: Iterable[int] = (4, 8, 16, 32, 64),
    min_cell_size: int = 256,
    band_mode: str = "rank",
    seed: int = 0,
) -> dict[str, object]:
    """Build an empirical report from a loaded latents.npz dictionary."""

    config = EmpiricalAuditConfig(
        condition=condition,
        clusters=clusters,
        ks=tuple(int(k) for k in ks),
        min_cell_size=min_cell_size,
        band_mode=band_mode,
        seed=seed,
        include_interventions=False,
    )
    return run_empirical_audit(arrays, config)
