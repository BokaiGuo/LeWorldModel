"""Theory-side spectral-transfer operators.

This package is intentionally separate from the empirical LeWM audit scripts.
It implements the analytical objects from the theory plan: nuisance/source
geometry, encoder transfer response, predictor transfer response, heat-filter
intervention, slope diagnostics, and residual-band penalties.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class TransferConfig:
    dim: int = 64
    intrinsic_dim: float = 2.0
    input_exponent: float = 1.0
    encoder_exponent: float = 1.0
    tail_encoder_exponent: float | None = None
    breakpoint: int | None = None
    predictor_exponent: float = 1.0
    heat_time: float = 0.0
    heat_exponent: float = 1.0


def _ranks(dim: int) -> np.ndarray:
    if dim <= 0:
        raise ValueError("dim must be positive")
    return np.arange(1, int(dim) + 1, dtype=np.float64)


def _band_edges(dim: int) -> list[tuple[int, int]]:
    cuts = [0, max(1, int(np.ceil(0.05 * dim))), max(2, int(np.ceil(0.2 * dim))), max(3, int(np.ceil(0.5 * dim))), dim]
    edges = sorted(set(min(dim, max(0, cut)) for cut in cuts))
    return [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo]


def band_energy_summary(eigenvalues: np.ndarray, bands: Iterable[tuple[int, int]] | None = None) -> dict[str, float]:
    vals = np.asarray(eigenvalues, dtype=np.float64).reshape(-1).clip(min=0)
    total = float(vals.sum())
    if total <= EPS:
        return {}
    band_list = list(bands) if bands is not None else _band_edges(vals.size)
    return {
        f"{max(0, int(lo))}:{min(vals.size, int(hi))}": float(vals[max(0, int(lo)) : min(vals.size, int(hi))].sum() / total)
        for lo, hi in band_list
        if min(vals.size, int(hi)) > max(0, int(lo))
    }


def power_law_spectrum(
    dim: int,
    exponent: float,
    intrinsic_dim: float = 1.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Return lambda_j = scale * j^{-exponent / intrinsic_dim}."""

    if intrinsic_dim <= 0:
        raise ValueError("intrinsic_dim must be positive")
    return float(scale) * _ranks(dim) ** (-float(exponent) / float(intrinsic_dim))


def broken_power_law_response(
    dim: int,
    head_exponent: float,
    tail_exponent: float,
    breakpoint: int,
    intrinsic_dim: float = 1.0,
) -> np.ndarray:
    """Continuous two-slope response used for encoder cutoff studies."""

    if not 1 <= int(breakpoint) <= int(dim):
        raise ValueError("breakpoint must be in [1, dim]")
    ranks = _ranks(dim)
    breakpoint = int(breakpoint)
    head = ranks ** (-float(head_exponent) / float(intrinsic_dim))
    out = head.copy()
    if breakpoint < dim:
        anchor = head[breakpoint - 1]
        out[breakpoint:] = anchor * (ranks[breakpoint:] / breakpoint) ** (
            -float(tail_exponent) / float(intrinsic_dim)
        )
    return out


def heat_filter(
    eigenvalues: np.ndarray,
    rates: np.ndarray,
    time: float,
) -> np.ndarray:
    """Apply lambda_j(t)=lambda_j(0) exp(-2 t nu_j)."""

    vals = np.asarray(eigenvalues, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    if vals.shape != rates.shape:
        raise ValueError(f"eigenvalues and rates must have the same shape, got {vals.shape} and {rates.shape}")
    return vals * np.exp(-2.0 * float(time) * rates)


def diagonal_transfer_spectra(config: TransferConfig) -> dict[str, np.ndarray]:
    """Build a diagonal spectral-transfer system.

    In the commuting/oracle case,
        Sigma_z = C_theta^{1/2} G_E C_theta^{1/2},
    so its eigenvalues are input_variance * encoder_metric.  The predictor is
    represented as a second transfer filter on latent dynamics; residual energy
    is the part not passed by that predictor response.
    """

    dim = int(config.dim)
    input_variance = power_law_spectrum(dim, config.input_exponent, config.intrinsic_dim)
    if config.tail_encoder_exponent is not None or config.breakpoint is not None:
        if config.tail_encoder_exponent is None or config.breakpoint is None:
            raise ValueError("tail_encoder_exponent and breakpoint must be set together")
        encoder_metric = broken_power_law_response(
            dim,
            config.encoder_exponent,
            config.tail_encoder_exponent,
            config.breakpoint,
            config.intrinsic_dim,
        )
    else:
        encoder_metric = power_law_spectrum(dim, config.encoder_exponent, config.intrinsic_dim)

    latent = input_variance * encoder_metric
    rates = _ranks(dim) ** float(config.heat_exponent)
    filtered_latent = heat_filter(latent, rates, config.heat_time) if config.heat_time else latent.copy()

    raw_predictor = power_law_spectrum(dim, config.predictor_exponent, config.intrinsic_dim)
    predictor_response = raw_predictor / max(float(raw_predictor.max()), EPS)
    predicted = filtered_latent * predictor_response
    residual = filtered_latent * np.square(1.0 - predictor_response)
    return {
        "input_variance": input_variance,
        "encoder_metric": encoder_metric,
        "latent_eigenvalues": latent,
        "heat_rates": rates,
        "filtered_latent_eigenvalues": filtered_latent,
        "predictor_response": predictor_response,
        "predicted_eigenvalues": predicted,
        "residual_eigenvalues": residual,
    }


def _fit_loglog_curve(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float]:
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"x and y must have the same length, got {x.size} and {y.size}")
    mask = (x > EPS) & (y > EPS)
    if mask.sum() < 2:
        return {"alpha": float("nan"), "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    x = np.log(x[mask])
    y = np.log(y[mask])
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / max(ss_tot, EPS)
    return {
        "alpha": float(-slope),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def fit_loglog_slope(
    eigenvalues: np.ndarray,
    start_rank: int = 1,
    stop_rank: int | None = None,
) -> dict[str, float]:
    """Fit log(lambda_j) = b - alpha log(j) over an inclusive rank window."""

    vals = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        raise ValueError("fit_loglog_slope requires at least one eigenvalue")
    start = max(1, int(start_rank))
    stop = vals.size if stop_rank is None else min(int(stop_rank), vals.size)
    if stop < start:
        raise ValueError("stop_rank must be >= start_rank")
    window = vals[start - 1 : stop]
    mask = window > EPS
    if mask.sum() < 2:
        return {"alpha": float("nan"), "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    ranks = np.arange(start, stop + 1, dtype=np.float64)[mask]
    y = np.log(window[mask])
    x = np.log(ranks)
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - ss_res / max(ss_tot, EPS)
    return {
        "alpha": float(-slope),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def multi_horizon_transfer_report(
    config: TransferConfig,
    horizons: Iterable[int] = (1, 2, 4, 8),
) -> dict[str, object]:
    """Compose the one-step predictor transfer across multiple horizons."""

    spectra = diagonal_transfer_spectra(config)
    latent = np.asarray(spectra["filtered_latent_eigenvalues"], dtype=np.float64)
    predictor = np.asarray(spectra["predictor_response"], dtype=np.float64)
    horizon_list = []
    for horizon in horizons:
        h = int(horizon)
        if h < 1:
            raise ValueError("horizons must be positive integers")
        if h not in horizon_list:
            horizon_list.append(h)
    if not horizon_list:
        raise ValueError("at least one horizon is required")

    reports: dict[str, object] = {}
    predicted_traces: dict[str, float] = {}
    residual_traces: dict[str, float] = {}
    predicted_fits = []
    residual_fits = []
    latent_trace = float(latent.sum())

    for h in horizon_list:
        horizon_response = predictor ** h
        predicted = latent * horizon_response
        residual = latent * np.square(1.0 - horizon_response)
        predicted_trace = float(predicted.sum())
        residual_trace = float(residual.sum())
        predicted_traces[str(h)] = predicted_trace
        residual_traces[str(h)] = residual_trace
        predicted_fits.append(predicted_trace)
        residual_fits.append(residual_trace)
        reports[str(h)] = {
            "horizon": int(h),
            "predictor_response": horizon_response.tolist(),
            "predicted_eigenvalues": predicted.tolist(),
            "residual_eigenvalues": residual.tolist(),
            "spectrum_stats": {
                "predicted_eigenvalues": _compact_spectrum_stats(predicted),
                "residual_eigenvalues": _compact_spectrum_stats(residual),
            },
            "band_energy": {
                "predicted_eigenvalues": band_energy_summary(predicted),
                "residual_eigenvalues": band_energy_summary(residual),
            },
            "trace_fraction": {
                "predicted": float(predicted_trace / max(latent_trace, EPS)),
                "residual": float(residual_trace / max(latent_trace, EPS)),
            },
        }

    return {
        "horizons": horizon_list,
        "latent_trace": latent_trace,
        "predictor_response_stats": _compact_spectrum_stats(predictor),
        "reports": reports,
        "growth": {
            "predicted_trace": predicted_traces,
            "residual_trace": residual_traces,
            "predicted_trace_fit": _fit_loglog_curve(np.asarray(horizon_list, dtype=np.float64), np.asarray(predicted_fits, dtype=np.float64)),
            "residual_trace_fit": _fit_loglog_curve(np.asarray(horizon_list, dtype=np.float64), np.asarray(residual_fits, dtype=np.float64)),
            "predicted_monotone_nonincreasing": bool(all(a >= b - EPS for a, b in zip(predicted_fits, predicted_fits[1:]))),
            "residual_monotone_nondecreasing": bool(all(a <= b + EPS for a, b in zip(residual_fits, residual_fits[1:]))),
        },
    }


def residual_band_penalty(
    residuals: np.ndarray,
    eigenvectors: np.ndarray,
    bands: Iterable[tuple[int, int]],
    weights: Iterable[float] | None = None,
) -> float:
    """Weighted residual energy in spectral bands for residual-band losses."""

    err = np.asarray(residuals, dtype=np.float64)
    basis = np.asarray(eigenvectors, dtype=np.float64)
    if err.ndim != 2 or basis.ndim != 2:
        raise ValueError("residuals and eigenvectors must be 2D")
    if err.shape[1] != basis.shape[0]:
        raise ValueError(f"residual dim {err.shape[1]} does not match basis dim {basis.shape[0]}")
    band_list = [(max(0, int(lo)), min(int(hi), basis.shape[1])) for lo, hi in bands]
    weight_list = list(weights) if weights is not None else [1.0] * len(band_list)
    if len(weight_list) != len(band_list):
        raise ValueError("weights must have the same length as bands")
    total = 0.0
    for (lo, hi), weight in zip(band_list, weight_list):
        if hi <= lo:
            continue
        proj = err @ basis[:, lo:hi]
        total += float(weight) * float(np.square(proj).sum(axis=1).mean())
    return total


def _orthonormal(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q


def _oblique_frame(dim: int, obliqueness: float, seed: int) -> np.ndarray:
    q = _orthonormal(dim, seed)
    rng = np.random.default_rng(seed + 1)
    shear = np.triu(rng.normal(size=(dim, dim)), k=1)
    frame = q @ (np.eye(dim) + float(obliqueness) * shear / max(np.sqrt(dim), 1.0))
    norms = np.linalg.norm(frame, axis=0, keepdims=True)
    return frame / np.maximum(norms, EPS)


def oblique_frame_stress_report(
    config: TransferConfig,
    obliqueness: float = 0.35,
    seed: int = 0,
    slope_windows: Iterable[tuple[int, int | None]] = ((1, None), (1, 16), (32, 64)),
) -> dict[str, object]:
    """Stress finite-window slopes when nuisance factors are not orthogonal."""

    spectra = diagonal_transfer_spectra(config)
    oracle = spectra["latent_eigenvalues"]
    frame = _oblique_frame(config.dim, obliqueness, seed)
    cov = frame @ np.diag(oracle) @ frame.T
    observed = np.linalg.eigvalsh(cov)[::-1].clip(min=0)
    windows = []
    for start, stop in slope_windows:
        if start > config.dim:
            continue
        safe_stop = None if stop is None else min(stop, config.dim)
        if safe_stop is not None and safe_stop < start:
            continue
        windows.append((start, safe_stop))
    if not windows:
        windows = [(1, None)]
    oracle_slopes = {f"{s}:{e or 'end'}": fit_loglog_slope(oracle, s, e) for s, e in windows}
    observed_slopes = {f"{s}:{e or 'end'}": fit_loglog_slope(observed, s, e) for s, e in windows}
    return {
        "obliqueness": float(obliqueness),
        "seed": int(seed),
        "frame_condition_number": float(np.linalg.cond(frame)),
        "oracle_slopes": oracle_slopes,
        "observed_slopes": observed_slopes,
        "finite_window_alpha_bias": {
            key: float(observed_slopes[key]["alpha"] - oracle_slopes[key]["alpha"]) for key in observed_slopes
        },
        "observed_spectrum_stats": _compact_spectrum_stats(observed),
    }


def heat_filter_recovery_report(
    config: TransferConfig,
    times: Iterable[float] = (0.01, 0.03, 0.05),
    noise_std: float = 0.0,
    seed: int = 0,
) -> dict[str, object]:
    """Recover generator rates from known heat-filtered spectra."""

    spectra = diagonal_transfer_spectra(config)
    base = spectra["latent_eigenvalues"].clip(min=EPS)
    rates = spectra["heat_rates"]
    rng = np.random.default_rng(seed)
    runs = {}
    r2s = []
    for time in times:
        time = float(time)
        filtered = heat_filter(base, rates, time).clip(min=EPS)
        if noise_std:
            filtered = filtered * np.exp(rng.normal(scale=float(noise_std), size=filtered.shape))
        y = np.log(filtered / base)
        x = -2.0 * time * rates
        slope, intercept = np.polyfit(x, y, deg=1)
        pred = slope * x + intercept
        ss_res = float(np.square(y - pred).sum())
        ss_tot = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - ss_res / max(ss_tot, EPS)
        r2s.append(r2)
        runs[str(time)] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": float(r2),
        }
    return {
        "noise_std": float(noise_std),
        "seed": int(seed),
        "runs": runs,
        "median_r2": float(np.median(r2s)) if r2s else float("nan"),
    }


def sigreg_local_anisotropy_report(dim: int = 32, cells: int = 8, anisotropy: float = 12.0, seed: int = 0) -> dict[str, object]:
    """Toy check: globally Gaussian-like covariance can hide local anisotropy."""

    local_covs = []
    for idx in range(int(cells)):
        q = _orthonormal(dim, seed + idx)
        vals = np.ones(dim)
        vals[0] = float(anisotropy)
        vals[1 : max(2, dim // 8)] = np.sqrt(float(anisotropy))
        local_covs.append(q @ np.diag(vals) @ q.T)
    global_cov = sum(local_covs) / len(local_covs)
    global_vals = np.linalg.eigvalsh(global_cov)[::-1].clip(min=0)
    local_vals = [np.linalg.eigvalsh(cov)[::-1].clip(min=0) for cov in local_covs]
    return {
        "dim": int(dim),
        "cells": int(cells),
        "anisotropy": float(anisotropy),
        "global_stats": _compact_spectrum_stats(global_vals),
        "mean_local_stats": {
            key: float(np.mean([_compact_spectrum_stats(vals)[key] for vals in local_vals]))
            for key in ["stable_rank", "participation_ratio"]
        },
        "global_over_local_stable_rank": float(
            _compact_spectrum_stats(global_vals)["stable_rank"]
            / max(np.mean([_compact_spectrum_stats(vals)["stable_rank"] for vals in local_vals]), EPS)
        ),
    }


def identifiability_controllability_report(dim: int = 32) -> dict[str, object]:
    """Toy cases showing high variance need not equal controllability alignment."""

    vals = power_law_spectrum(dim, exponent=1.5, intrinsic_dim=1.0)
    basis = np.eye(dim)
    aligned = np.zeros((dim, dim))
    aligned[:, 0] = 1.0
    mismatch = np.zeros((dim, dim))
    mismatch[:, -1] = 1.0
    top_k = max(1, dim // 8)
    return {
        "dim": int(dim),
        "top_k": int(top_k),
        "spectrum_stats": _compact_spectrum_stats(vals),
        "aligned_controllability_lift": float(
            np.square(aligned @ basis[:, :top_k]).sum(axis=1).mean()
            / max((top_k / dim) * np.square(aligned).sum(axis=1).mean(), EPS)
        ),
        "mismatched_controllability_lift": float(
            np.square(mismatch @ basis[:, :top_k]).sum(axis=1).mean()
            / max((top_k / dim) * np.square(mismatch).sum(axis=1).mean(), EPS)
        ),
    }


def _plane_rotation(dim: int, angle_degrees: float, i: int = 0, j: int | None = None) -> np.ndarray:
    if dim < 2:
        raise ValueError("dim must be at least 2")
    if j is None:
        j = dim - 1
    if not 0 <= int(i) < dim or not 0 <= int(j) < dim or int(i) == int(j):
        raise ValueError("rotation plane must use two distinct valid coordinates")
    theta = np.deg2rad(float(angle_degrees))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    rot = np.eye(dim, dtype=np.float64)
    rot[int(i), int(i)] = c
    rot[int(j), int(j)] = c
    rot[int(i), int(j)] = -s
    rot[int(j), int(i)] = s
    return rot


def _covariance_band_lift(cov: np.ndarray, k: int) -> float:
    matrix = np.asarray(cov, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    dim = matrix.shape[0]
    k = min(max(1, int(k)), dim)
    total = float(np.trace(matrix))
    band = float(np.trace(matrix[:k, :k]))
    return float((band / max(total, EPS)) / (k / dim))


def _matrix_trace_ratio(cov: np.ndarray, total: float) -> float:
    return float(np.trace(np.asarray(cov, dtype=np.float64)) / max(total, EPS))


def _plane_rotation_commutator_norm_exact(
    latent_eigenvalues: np.ndarray,
    predictor_eigenvalues: np.ndarray,
    angle_degrees: float,
    i: int = 0,
    j: int | None = None,
) -> float:
    latent = np.asarray(latent_eigenvalues, dtype=np.float64).reshape(-1)
    predictor = np.asarray(predictor_eigenvalues, dtype=np.float64).reshape(-1)
    if latent.shape != predictor.shape:
        raise ValueError("latent and predictor spectra must have the same shape")
    dim = latent.size
    if dim < 2:
        raise ValueError("spectra must have at least two dimensions")
    if j is None:
        j = dim - 1
    if not 0 <= int(i) < dim or not 0 <= int(j) < dim or int(i) == int(j):
        raise ValueError("rotation plane must use two distinct valid coordinates")
    theta = np.deg2rad(float(angle_degrees))
    latent_gap = float(abs(latent[int(i)] - latent[int(j)]))
    predictor_gap = float(abs(predictor[int(i)] - predictor[int(j)]))
    if latent_gap <= EPS or predictor_gap <= EPS:
        return 0.0
    exact = latent_gap * predictor_gap * abs(np.sin(2.0 * theta)) / np.sqrt(2.0)
    return float(exact / max(np.linalg.norm(latent) * np.linalg.norm(predictor), EPS))


def operator_alignment_theorem_report(
    config: TransferConfig | None = None,
    angles: Iterable[float] = (0, 15, 30, 45, 60, 75, 90),
    seed: int = 0,
) -> dict[str, object]:
    """Analytic theorem candidate for same-spectrum operator alignment.

    For a single-plane rotation, the commutator norm is an explicit sine law
    controlled by the eigenvalue gaps in the rotated plane. This is the law
    card that the sweep numerically verifies.
    """

    cfg = config or TransferConfig(dim=64, input_exponent=1.0, encoder_exponent=1.0, predictor_exponent=1.0)
    spectra = diagonal_transfer_spectra(cfg)
    latent = np.asarray(spectra["filtered_latent_eigenvalues"], dtype=np.float64)
    predictor_response = np.asarray(spectra["predictor_response"], dtype=np.float64)
    angle_values = sorted({float(value) for value in angles})
    if len(angle_values) == 0:
        raise ValueError("angles must contain at least one value")
    exact = {
        str(angle): _plane_rotation_commutator_norm_exact(latent, predictor_response, angle)
        for angle in angle_values
    }
    flat_latent = np.full_like(latent, float(latent.mean()))
    flat_predictor = np.full_like(predictor_response, float(predictor_response.mean()))
    flat_latent_exact = {
        str(angle): _plane_rotation_commutator_norm_exact(flat_latent, predictor_response, angle)
        for angle in angle_values
    }
    flat_predictor_exact = {
        str(angle): _plane_rotation_commutator_norm_exact(latent, flat_predictor, angle)
        for angle in angle_values
    }
    peak_angle = 45.0 if (
        abs(float(latent[0] - latent[-1])) > EPS and abs(float(predictor_response[0] - predictor_response[-1])) > EPS
    ) else float("nan")
    return {
        "config": asdict(cfg),
        "angles": angle_values,
        "rotation_plane": {"i": 0, "j": int(cfg.dim) - 1},
        "eigenvalue_gaps": {
            "latent": float(abs(latent[0] - latent[-1])),
            "predictor": float(abs(predictor_response[0] - predictor_response[-1])),
        },
        "theorem": {
            "statement": (
                "For a single-plane rotation of a diagonal predictor operator, the normalized commutator "
                "norm is proportional to |sin(2 theta)| and is maximized at an oblique angle."
            ),
            "peak_angle_degrees": peak_angle,
            "exact_by_angle": exact,
        },
        "falsifiers": {
            "flat_latent_spectrum": flat_latent_exact,
            "flat_predictor_spectrum": flat_predictor_exact,
        },
        "claim_boundary": (
            "This theorem is only about the fixed-spectrum, single-plane, finite-dimensional toy model; "
            "it does not by itself establish any learned LeWM property."
        ),
    }


def operator_alignment_report(
    config: TransferConfig | None = None,
    angles: Iterable[float] = (0, 15, 30, 45, 60, 75, 90),
    horizons: Iterable[int] = (1, 2, 4, 8),
    ks: Iterable[int] = (4, 8, 16, 32),
    random_trials: int = 128,
    seed: int = 0,
) -> dict[str, object]:
    """Hold the spectrum fixed while rotating the predictor basis.

    The predictor eigenvalues stay unchanged, but the relative eigenspace
    orientation against the latent covariance changes. This isolates an
    operator-alignment effect that eigenvalue spectra alone cannot identify.
    """

    cfg = config or TransferConfig(dim=64, input_exponent=1.0, encoder_exponent=1.0, predictor_exponent=1.0)
    dim = int(cfg.dim)
    if dim < 2:
        raise ValueError("config.dim must be at least 2")
    spectra = diagonal_transfer_spectra(cfg)
    latent = np.asarray(spectra["filtered_latent_eigenvalues"], dtype=np.float64)
    predictor_response = np.asarray(spectra["predictor_response"], dtype=np.float64)
    if latent.shape != predictor_response.shape:
        raise ValueError("latent and predictor spectra must have the same shape")

    latent_cov = np.diag(latent)
    predictor_cov = np.diag(predictor_response)
    latent_trace = float(np.trace(latent_cov))
    angle_values = sorted({float(value) for value in angles})
    horizon_list = sorted({int(h) for h in horizons})
    if not horizon_list or horizon_list[0] < 1:
        raise ValueError("horizons must contain positive integers")
    ks_list = sorted({min(max(1, int(k)), dim) for k in ks})
    if not ks_list:
        raise ValueError("ks must contain positive integers")
    rng = np.random.default_rng(seed)

    def _horizon_metrics(matrix: np.ndarray) -> dict[str, object]:
        metrics: dict[str, object] = {}
        eye = np.eye(dim, dtype=np.float64)
        for horizon in horizon_list:
            power = np.linalg.matrix_power(matrix, int(horizon))
            predicted_cov = power @ latent_cov @ power.T
            residual_cov = (eye - power) @ latent_cov @ (eye - power).T
            metrics[str(horizon)] = {
                "predicted_trace_ratio": _matrix_trace_ratio(predicted_cov, latent_trace),
                "residual_trace_ratio": _matrix_trace_ratio(residual_cov, latent_trace),
                "predicted_band_lift": {str(k): _covariance_band_lift(predicted_cov, k) for k in ks_list},
                "residual_band_lift": {str(k): _covariance_band_lift(residual_cov, k) for k in ks_list},
            }
        return metrics

    records = []
    for angle in angle_values:
        basis = _plane_rotation(dim, angle)
        predictor = basis @ predictor_cov @ basis.T
        commutator = latent_cov @ predictor - predictor @ latent_cov
        horizon_metrics = _horizon_metrics(predictor)
        min_h = str(horizon_list[0])
        max_h = str(horizon_list[-1])
        records.append(
            {
                "angle_degrees": float(angle),
                "commutator_norm": float(
                    np.linalg.norm(commutator, ord="fro")
                    / max(np.linalg.norm(latent_cov, ord="fro") * np.linalg.norm(predictor, ord="fro"), EPS)
                ),
                "horizons": horizon_metrics,
                "residual_growth_ratio": float(
                    horizon_metrics[max_h]["residual_trace_ratio"] / max(horizon_metrics[min_h]["residual_trace_ratio"], EPS)
                ),
                "predicted_decay_ratio": float(
                    horizon_metrics[max_h]["predicted_trace_ratio"] / max(horizon_metrics[min_h]["predicted_trace_ratio"], EPS)
                ),
            }
        )

    random_rows = []
    for trial in range(int(random_trials)):
        basis = _orthonormal(dim, seed + trial + 1)
        predictor = basis @ predictor_cov @ basis.T
        commutator = latent_cov @ predictor - predictor @ latent_cov
        horizon_metrics = _horizon_metrics(predictor)
        min_h = str(horizon_list[0])
        max_h = str(horizon_list[-1])
        random_rows.append(
            {
                "commutator_norm": float(
                    np.linalg.norm(commutator, ord="fro")
                    / max(np.linalg.norm(latent_cov, ord="fro") * np.linalg.norm(predictor, ord="fro"), EPS)
                ),
                "residual_growth_ratio": float(
                    horizon_metrics[max_h]["residual_trace_ratio"] / max(horizon_metrics[min_h]["residual_trace_ratio"], EPS)
                ),
                "predicted_decay_ratio": float(
                    horizon_metrics[max_h]["predicted_trace_ratio"] / max(horizon_metrics[min_h]["predicted_trace_ratio"], EPS)
                ),
                "horizons": horizon_metrics,
            }
        )

    def _summarize(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p05": float(np.quantile(arr, 0.05)),
            "p95": float(np.quantile(arr, 0.95)),
        }

    random_summary = {
        "commutator_norm": _summarize([row["commutator_norm"] for row in random_rows]),
        "residual_growth_ratio": _summarize([row["residual_growth_ratio"] for row in random_rows]),
        "predicted_decay_ratio": _summarize([row["predicted_decay_ratio"] for row in random_rows]),
    }

    aligned = records[0]
    peak_oblique = max(records, key=lambda row: row["commutator_norm"])
    endpoint = records[-1]
    min_h = str(horizon_list[0])
    max_h = str(horizon_list[-1])
    aligned_horizon = aligned["horizons"][max_h]
    peak_horizon = peak_oblique["horizons"][max_h]
    endpoint_horizon = endpoint["horizons"][max_h]
    top_k = str(ks_list[-1])
    return {
        "config": asdict(cfg),
        "dim": int(dim),
        "angles": angle_values,
        "horizons": horizon_list,
        "ks": ks_list,
        "records": records,
        "random_null": random_summary,
        "contrast": {
            "aligned_angle_degrees": float(aligned["angle_degrees"]),
            "peak_oblique_angle_degrees": float(peak_oblique["angle_degrees"]),
            "endpoint_angle_degrees": float(endpoint["angle_degrees"]),
            "aligned_commutator_norm": aligned["commutator_norm"],
            "peak_oblique_commutator_norm": peak_oblique["commutator_norm"],
            "endpoint_commutator_norm": endpoint["commutator_norm"],
            "aligned_residual_growth_ratio": aligned["residual_growth_ratio"],
            "peak_oblique_residual_growth_ratio": peak_oblique["residual_growth_ratio"],
            "endpoint_residual_growth_ratio": endpoint["residual_growth_ratio"],
            "aligned_predicted_decay_ratio": aligned["predicted_decay_ratio"],
            "peak_oblique_predicted_decay_ratio": peak_oblique["predicted_decay_ratio"],
            "endpoint_predicted_decay_ratio": endpoint["predicted_decay_ratio"],
            "aligned_horizon": {
                "predicted_trace_ratio": aligned_horizon["predicted_trace_ratio"],
                "residual_trace_ratio": aligned_horizon["residual_trace_ratio"],
                "predicted_band_lift": aligned_horizon["predicted_band_lift"][top_k],
                "residual_band_lift": aligned_horizon["residual_band_lift"][top_k],
            },
            "peak_oblique_horizon": {
                "predicted_trace_ratio": peak_horizon["predicted_trace_ratio"],
                "residual_trace_ratio": peak_horizon["residual_trace_ratio"],
                "predicted_band_lift": peak_horizon["predicted_band_lift"][top_k],
                "residual_band_lift": peak_horizon["residual_band_lift"][top_k],
            },
            "endpoint_horizon": {
                "predicted_trace_ratio": endpoint_horizon["predicted_trace_ratio"],
                "residual_trace_ratio": endpoint_horizon["residual_trace_ratio"],
                "predicted_band_lift": endpoint_horizon["predicted_band_lift"][top_k],
                "residual_band_lift": endpoint_horizon["residual_band_lift"][top_k],
            },
        },
        "claim_boundary": (
            "Synthetic same-spectrum / different-basis check. The result identifies operator alignment and "
            "noncommutativity, not control, as the boundary object that eigenvalue spectra alone miss."
        ),
    }


def _top_band_lift_numpy(tangents: np.ndarray, k: int) -> float:
    t = np.asarray(tangents, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError("tangents must be 2D")
    dim = t.shape[1]
    k = min(max(1, int(k)), dim)
    total = float(np.square(t).sum(axis=1).mean())
    top = float(np.square(t[:, :k]).sum(axis=1).mean())
    return float(top / max((k / dim) * total, EPS))


def _controllable_subspace(dim: int, controllable_dim: int, angle_degrees: float) -> np.ndarray:
    if dim < 2:
        raise ValueError("dim must be at least 2")
    sub_dim = min(max(1, int(controllable_dim)), dim // 2)
    theta = np.deg2rad(float(angle_degrees))
    basis = np.zeros((dim, sub_dim), dtype=np.float64)
    for idx in range(sub_dim):
        basis[idx, idx] = np.cos(theta)
        basis[dim - sub_dim + idx, idx] = np.sin(theta)
    return basis


def _random_subspace(dim: int, controllable_dim: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, controllable_dim)))
    return q[:, :controllable_dim]


def _subspace_transfer_metrics(
    subspace: np.ndarray,
    coefficients: np.ndarray,
    latent_eigenvalues: np.ndarray,
    predictor_response: np.ndarray,
    ks: Iterable[int],
) -> dict[str, object]:
    tangents = coefficients @ subspace.T
    total = float(np.square(tangents).sum(axis=1).mean())
    predicted = tangents * predictor_response.reshape(1, -1)
    residual = tangents - predicted
    projection_diag = np.square(subspace).sum(axis=1)
    return {
        "latent_mass_share": float(
            np.dot(latent_eigenvalues, projection_diag) / max(float(latent_eigenvalues.sum()), EPS)
        ),
        "predictor_response_energy_share": float(np.square(predicted).sum(axis=1).mean() / max(total, EPS)),
        "prediction_residual_energy_share": float(np.square(residual).sum(axis=1).mean() / max(total, EPS)),
        "top_band_lift": {str(k): _top_band_lift_numpy(tangents, int(k)) for k in ks},
    }


def controllability_alignment_sweep_report(
    config: TransferConfig | None = None,
    controllable_dim: int = 8,
    angles: Iterable[float] = (0, 15, 30, 45, 60, 75, 90),
    ks: Iterable[int] = (4, 8, 16, 32),
    samples: int = 4096,
    random_trials: int = 128,
    seed: int = 0,
) -> dict[str, object]:
    """Hold the spectrum fixed while rotating controllable directions.

    This separates a spectrum-only explanation from the stronger claim that
    planning-relevant directions must align with the eigenspace that the
    predictor preserves across horizon.
    """

    cfg = config or TransferConfig(dim=64, input_exponent=1.5, predictor_exponent=1.0)
    dim = int(cfg.dim)
    if dim < 2:
        raise ValueError("config.dim must be at least 2")
    sub_dim = min(max(1, int(controllable_dim)), dim // 2)
    horizon_ks = sorted({min(max(1, int(k)), dim) for k in ks})
    angle_values = sorted({float(value) for value in angles})
    if not angle_values:
        raise ValueError("at least one angle is required")
    if int(samples) < 2:
        raise ValueError("samples must be at least 2")
    if int(random_trials) < 1:
        raise ValueError("random_trials must be at least 1")

    spectra = diagonal_transfer_spectra(cfg)
    latent = np.asarray(spectra["filtered_latent_eigenvalues"], dtype=np.float64)
    predictor = np.asarray(spectra["predictor_response"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=(int(samples), sub_dim))

    records = []
    for angle in angle_values:
        subspace = _controllable_subspace(dim, sub_dim, angle)
        metrics = _subspace_transfer_metrics(subspace, coefficients, latent, predictor, horizon_ks)
        records.append(
            {
                "angle_degrees": float(angle),
                "controllable_dim": int(sub_dim),
                **metrics,
            }
        )

    random_rows = []
    for _ in range(int(random_trials)):
        subspace = _random_subspace(dim, sub_dim, rng)
        random_rows.append(_subspace_transfer_metrics(subspace, coefficients, latent, predictor, horizon_ks))

    def _summarize_random_scalar(key: str) -> dict[str, float]:
        values = np.array([float(row[key]) for row in random_rows], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }

    random_lift_summary = {}
    for k in horizon_ks:
        values = np.array([float(row["top_band_lift"][str(k)]) for row in random_rows], dtype=np.float64)
        random_lift_summary[str(k)] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }

    aligned = records[0]
    orthogonal = records[-1]
    top_k = str(horizon_ks[-1])
    return {
        "config": asdict(cfg),
        "dim": int(dim),
        "controllable_dim": int(sub_dim),
        "angles": angle_values,
        "ks": horizon_ks,
        "samples": int(samples),
        "random_trials": int(random_trials),
        "seed": int(seed),
        "records": records,
        "random_rotation_null": {
            "latent_mass_share": _summarize_random_scalar("latent_mass_share"),
            "predictor_response_energy_share": _summarize_random_scalar("predictor_response_energy_share"),
            "prediction_residual_energy_share": _summarize_random_scalar("prediction_residual_energy_share"),
            "top_band_lift": random_lift_summary,
        },
        "contrast": {
            "top_k": int(top_k),
            "aligned_top_k_lift": float(aligned["top_band_lift"][top_k]),
            "orthogonal_top_k_lift": float(orthogonal["top_band_lift"][top_k]),
            "aligned_residual_energy_share": float(aligned["prediction_residual_energy_share"]),
            "orthogonal_residual_energy_share": float(orthogonal["prediction_residual_energy_share"]),
            "aligned_latent_mass_share": float(aligned["latent_mass_share"]),
            "orthogonal_latent_mass_share": float(orthogonal["latent_mass_share"]),
        },
        "claim_boundary": (
            "Synthetic subspace rotation with a fixed diagonal spectrum; validates "
            "a mechanism-level distinction between spectral mass and controllability "
            "alignment, not an empirical PushT control result."
        ),
    }


def operator_alignment_null_sweep_report(
    config: TransferConfig | None = None,
    angles: Iterable[float] = (0, 15, 30, 45, 60, 75, 90),
    ks: Iterable[int] = (4, 8, 16, 32),
    random_trials: int = 128,
    seed: int = 0,
) -> dict[str, object]:
    """Compare the oblique peak against stronger nulls.

    The sweep keeps the same latent / predictor spectra but contrasts three
    null families against the oblique basis-rotation signal:

    - permutation-preserving nulls;
    - spectrum-matched randomized bases;
    - repeated-seed randomized bases.

    This is still a toy-model check, but it sharpens the falsifier package
    beyond the single random-rotation baseline.
    """

    cfg = config or TransferConfig(dim=64, input_exponent=1.0, encoder_exponent=1.0, predictor_exponent=1.0)
    dim = int(cfg.dim)
    spectra = diagonal_transfer_spectra(cfg)
    latent = np.asarray(spectra["filtered_latent_eigenvalues"], dtype=np.float64)
    predictor_response = np.asarray(spectra["predictor_response"], dtype=np.float64)
    latent_cov = np.diag(latent)
    predictor_cov = np.diag(predictor_response)
    angle_values = sorted({float(value) for value in angles})
    k_list = sorted({min(max(1, int(k)), dim) for k in ks})
    rng = np.random.default_rng(seed)

    def _score(predictor: np.ndarray) -> dict[str, float]:
        commutator = latent_cov @ predictor - predictor @ latent_cov
        comm_norm = float(
            np.linalg.norm(commutator, ord="fro")
            / max(np.linalg.norm(latent_cov, ord="fro") * np.linalg.norm(predictor, ord="fro"), EPS)
        )
        predicted_cov = predictor @ latent_cov @ predictor.T
        residual_cov = (np.eye(dim, dtype=np.float64) - predictor) @ latent_cov @ (np.eye(dim, dtype=np.float64) - predictor).T
        top_k = str(k_list[-1])
        return {
            "commutator_norm": comm_norm,
            "top_band_lift": _covariance_band_lift(predicted_cov, k_list[-1]),
            "residual_band_lift": _covariance_band_lift(residual_cov, k_list[-1]),
            "predicted_trace_ratio": _matrix_trace_ratio(predicted_cov, float(np.trace(latent_cov))),
            "residual_trace_ratio": _matrix_trace_ratio(residual_cov, float(np.trace(latent_cov))),
        }

    aligned = _score(predictor_cov)
    oblique = _score(_plane_rotation(dim, 45.0) @ predictor_cov @ _plane_rotation(dim, 45.0).T)

    def _permute_null(trial_seed: int) -> dict[str, float]:
        perm = np.random.default_rng(trial_seed).permutation(dim)
        perm_basis = np.eye(dim, dtype=np.float64)[:, perm]
        return _score(perm_basis @ predictor_cov @ perm_basis.T)

    def _random_basis_null(trial_seed: int) -> dict[str, float]:
        basis = _orthonormal(dim, trial_seed)
        return _score(basis @ predictor_cov @ basis.T)

    permutations = [_permute_null(seed + 1000 + idx) for idx in range(int(random_trials))]
    random_bases = [_random_basis_null(seed + 2000 + idx) for idx in range(int(random_trials))]

    def _summarize(rows: list[dict[str, float]], key: str) -> dict[str, float]:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }

    return {
        "config": asdict(cfg),
        "angles": angle_values,
        "ks": k_list,
        "aligned": aligned,
        "oblique": oblique,
        "permutation_preserving_null": {
            "commutator_norm": _summarize(permutations, "commutator_norm"),
            "top_band_lift": _summarize(permutations, "top_band_lift"),
            "residual_band_lift": _summarize(permutations, "residual_band_lift"),
            "predicted_trace_ratio": _summarize(permutations, "predicted_trace_ratio"),
            "residual_trace_ratio": _summarize(permutations, "residual_trace_ratio"),
        },
        "spectrum_matched_random_basis_null": {
            "commutator_norm": _summarize(random_bases, "commutator_norm"),
            "top_band_lift": _summarize(random_bases, "top_band_lift"),
            "residual_band_lift": _summarize(random_bases, "residual_band_lift"),
            "predicted_trace_ratio": _summarize(random_bases, "predicted_trace_ratio"),
            "residual_trace_ratio": _summarize(random_bases, "residual_trace_ratio"),
        },
        "claim_boundary": (
            "Same-spectrum toy-model null sweep comparing permutation-preserving and randomized-basis controls; "
            "it sharpens the falsifier package for the alignment law but remains analytical."
        ),
    }


def operator_alignment_anisotropy_sweep_report(
    config: TransferConfig | None = None,
    input_exponents: Iterable[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    angles: Iterable[float] = (0, 15, 30, 45, 60, 75, 90),
    horizons: Iterable[int] = (1, 2, 4, 8),
    ks: Iterable[int] = (4, 8, 16, 32),
    random_trials: int = 64,
    seed: int = 0,
) -> dict[str, object]:
    """Sweep anisotropy continuously rather than only using flat vs anisotropic.

    This treats input_exponent as a proxy for spectral anisotropy. The goal is
    to locate the regime in which the oblique commutator peak emerges, how fast
    it grows, and whether the same ordering survives the transfer proxies.
    """

    base = config or TransferConfig(dim=64, input_exponent=1.0, encoder_exponent=1.0, predictor_exponent=1.0)
    exponent_values = sorted({float(value) for value in input_exponents})
    if not exponent_values:
        raise ValueError("input_exponents must contain at least one value")
    rows = []
    for idx, exponent in enumerate(exponent_values):
        cfg = TransferConfig(
            dim=base.dim,
            intrinsic_dim=base.intrinsic_dim,
            input_exponent=exponent,
            encoder_exponent=base.encoder_exponent,
            tail_encoder_exponent=base.tail_encoder_exponent,
            breakpoint=base.breakpoint,
            predictor_exponent=base.predictor_exponent,
            heat_time=base.heat_time,
            heat_exponent=base.heat_exponent,
        )
        report = operator_alignment_report(
            cfg,
            angles=angles,
            horizons=horizons,
            ks=ks,
            random_trials=random_trials,
            seed=seed + idx * 17,
        )
        spectra = diagonal_transfer_spectra(cfg)
        stable_rank = _compact_spectrum_stats(spectra["filtered_latent_eigenvalues"])["stable_rank"]
        contrast = report["contrast"]
        rows.append(
            {
                "input_exponent": float(exponent),
                "latent_stable_rank": float(stable_rank),
                "peak_angle_degrees": float(contrast["peak_oblique_angle_degrees"]),
                "peak_commutator_norm": float(contrast["peak_oblique_commutator_norm"]),
                "aligned_commutator_norm": float(contrast["aligned_commutator_norm"]),
                "endpoint_commutator_norm": float(contrast["endpoint_commutator_norm"]),
                "peak_top_band_lift": float(contrast["peak_oblique_horizon"]["predicted_band_lift"]),
                "aligned_top_band_lift": float(contrast["aligned_horizon"]["predicted_band_lift"]),
                "peak_residual_band_lift": float(contrast["peak_oblique_horizon"]["residual_band_lift"]),
                "aligned_residual_band_lift": float(contrast["aligned_horizon"]["residual_band_lift"]),
                "peak_residual_growth_ratio": float(contrast["peak_oblique_residual_growth_ratio"]),
                "aligned_residual_growth_ratio": float(contrast["aligned_residual_growth_ratio"]),
                "random_null_commutator_mean": float(report["random_null"]["commutator_norm"]["mean"]),
            }
        )
    return {
        "config": asdict(base),
        "input_exponents": exponent_values,
        "rows": rows,
        "claim_boundary": (
            "Continuous anisotropy sweep in the synthetic same-spectrum setup; it maps when the oblique peak emerges "
            "but remains an analytical boundary scan."
        ),
    }


def operator_alignment_metric_robustness_report(
    config: TransferConfig | None = None,
    angles: Iterable[float] = (0, 15, 30, 45, 60, 75, 90),
    horizons: Iterable[int] = (1, 2, 4, 8),
    ks: Iterable[int] = (4, 8, 16, 32),
    random_trials: int = 64,
    seed: int = 0,
) -> dict[str, object]:
    """Check that the angle ordering survives more than one transfer proxy."""

    report = operator_alignment_report(
        config=config,
        angles=angles,
        horizons=horizons,
        ks=ks,
        random_trials=random_trials,
        seed=seed,
    )
    top_k = str(report["ks"][-1])
    rows = []
    for row in report["records"]:
        horizon1 = row["horizons"]["1"]
        horizon_max = row["horizons"][str(report["horizons"][-1])]
        rows.append(
            {
                "angle_degrees": float(row["angle_degrees"]),
                "commutator_norm": float(row["commutator_norm"]),
                "h1_top_band_lift": float(horizon1["predicted_band_lift"][top_k]),
                "h1_residual_band_lift": float(horizon1["residual_band_lift"][top_k]),
                "hmax_top_band_lift": float(horizon_max["predicted_band_lift"][top_k]),
                "hmax_residual_band_lift": float(horizon_max["residual_band_lift"][top_k]),
                "h1_predicted_trace_ratio": float(horizon1["predicted_trace_ratio"]),
                "h1_residual_trace_ratio": float(horizon1["residual_trace_ratio"]),
            }
        )
    return {
        "config": asdict(config or TransferConfig(dim=64, input_exponent=1.0, encoder_exponent=1.0, predictor_exponent=1.0)),
        "rows": rows,
        "random_null": report["random_null"],
        "contrast": report["contrast"],
        "claim_boundary": (
            "Metric robustness check for same-spectrum operator alignment; it compares multiple transfer proxies but "
            "does not extend the theory beyond the finite-dimensional toy model."
        ),
    }


def operator_alignment_decoupling_report(
    config: TransferConfig | None = None,
    alignment_angles: Iterable[float] = (0, 45, 90),
    spectrum_exponents: Iterable[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    fixed_alignment_angle: float = 45.0,
    horizons: Iterable[int] = (1, 2, 4, 8),
    ks: Iterable[int] = (4, 8, 16, 32),
    random_trials: int = 64,
    seed: int = 0,
) -> dict[str, object]:
    """Decouple alignment variation from spectrum-shape variation.

    The first axis freezes the latent and predictor spectra and varies only
    the relative basis alignment. The second axis freezes the oblique alignment
    angle and varies spectral anisotropy. Together they make the claim sharper:
    spectra modulate the effect size, but spectra alone do not identify the
    transfer geometry because alignment changes transfer under fixed spectra.
    """

    base = config or TransferConfig(dim=64, input_exponent=1.0, encoder_exponent=1.0, predictor_exponent=1.0)
    angle_values = sorted({float(value) for value in alignment_angles})
    exponent_values = sorted({float(value) for value in spectrum_exponents})
    if not angle_values:
        raise ValueError("alignment_angles must contain at least one value")
    if not exponent_values:
        raise ValueError("spectrum_exponents must contain at least one value")

    alignment_report = operator_alignment_report(
        config=base,
        angles=angle_values,
        horizons=horizons,
        ks=ks,
        random_trials=random_trials,
        seed=seed,
    )
    top_k = str(alignment_report["ks"][-1])
    max_horizon = str(alignment_report["horizons"][-1])
    alignment_rows = []
    for row in alignment_report["records"]:
        horizon1 = row["horizons"]["1"]
        horizon_max = row["horizons"][max_horizon]
        alignment_rows.append(
            {
                "axis": "alignment_varied_spectrum_fixed",
                "angle_degrees": float(row["angle_degrees"]),
                "input_exponent": float(base.input_exponent),
                "latent_stable_rank": float(
                    _compact_spectrum_stats(diagonal_transfer_spectra(base)["filtered_latent_eigenvalues"])["stable_rank"]
                ),
                "commutator_norm": float(row["commutator_norm"]),
                "h1_top_band_lift": float(horizon1["predicted_band_lift"][top_k]),
                "h1_residual_band_lift": float(horizon1["residual_band_lift"][top_k]),
                "hmax_residual_growth_ratio": float(row["residual_growth_ratio"]),
                "hmax_top_band_lift": float(horizon_max["predicted_band_lift"][top_k]),
            }
        )

    spectrum_rows = []
    for idx, exponent in enumerate(exponent_values):
        cfg = TransferConfig(
            dim=base.dim,
            intrinsic_dim=base.intrinsic_dim,
            input_exponent=exponent,
            encoder_exponent=base.encoder_exponent,
            tail_encoder_exponent=base.tail_encoder_exponent,
            breakpoint=base.breakpoint,
            predictor_exponent=base.predictor_exponent,
            heat_time=base.heat_time,
            heat_exponent=base.heat_exponent,
        )
        report = operator_alignment_report(
            config=cfg,
            angles=(fixed_alignment_angle,),
            horizons=horizons,
            ks=ks,
            random_trials=random_trials,
            seed=seed + 100 + idx * 17,
        )
        row = report["records"][0]
        horizon1 = row["horizons"]["1"]
        horizon_max = row["horizons"][str(report["horizons"][-1])]
        spectrum_rows.append(
            {
                "axis": "spectrum_varied_alignment_fixed",
                "angle_degrees": float(fixed_alignment_angle),
                "input_exponent": float(exponent),
                "latent_stable_rank": float(
                    _compact_spectrum_stats(diagonal_transfer_spectra(cfg)["filtered_latent_eigenvalues"])["stable_rank"]
                ),
                "commutator_norm": float(row["commutator_norm"]),
                "h1_top_band_lift": float(horizon1["predicted_band_lift"][str(report["ks"][-1])]),
                "h1_residual_band_lift": float(horizon1["residual_band_lift"][str(report["ks"][-1])]),
                "hmax_residual_growth_ratio": float(row["residual_growth_ratio"]),
                "hmax_top_band_lift": float(horizon_max["predicted_band_lift"][str(report["ks"][-1])]),
            }
        )

    alignment_values = np.asarray([row["commutator_norm"] for row in alignment_rows], dtype=np.float64)
    spectrum_values = np.asarray([row["commutator_norm"] for row in spectrum_rows], dtype=np.float64)
    return {
        "config": asdict(base),
        "fixed_spectrum_alignment_rows": alignment_rows,
        "fixed_alignment_spectrum_rows": spectrum_rows,
        "summary": {
            "alignment_axis_commutator_range": float(alignment_values.max() - alignment_values.min()),
            "spectrum_axis_commutator_range": float(spectrum_values.max() - spectrum_values.min()),
            "fixed_alignment_angle_degrees": float(fixed_alignment_angle),
            "top_k": int(top_k),
        },
        "claim_boundary": (
            "Two-axis synthetic decoupling table: under fixed spectra, alignment alone changes transfer; "
            "under fixed alignment, spectrum anisotropy changes effect size. This supports under-identification "
            "of transfer geometry by spectra alone, not a claim that spectra are irrelevant."
        ),
    }


def _compact_spectrum_stats(values: np.ndarray) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64).clip(min=0)
    total = float(vals.sum())
    return {
        "trace": total,
        "lambda_max": float(vals.max(initial=0.0)),
        "stable_rank": float(total / max(float(vals.max(initial=0.0)), EPS)),
        "participation_ratio": float(total * total / max(float(np.square(vals).sum()), EPS)),
    }


def spectral_transfer_report(
    config: TransferConfig,
    slope_windows: Iterable[tuple[int, int | None]] = ((1, None),),
    horizons: Iterable[int] = (1, 2, 4, 8),
) -> dict[str, object]:
    """Return a JSON-serializable theory report for the configured system."""

    spectra = diagonal_transfer_spectra(config)
    windows = [(int(start), stop if stop is None else int(stop)) for start, stop in slope_windows]
    slopes = {
        name: {f"{start}:{stop or 'end'}": fit_loglog_slope(values, start, stop) for start, stop in windows}
        for name, values in spectra.items()
        if name.endswith("eigenvalues") or name in {"input_variance", "encoder_metric", "predictor_response"}
    }
    return {
        "config": asdict(config),
        "expected_oracle_alpha": float((config.input_exponent + config.encoder_exponent) / config.intrinsic_dim),
        "spectra": {name: values.tolist() for name, values in spectra.items()},
        "spectrum_stats": {name: _compact_spectrum_stats(values) for name, values in spectra.items()},
        "band_energy": {name: band_energy_summary(values) for name, values in spectra.items()},
        "slopes": slopes,
        "operator_alignment_theorem": operator_alignment_theorem_report(config),
        "multi_horizon": multi_horizon_transfer_report(config, horizons=horizons),
        "operator_alignment": operator_alignment_report(config),
        "oblique_frame_stress": oblique_frame_stress_report(config),
        "heat_filter_recovery": heat_filter_recovery_report(config),
        "sigreg_local_anisotropy": sigreg_local_anisotropy_report(dim=config.dim),
        "identifiability_controllability": identifiability_controllability_report(dim=config.dim),
    }
