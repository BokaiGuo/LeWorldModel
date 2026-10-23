"""Root-disjoint empirical decision-aliasing diagnostics.

This module audits a *fixed* history representation against vectors of real
within-root candidate contrasts.  It deliberately separates two statements:

* exact representation collisions with differing contrasts imply the elementary
  half-gap lower bound for any deterministic contrast readout of that interface;
* nearby representations with differing contrasts are only empirical evidence
  of local decision aliasing, unless additional smoothness assumptions are
  independently justified.

It is not a causal-identifiability theorem, a conditional risk certificate, or
a substitute for root-disjoint intervention execution.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SCHEMA = "empirical_decision_aliasing_audit.v1"


def _arrays(
    representation: np.ndarray,
    contrast: np.ndarray | None,
    root_ids: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    rep = np.asarray(representation, dtype=np.float64)
    roots = np.asarray(root_ids).reshape(-1)
    if rep.ndim != 2 or not len(rep) or not np.isfinite(rep).all():
        raise ValueError(f"{name} representation must be non-empty finite [rows, dimensions]")
    if roots.shape != (len(rep),) or any(not str(value) for value in roots):
        raise ValueError(f"{name} root_ids must align with non-empty root identifiers")
    if contrast is None:
        return rep, None, roots
    values = np.asarray(contrast, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(rep) or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} contrast must be finite [rows, pair_dimensions]")
    return rep, values, roots


def _nearest_cross_root(reference: np.ndarray, reference_roots: np.ndarray, query: np.ndarray, query_root: Any) -> tuple[int | None, float | None]:
    allowed = reference_roots != query_root
    if not np.any(allowed):
        return None, None
    indices = np.flatnonzero(allowed)
    distances = np.linalg.norm(reference[indices] - query[None, :], axis=1)
    offset = int(np.argmin(distances))
    return int(indices[offset]), float(distances[offset])


def calibrate_representation_radius(
    representation: np.ndarray,
    root_ids: np.ndarray,
    *,
    quantile: float = 0.25,
) -> dict[str, Any]:
    """Freeze a neighborhood radius from fit-only, cross-root representation distances.

    This is a scale-selection utility, not a conformal or decision-risk bound.
    It intentionally does not accept contrast labels.
    """

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    rep, _contrast, roots = _arrays(representation, None, root_ids, name="fit")
    distances: list[float] = []
    for index, value in enumerate(rep):
        _neighbor, distance = _nearest_cross_root(rep, roots, value, roots[index])
        if distance is not None:
            distances.append(distance)
    if not distances:
        raise ValueError("fit panel has no cross-root representation neighbors")
    return {
        "schema": "decision_alias_radius_calibration.v1",
        "representation_radius": float(np.quantile(np.asarray(distances), float(quantile))),
        "quantile": float(quantile),
        "fit_row_count": int(len(rep)),
        "cross_root_neighbor_count": int(len(distances)),
        "claim_boundary": "fit-only representation-neighborhood scale; not a decision or coverage guarantee",
    }


def empirical_decision_aliasing_audit(
    reference_representation: np.ndarray,
    reference_contrast: np.ndarray,
    reference_root_ids: np.ndarray,
    query_representation: np.ndarray,
    query_contrast: np.ndarray,
    query_root_ids: np.ndarray,
    *,
    representation_radius: float,
    contrast_gap_threshold: float,
    exact_tolerance: float = 1.0e-12,
    max_reported_pairs: int = 64,
) -> dict[str, Any]:
    """Audit nearest cross-root histories for empirical decision aliasing.

    The reference panel must be disjoint in root identity from the query panel
    for the resulting pairs to be independent across the two roles.  Repeated
    root identifiers are permitted within either panel, but same-root matches
    are always excluded.
    """

    radius, gap, tolerance = float(representation_radius), float(contrast_gap_threshold), float(exact_tolerance)
    if radius < 0.0 or gap < 0.0 or tolerance < 0.0 or not np.isfinite([radius, gap, tolerance]).all():
        raise ValueError("radius, contrast_gap_threshold, and exact_tolerance must be finite and non-negative")
    if int(max_reported_pairs) < 1:
        raise ValueError("max_reported_pairs must be positive")
    ref_rep, ref_contrast, ref_roots = _arrays(reference_representation, reference_contrast, reference_root_ids, name="reference")
    query_rep, query_values, query_roots = _arrays(query_representation, query_contrast, query_root_ids, name="query")
    assert ref_contrast is not None and query_values is not None
    if ref_rep.shape[1] != query_rep.shape[1] or ref_contrast.shape[1] != query_values.shape[1]:
        raise ValueError("reference and query representation/contrast dimensions must match")
    root_overlap = sorted(set(ref_roots.tolist()) & set(query_roots.tolist()), key=str)
    if root_overlap:
        raise ValueError("reference and query root panels must be disjoint")

    pairs: list[dict[str, Any]] = []
    skipped = 0
    for index, representation in enumerate(query_rep):
        reference_index, distance = _nearest_cross_root(ref_rep, ref_roots, representation, query_roots[index])
        if reference_index is None or distance is None:
            skipped += 1
            continue
        contrast_gap = float(np.max(np.abs(query_values[index] - ref_contrast[reference_index])))
        exact = bool(distance <= tolerance)
        near = bool(distance <= radius and contrast_gap >= gap)
        pairs.append({
            "query_index": int(index), "query_root_id": str(query_roots[index]),
            "reference_index": int(reference_index), "reference_root_id": str(ref_roots[reference_index]),
            "representation_distance": distance, "contrast_linf_gap": contrast_gap,
            "near_decision_alias": near, "exact_representation_collision": exact,
            "exact_collision_half_gap_lower_bound": float(0.5 * contrast_gap) if exact else None,
        })
    near_pairs = [row for row in pairs if row["near_decision_alias"]]
    exact_pairs = [row for row in pairs if row["exact_representation_collision"] and row["contrast_linf_gap"] > 0.0]
    ordered = sorted(pairs, key=lambda row: (-int(row["near_decision_alias"]), -row["contrast_linf_gap"], row["representation_distance"], row["query_index"]))
    return {
        "schema": SCHEMA,
        "reference_row_count": int(len(ref_rep)), "query_row_count": int(len(query_rep)),
        "reference_query_root_disjoint": True, "root_overlap": [],
        "representation_radius": radius, "contrast_gap_threshold": gap, "exact_tolerance": tolerance,
        "evaluated_query_count": int(len(pairs)), "skipped_query_count": int(skipped),
        "near_decision_alias_count": int(len(near_pairs)),
        "near_decision_alias_rate": float(len(near_pairs) / len(pairs)) if pairs else None,
        "exact_collision_with_contrast_gap_count": int(len(exact_pairs)),
        "max_exact_collision_half_gap_lower_bound": (
            None if not exact_pairs else float(max(row["exact_collision_half_gap_lower_bound"] for row in exact_pairs))
        ),
        "top_pairs": ordered[: int(max_reported_pairs)],
        "claim_boundary": (
            "near-decision-alias rows are root-disjoint empirical diagnostics, not a global identifiability lower bound; "
            "the half-gap bound is reported only for exact representation collisions under a deterministic contrast readout."
        ),
    }
