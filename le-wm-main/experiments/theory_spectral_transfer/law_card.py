"""Law-card style summaries for theory-side spectral transfer."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .operators import TransferConfig, operator_alignment_report, operator_alignment_theorem_report


def operator_alignment_law_card(
    config: TransferConfig | None = None,
    angles: tuple[float, ...] = (0, 15, 30, 45, 60, 75, 90),
    horizons: tuple[int, ...] = (1, 2, 4, 8),
    ks: tuple[int, ...] = (4, 8, 16, 32),
    random_trials: int = 128,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a structured law card for the operator-alignment claim."""

    cfg = config or TransferConfig(dim=64, input_exponent=1.0, encoder_exponent=1.0, predictor_exponent=1.0)
    report = operator_alignment_report(
        config=cfg,
        angles=angles,
        horizons=horizons,
        ks=ks,
        random_trials=random_trials,
        seed=seed,
    )
    theorem = operator_alignment_theorem_report(cfg, angles=angles)
    contrast = report["contrast"]
    peak = max(report["records"], key=lambda row: row["commutator_norm"])
    top_k = str(report["ks"][-1])
    return {
        "title": "Operator alignment law card",
        "object": "same-spectrum operator alignment under predictor basis rotation",
        "config": asdict(cfg),
        "phenomenon": (
            "Holding the latent and predictor eigenvalues fixed still changes top-band transfer "
            "and residual growth when the predictor basis is rotated away from the commuting case."
        ),
        "independent_unit": "one latent covariance and one predictor operator with fixed spectra",
        "candidate_law": {
            "statement": (
                "For anisotropic latent spectra, the commutator norm peaks at an oblique rotation, "
                "and the same oblique rotation produces a distinct top-band transfer profile."
            ),
            "symbol": "N(Σ, P) = ||[Σ, P]||_F / (||Σ||_F ||P||_F)",
            "peak_angle_degrees": theorem["theorem"]["peak_angle_degrees"],
        },
        "scope": {
            "finite_dimensional": True,
            "anisotropic_latent_spectrum_required": True,
            "fixed_eigenvalues": True,
            "single_plane_rotation": True,
            "no_learning_dynamics": True,
        },
        "boundary_conditions": [
            "flat latent spectrum collapses the alignment signal",
            "commuting endpoint / permutation cases remove the commutator peak",
            "top-band effects are basis dependent and therefore need the noncommutative object",
        ],
        "nearest_rivals": [
            "spectrum-only explanation",
            "coordinate-order artifact",
            "finite-window slope artifact",
        ],
        "falsifiers": [
            "if anisotropic same-spectrum rotation does not create a nonzero commutator peak, the alignment law fails",
            "if the peak-oblique case is not separable from the random-rotation null, the boundary is not identifiable",
            "if a flat-spectrum null still shows the same commutator/top-band pattern, the law is overclaiming",
        ],
        "evidence": {
            "aligned": {
                "angle_degrees": contrast["aligned_angle_degrees"],
                "commutator_norm": contrast["aligned_commutator_norm"],
                "top_band_lift": contrast["aligned_horizon"]["predicted_band_lift"],
                "residual_growth_ratio": contrast["aligned_residual_growth_ratio"],
            },
            "peak_oblique": {
                "angle_degrees": contrast["peak_oblique_angle_degrees"],
                "commutator_norm": contrast["peak_oblique_commutator_norm"],
                "top_band_lift": contrast["peak_oblique_horizon"]["predicted_band_lift"],
                "residual_growth_ratio": contrast["peak_oblique_residual_growth_ratio"],
            },
            "endpoint": {
                "angle_degrees": contrast["endpoint_angle_degrees"],
                "commutator_norm": contrast["endpoint_commutator_norm"],
                "top_band_lift": contrast["endpoint_horizon"]["predicted_band_lift"],
                "residual_growth_ratio": contrast["endpoint_residual_growth_ratio"],
            },
            "random_null": report["random_null"]["commutator_norm"],
            "theorem_exact_by_angle": theorem["theorem"]["exact_by_angle"],
            "top_k": int(top_k),
        },
        "claim_boundary": (
            "This is a synthetic same-spectrum / different-basis theorem candidate. "
            "It identifies operator alignment and noncommutativity as the relevant boundary object, not control."
        ),
    }
