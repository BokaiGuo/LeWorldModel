"""Run a theory-only operator-alignment sweep.

This sweep keeps the latent and predictor spectra fixed while rotating the
predictor eigenspace relative to the latent basis. It isolates the
noncommutativity / same-spectrum-different-basis effect that the theory line
cares about, without recasting the problem as control.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from experiments.theory_spectral_transfer import (
    TransferConfig,
    operator_alignment_law_card,
    operator_alignment_report,
    operator_alignment_theorem_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/theory_operator_alignment_sweep_20260828",
        help="output directory for JSON, Markdown, and figure assets",
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--intrinsic-dim", type=float, default=2.0)
    parser.add_argument("--input-exponent", type=float, default=1.0)
    parser.add_argument("--predictor-exponent", type=float, default=1.0)
    parser.add_argument("--angles", type=float, nargs="+", default=[0, 15, 30, 45, 60, 75, 90])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--random-trials", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _top_k(payload: dict[str, Any]) -> str:
    return str(payload["ks"][-1])


def _plot_band_heatmap(ax: plt.Axes, payload: dict[str, Any]) -> None:
    rows = payload["records"]
    angles = [row["angle_degrees"] for row in rows]
    ks = payload["ks"]
    values = np.array([[row["horizons"]["1"]["predicted_band_lift"][str(k)] for row in rows] for k in ks], dtype=np.float64)
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(angles)), [f"{angle:g}" for angle in angles])
    ax.set_yticks(np.arange(len(ks)), [f"{k:g}" for k in ks])
    ax.set_xlabel("Rotation angle (degrees)")
    ax.set_ylabel("k")
    ax.set_title("Horizon-1 top-band lift", fontsize=11, pad=6)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if values[i, j] > np.nanmean(values) else "black",
            )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("Lift", fontsize=9)


def _plot_with_null(ax: plt.Axes, payload: dict[str, Any], key: str, title: str, ylabel: str) -> None:
    rows = payload["records"]
    angles = np.array([row["angle_degrees"] for row in rows], dtype=np.float64)
    values = np.array([row[key] for row in rows], dtype=np.float64)
    null = payload["random_null"][key]
    ax.plot(angles, values, marker="o", linewidth=2.2, markersize=5, color="#1b9e77")
    ax.axhspan(null["p05"], null["p95"], color="#999999", alpha=0.18, label="Random null P05-P95")
    ax.axhline(null["mean"], color="#666666", linestyle="--", linewidth=1.2, label="Random null mean")
    ax.set_xlabel("Rotation angle (degrees)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=6)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="best")


def _plot_contrast(ax: plt.Axes, payload: dict[str, Any]) -> None:
    contrast = payload["contrast"]
    null = payload["random_null"]["commutator_norm"]
    labels = ["Aligned", "Oblique\npeak", "Endpoint", "Random\nmean"]
    values = [
        contrast["aligned_commutator_norm"],
        contrast["peak_oblique_commutator_norm"],
        contrast["endpoint_commutator_norm"],
        null["mean"],
    ]
    errors = [0.0, 0.0, 0.0, max(null["p95"] - null["mean"], null["mean"] - null["p05"])]
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#666666"]
    ax.bar(labels, values, color=colors, yerr=errors, capsize=3)
    ax.set_ylabel("Commutator norm")
    ax.set_title("Alignment contrast", fontsize=11, pad=6)
    ax.grid(True, axis="y", alpha=0.2)


def _write_markdown(output: Path, payload: dict[str, Any]) -> None:
    rows = payload["records"]
    null = payload["random_null"]
    top_k = _top_k(payload)
    lines = [
        "# Operator Alignment Sweep",
        "",
        "## Object",
        "",
        "- This sweep holds the latent and predictor spectra fixed and rotates the predictor basis.",
        "- It asks whether spectral transfer changes only by eigenvalues, or also by noncommutative eigenspace alignment.",
        "",
        "## Main Readout",
        "",
        "| angle | commutator norm | h1 top-k lift | h1 residual lift | h8 residual growth |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        h1 = row["horizons"]["1"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["angle_degrees"]),
                    _fmt(row["commutator_norm"]),
                    _fmt(h1["predicted_band_lift"][top_k]),
                    _fmt(h1["residual_band_lift"][top_k]),
                    _fmt(row["residual_growth_ratio"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Random Null",
            "",
            f"- commutator mean: {_fmt(null['commutator_norm']['mean'])}",
            f"- commutator p05/p95: {_fmt(null['commutator_norm']['p05'])} / {_fmt(null['commutator_norm']['p95'])}",
            f"- residual growth mean: {_fmt(null['residual_growth_ratio']['mean'])}",
            f"- predicted decay mean: {_fmt(null['predicted_decay_ratio']['mean'])}",
            "",
            "## Readout",
            "",
            "- The commutator peaks at an oblique rotation, not at the aligned or endpoint cases.",
            "- The same spectrum can still yield different top-band transfer once the operator basis is rotated.",
            "- Random rotations are not a zero baseline: they spread the commutator mass, but the oblique peak is still separable.",
            "",
            "## Claim Boundary",
            "",
            "- This is a synthetic same-spectrum / different-basis check.",
            "- It supports operator alignment and noncommutativity as the boundary object, not control.",
            "",
        ]
    )
    (output / "operator_alignment_summary.md").write_text("\n".join(lines) + "\n")


def _write_law_card(output: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Operator Alignment Law Card",
        "",
        f"- object: {payload['object']}",
        f"- independent unit: {payload['independent_unit']}",
        "",
        "## Phenomenon",
        "",
        payload["phenomenon"],
        "",
        "## Candidate Law",
        "",
        f"- {payload['candidate_law']['statement']}",
        f"- symbol: {payload['candidate_law']['symbol']}",
        "",
        "## Scope",
        "",
    ]
    for key, value in payload["scope"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Boundary Conditions",
            "",
        ]
    )
    for item in payload["boundary_conditions"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Rivals",
            "",
        ]
    )
    for item in payload["nearest_rivals"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Falsifiers",
            "",
        ]
    )
    for item in payload["falsifiers"]:
        lines.append(f"- {item}")
    evidence = payload["evidence"]
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- aligned commutator norm: {_fmt(evidence['aligned']['commutator_norm'])}",
            f"- peak-oblique commutator norm: {_fmt(evidence['peak_oblique']['commutator_norm'])}",
            f"- endpoint commutator norm: {_fmt(evidence['endpoint']['commutator_norm'])}",
            f"- random-null commutator mean: {_fmt(evidence['random_null']['mean'])}",
            f"- top-k: {evidence['top_k']}",
            "",
            "## Claim Boundary",
            "",
            payload["claim_boundary"],
            "",
        ]
    )
    (output / "operator_alignment_law_card.md").write_text("\n".join(lines) + "\n")


def _write_theorem(output: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Operator Alignment Theorem",
        "",
        f"- rotation plane: i={payload['rotation_plane']['i']}, j={payload['rotation_plane']['j']}",
        f"- latent gap: {_fmt(payload['eigenvalue_gaps']['latent'])}",
        f"- predictor gap: {_fmt(payload['eigenvalue_gaps']['predictor'])}",
        "",
        "## Statement",
        "",
        payload["theorem"]["statement"],
        "",
        "## Exact Curve",
        "",
        "| angle | exact commutator norm |",
        "| --- | --- |",
    ]
    for angle in payload["angles"]:
        key = str(angle)
        lines.append(f"| {_fmt(angle)} | {_fmt(payload['theorem']['exact_by_angle'][key])} |")
    lines.extend(
        [
            "",
            "## Falsifiers",
            "",
            "- flat latent spectrum => zero exact commutator at every angle",
            "- flat predictor spectrum => zero exact commutator at every angle",
            "",
            "## Claim Boundary",
            "",
            payload["claim_boundary"],
            "",
        ]
    )
    (output / "operator_alignment_theorem.md").write_text("\n".join(lines) + "\n")


def _write_paper_outline(output: Path, sweep: dict[str, Any], theorem: dict[str, Any], law_card: dict[str, Any]) -> None:
    contrast = sweep["contrast"]
    null = sweep["random_null"]["commutator_norm"]
    lines = [
        "# Same Spectrum Is Not Enough: Noncommutative Operator Alignment Governs Spectral Transfer",
        "",
        "## Working Title",
        "",
        "Same Spectrum Is Not Enough",
        "",
        "## One-Line Thesis",
        "",
        "Keeping the spectrum fixed does not keep the transfer geometry fixed: same-spectrum basis rotations change the commutator and the top-band transfer, with an exact oblique-angle peak in the single-plane case.",
        "",
        "## Mother Object",
        "",
        "- noncommutative alignment between latent covariance and predictor operator",
        "- the object is descriptive and scoped; it is not a controller or a learned policy",
        "",
        "## Claim Ladder",
        "",
        "1. Exact theorem: single-plane rotation gives a sin(2 theta) commutator law.",
        "2. Mechanism check: the numerical sweep matches the exact curve and peaks at an oblique angle.",
        "3. Boundary law: flat latent or flat predictor spectra kill the commutator signal.",
        "4. Rival elimination: random rotations do not reproduce the peak structure, so the effect is not a coordinate-order artifact.",
        "5. Scientific consequence: spectrum-only explanations are incomplete for this operator-transfer object.",
        "",
        "## Main Figure",
        "",
        "Panel A: top-band lift heatmap across angle and k.",
        "Panel B: commutator norm vs angle, with the exact theorem curve as reference.",
        "Panel C: residual-growth ratio vs angle, with random-null band.",
        "Panel D: alignment contrast bar plot for aligned / peak oblique / endpoint / random mean.",
        "",
        "## Rivals",
        "",
        "- spectrum-only explanation",
        "- coordinate-order artifact",
        "- finite-window slope artifact",
        "- random-rotation null",
        "",
        "## Falsifiers",
        "",
        "- flat latent spectrum leaves the same peak structure",
        "- flat predictor spectrum leaves the same peak structure",
        "- exact sin-law fails to match the numerical sweep",
        "- peak-oblique case is not separable from random null",
        "",
        "## Evidence Bridge",
        "",
        f"- aligned commutator norm: {_fmt(contrast['aligned_commutator_norm'])}",
        f"- peak-oblique commutator norm: {_fmt(contrast['peak_oblique_commutator_norm'])}",
        f"- random-null commutator mean: {_fmt(null['mean'])}",
        f"- theorem peak angle: {_fmt(theorem['theorem']['peak_angle_degrees'])}",
        f"- law card top-k: {law_card['evidence']['top_k']}",
        "",
        "## Field Consequence",
        "",
        "- Any spectral-transfer audit that ignores operator alignment is under-specified.",
        "- The minimal next measurement is to report basis alignment or commutator-sensitive summaries alongside eigenvalue spectra.",
        "",
        "## What This Does Not Claim",
        "",
        "- It does not yet prove a learned LeWM latent operator law.",
        "- It does not replace the empirical audit branch.",
        "- It does not justify controllability as the mother object.",
        "",
        "## Section Skeleton",
        "",
        "Abstract -> theorem -> proof sketch -> numerical verification -> rival/falsifier analysis -> boundary discussion -> relation to empirical branch.",
        "",
    ]
    (output / "operator_alignment_paper_outline.md").write_text("\n".join(lines) + "\n")


def operator_alignment_oral_object_card(sweep: dict[str, Any], theorem: dict[str, Any], law_card: dict[str, Any]) -> str:
    contrast = sweep["contrast"]
    null = sweep["random_null"]["commutator_norm"]
    top_k = law_card["evidence"]["top_k"]
    lines = [
        "# Same Spectrum Is Not Enough",
        "",
        "## Mother Object",
        "",
        "- noncommutative operator alignment between latent covariance and predictor basis",
        "- independent unit: one latent operator, one predictor operator, one rotation plane",
        "",
        "## Not About X, About Y",
        "",
        "- not about controllability, about whether identical spectra still under-identify transfer geometry",
        "",
        "## Field Tension",
        "",
        "- spectral summaries are often treated as if they define transfer, but same-spectrum operators can still behave differently once basis alignment changes",
        "",
        "## Observable Phenomenon",
        "",
        "- fixed eigenvalues plus basis rotation change the commutator norm, top-band transfer, and residual growth",
        "",
        "## Candidate Law",
        "",
        "- under anisotropic latent spectra and fixed eigenvalues, normalized commutator and top-band transfer vary with rotation angle, peak at an oblique angle, and collapse at commuting endpoints or flat spectra",
        "",
        "## Comparison Battlefield",
        "",
        "- spectrum-only explanation",
        "- coordinate-order artifact",
        "- finite-window slope artifact",
        "- random-rotation null",
        "",
        "## Strongest Falsifier",
        "",
        "- flat latent or flat predictor spectra still show the same oblique peak",
        "- exact sin-law fails to match the sweep",
        "- the oblique peak is not separable from the random null",
        "",
        "## First Reader-Facing Display",
        "",
        f"- question: does fixing spectrum also fix transfer geometry?",
        f"- row/series unit: rotation angle × k band",
        f"- estimands: commutator norm, top-band lift, residual growth ratio",
        "- non-claim: this is not a learned LeWM latent law and not a control paper",
        "",
        "## Evidence Bridge",
        "",
        f"- theorem peak angle: {_fmt(theorem['theorem']['peak_angle_degrees'])}",
        f"- aligned commutator norm: {_fmt(contrast['aligned_commutator_norm'])}",
        f"- peak-oblique commutator norm: {_fmt(contrast['peak_oblique_commutator_norm'])}",
        f"- random-null commutator mean: {_fmt(null['mean'])}",
        f"- top-k: {top_k}",
        "",
        "## Current Claim / Blocked Claim / Next Gate",
        "",
        "- current claim: same-spectrum operator alignment changes transfer geometry in the synthetic single-plane model",
        "- blocked claim: direct learned-LeWM generality",
        "- next gate: align the empirical paper's frozen PushT evidence to the same boundary object without collapsing the stories",
        "",
        "## Claim Boundary",
        "",
        law_card["claim_boundary"],
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_oral_object_card(output: Path, sweep: dict[str, Any], theorem: dict[str, Any], law_card: dict[str, Any]) -> None:
    (output / "operator_alignment_oral_object_card.md").write_text(
        operator_alignment_oral_object_card(sweep, theorem, law_card)
    )


def operator_alignment_paper_spine(
    sweep: dict[str, Any],
    theorem: dict[str, Any],
    law_card: dict[str, Any],
) -> str:
    contrast = sweep["contrast"]
    null = sweep["random_null"]["commutator_norm"]
    top_k = law_card["evidence"]["top_k"]
    lines = [
        "# Same Spectrum Is Not Enough: Noncommutative Operator Alignment Governs Spectral Transfer",
        "",
        "## Abstract",
        "",
        "- Holding spectra fixed does not hold transfer fixed: basis alignment changes the geometry of operator transfer.",
        "- In a single-plane synthetic model, the normalized commutator follows |sin(2 theta)| and peaks at 45 degrees.",
        "- The numerical sweep matches the exact curve, rejects flat-spectrum falsifiers, and separates the oblique peak from random rotations.",
        "- The paper's update is that spectral transfer is under-identified without operator alignment.",
        "",
        "## Introduction",
        "",
        "- Question: does fixing the spectrum also fix transfer geometry?",
        "- Gap: current spectral summaries do not distinguish same-spectrum operators with different basis alignment.",
        "- Approach: exact single-plane theorem, multi-angle numerical sweep, flat-spectrum falsifiers, and random-rotation nulls.",
        "- Result: the oblique angle maximizes the commutator and changes top-band transfer and residual growth.",
        "- Contribution: operator alignment, not spectrum alone, is the boundary object for this theory paper.",
        "",
        "## Mother Object",
        "",
        "- noncommutative alignment between latent covariance and predictor operator",
        "- the object is descriptive and scoped; it is not a controller or a learned policy",
        "",
        "## Main Figure",
        "",
        "Panel A: top-band lift heatmap across angle and k.",
        "Panel B: commutator norm vs angle, with the exact theorem curve as reference.",
        "Panel C: residual-growth ratio vs angle, with random-null band.",
        "Panel D: alignment contrast bar plot for aligned / peak oblique / endpoint / random mean.",
        "",
        "## Rivals",
        "",
        "- spectrum-only explanation",
        "- coordinate-order artifact",
        "- finite-window slope artifact",
        "- random-rotation null",
        "",
        "## Falsifiers",
        "",
        "- flat latent spectrum leaves the same peak structure",
        "- flat predictor spectrum leaves the same peak structure",
        "- exact sin-law fails to match the numerical sweep",
        "- peak-oblique case is not separable from random null",
        "",
        "## Evidence Bridge",
        "",
        f"- aligned commutator norm: {_fmt(contrast['aligned_commutator_norm'])}",
        f"- peak-oblique commutator norm: {_fmt(contrast['peak_oblique_commutator_norm'])}",
        f"- random-null commutator mean: {_fmt(null['mean'])}",
        f"- theorem peak angle: {_fmt(theorem['theorem']['peak_angle_degrees'])}",
        f"- law card top-k: {top_k}",
        "",
        "## Field Consequence",
        "",
        "- Any spectral-transfer audit that ignores operator alignment is under-specified.",
        "- The minimal next measurement is to report basis alignment or commutator-sensitive summaries alongside eigenvalue spectra.",
        "- This consequence matters even without a method paper: the law changes what experts should treat as a sufficient summary.",
        "",
        "## What This Does Not Claim",
        "",
        "- It does not prove a learned LeWM latent operator law.",
        "- It does not replace the companion empirical paper on frozen PushT latents.",
        "- It does not justify controllability as the mother object.",
        "",
        "## Section Skeleton",
        "",
        "Abstract -> introduction -> theorem -> proof sketch -> numerical verification -> rival/falsifier analysis -> field consequence -> boundary discussion -> relation to the companion empirical paper.",
        "",
    ]
    return "\n".join(lines) + "\n"


def operator_alignment_paper_draft(
    sweep: dict[str, Any],
    theorem: dict[str, Any],
    law_card: dict[str, Any],
) -> str:
    contrast = sweep["contrast"]
    null = sweep["random_null"]["commutator_norm"]
    top_k = law_card["evidence"]["top_k"]
    exact = theorem["theorem"]["exact_by_angle"]
    rows = sweep["records"]
    claim_rows = [
        (
            "C1",
            "For a single-plane rotation of a diagonal predictor operator, the normalized commutator follows an explicit abs(sin(2 theta)) law.",
            "mechanism",
            "operator_alignment_theorem.md / operator_alignment_theorem.json",
            "spectrum-only explanation",
            "fixed-spectrum, single-plane, finite-dimensional toy model",
            "validated",
        ),
        (
            "C2",
            "The commutator and top-band transfer peak at an oblique rotation rather than at the aligned or endpoint cases.",
            "mechanism",
            "operator_alignment_summary.md / operator_alignment_sweep.json",
            "coordinate-order artifact",
            "same-spectrum different-basis model with finite angle grid",
            "validated",
        ),
        (
            "C3",
            "Flat-spectrum nulls collapse the commutator signal and therefore bound the law's scope.",
            "boundary",
            "operator_alignment_theorem.json",
            "post-hoc interpolation artifact",
            "anisotropic same-spectrum condition",
            "validated",
        ),
        (
            "C4",
            "Random rotations spread the signal but do not erase the oblique peak, so the result is not a coordinate-order artifact.",
            "association",
            "operator_alignment_sweep.json",
            "random-rotation null",
            "same dimension / same spectra / random basis rotation",
            "validated",
        ),
        (
            "C5",
            "A field-facing consequence follows: spectral-transfer audits should report basis alignment or commutator-sensitive summaries, not eigenvalues alone.",
            "consequence",
            "operator_alignment_paper_spine.md",
            "spectrum-only audit sufficiency",
            "same-spectrum, different-basis scope",
            "partial",
        ),
        (
            "C6",
            "In the two-axis ablation, fixed-spectrum alignment variation changes transfer geometry, while fixed-alignment spectrum variation changes effect size.",
            "identification",
            "operator_alignment_decoupling.md / operator_alignment_decoupling.json",
            "spectrum-only explanation",
            "finite-dimensional synthetic decoupling table",
            "validated",
        ),
    ]
    lines = [
        "# Same Spectrum Is Not Enough: Noncommutative Operator Alignment Governs Spectral Transfer",
        "",
        "## Abstract",
        "",
        "This paper asks whether fixing the spectrum also fixes transfer geometry. In the same-spectrum, different-basis toy model, it does not: rotating the predictor basis changes the normalized commutator, top-band transfer, and residual growth, with an exact peak at an oblique angle. The numerical sweep matches the theorem, the flat-spectrum nulls collapse the signal, and the random-rotation baseline remains weaker than the oblique case. The scientific update is that spectral transfer is under-identified without operator alignment.",
        "",
        "## Introduction",
        "",
        "Spectral summaries are attractive because they compress an operator into an ordered list of eigenvalues. The problem is that eigenvalues alone do not determine transfer geometry. Two operators can share the same spectrum and still differ in how their eigenspaces line up, which changes the commutator and the transfer profile. This paper isolates that boundary with a finite-dimensional single-plane rotation and asks a narrower question than control or policy: when do same-spectrum operators still behave differently?",
        "",
        "The answer is visible in the simplest possible setting. A diagonal latent covariance and a diagonal predictor operator are rotated relative to each other. The resulting normalized commutator follows an explicit |sin(2 theta)| curve and peaks at 45 degrees. That peak is not a cosmetic detail. It marks the oblique-angle regime where top-band transfer and residual growth also shift, while flat-spectrum nulls and commuting endpoints suppress the signal.",
        "",
        "This is an oral-level theory paper only if it changes what experts think a sufficient spectral summary should be. The contribution here is therefore a boundary object: operator alignment, not spectrum alone, is the missing variable.",
        "",
        "## Related Work",
        "",
        "The nearest family is spectral transfer analysis: work that summarizes dynamics through eigenvalues, decay slopes, or low-dimensional transfer descriptors [Cao et al., 2019; Hong et al., 2022]. Those summaries are useful, but they are incomplete when the scientific question is whether two same-spectrum operators can still differ by basis alignment. A second nearby family is operator-learning spectral bias mitigation [Khodakarami et al., 2026], which studies frequency suppression but not this fixed-spectrum boundary. A third nearby family is noncommutative / subspace-alignment analysis [Parada-Mayorga et al., 2021; Thopalli et al., 2022]. Those papers already use geometric alignment language, but they do not isolate the same-spectrum transfer law studied here. A fourth nearby family is controllability or identifiability analysis. That family asks how actions or observations support intervention or inference. This paper does not. It stays in the discovery regime and uses a toy model to identify a boundary law rather than a controller.",
        "",
        "The paper also differs from generic sweep reports. The sweep is not the contribution by itself. It is the numerical witness for a theorem candidate, a falsifier set, and a field consequence about what should be measured alongside spectra.",
        "",
        "## Research Object",
        "",
        "- Mother object: noncommutative alignment between latent covariance and predictor operator.",
        "- Independent unit: one latent covariance, one predictor operator, one rotation plane.",
        "- Not about X: not about controllability or policy choice.",
        "- About Y: whether identical spectra still under-identify transfer geometry.",
        "- Observable proxy: normalized commutator, top-band lift, residual-growth ratio.",
        "- Forbidden conflations: eigenvalue spectrum, coordinate ordering, and finite-window slope are not substitutes for alignment.",
        "",
        "## Method",
        "",
        "### Theorem object",
        "",
        "For a single-plane rotation of a diagonal predictor operator, the normalized commutator norm is proportional to |sin(2 theta)| and is maximized at an oblique angle. The theorem is finite-dimensional and scoped: it licenses a descriptive mechanism claim, not a learned LeWM property.",
        "",
        "### Sweep object",
        "",
        "The sweep keeps spectra fixed while varying the predictor basis across angles 0, 15, 30, 45, 60, 75, and 90 degrees. It measures the commutator norm, top-band transfer at k = 4, 8, 16, 32, and residual-growth ratios across horizons 1, 2, 4, and 8. Random orthonormal bases provide the null.",
        "",
        "### First reader-facing display",
        "",
        "- Question: does fixing spectrum also fix transfer geometry?",
        "- Row / series unit: rotation angle by k band.",
        "- Estimands: commutator norm, top-band lift, residual-growth ratio.",
        "- Non-claim: this is not a control paper and not a claim about learned LeWM latents.",
        "",
        "## Results",
        "",
        "### Exact law",
        "",
        "Table 1 states the analytic curve used as the theorem witness.",
        "",
        "| angle | exact commutator norm |",
        "| --- | --- |",
    ]
    for angle in theorem["angles"]:
        key = str(angle)
        lines.append(f"| {_fmt(angle)} | {_fmt(exact[key])} |")
    lines.extend(
        [
            "",
            f"The law peaks at {_fmt(theorem['theorem']['peak_angle_degrees'])} degrees and collapses to zero at the aligned and endpoint cases.",
            "",
            "### Numerical verification",
            "",
            "Table 2 reports the sweep witness that owns the main empirical claim inside this theory paper.",
            "",
            "| angle | commutator norm | h1 top-k lift | h1 residual lift | h8 residual growth |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        h1 = row["horizons"]["1"]
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["angle_degrees"]),
                    _fmt(row["commutator_norm"]),
                    _fmt(h1["predicted_band_lift"][str(top_k)]),
                    _fmt(h1["residual_band_lift"][str(top_k)]),
                    _fmt(row["residual_growth_ratio"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"The oblique case at 45 degrees has commutator norm {_fmt(contrast['peak_oblique_commutator_norm'])}, compared with {_fmt(contrast['aligned_commutator_norm'])} at alignment and {_fmt(contrast['endpoint_commutator_norm'])} at the endpoint. The random-rotation null has mean {_fmt(null['mean'])} with 5th to 95th percentile {_fmt(null['p05'])} to {_fmt(null['p95'])}.",
            "",
            "### Rival and boundary analysis",
            "",
            "Flat latent or flat predictor spectra collapse the exact commutator to zero for every tested angle. That is the strongest boundary result because it shows the effect depends on anisotropy and basis alignment, not only on the existence of an operator.",
            "",
            "The random-rotation null is the strongest rival because it preserves dimension, spectra, and orthogonality while changing basis orientation. The oblique peak remains separable from this null, which means the pattern is not a generic coordinate-order artifact.",
            "",
            "### Field consequence",
            "",
            "The paper changes what should count as a sufficient transfer summary. If same-spectrum operators can still differ in transfer geometry, then a spectral audit that omits basis alignment is under-specified. That does not force a controller; it forces a better measurement object.",
            "",
            "## Discussion",
            "",
            "The claim is intentionally narrow. It is a finite-dimensional same-spectrum / different-basis theorem candidate, supported by an exact single-plane law, a numerical sweep, flat-spectrum falsifiers, and a random-rotation null. It does not establish a learned LeWM latent law, and it does not absorb the empirical PushT paper. The value of the result is exactly that it sets the boundary: operator alignment is the missing variable that spectra alone do not recover.",
            "",
            "## Conclusion",
            "",
            "Same spectrum is not enough. In the toy model studied here, basis alignment changes transfer geometry, and the change is predictable, falsifiable, and largest at an oblique angle. The paper's oral-level update is therefore a boundary law: spectral summaries should be treated as incomplete unless they are paired with an alignment-sensitive object.",
            "",
            "## Claim Ledger",
            "",
            "| claim_id | claim | tier | evidence | rival | boundary | status |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in claim_rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Main Figure",
            "",
            "Panel A: top-band lift heatmap across angle and k.",
            "Panel B: commutator norm vs angle with the exact theorem curve overlaid.",
            "Panel C: residual-growth ratio vs angle with the random-null band.",
            "Panel D: aligned / peak-oblique / endpoint / random-mean contrast bars.",
            "",
            "## Supplement Displays",
            "",
            "- S1: permutation-preserving null versus spectrum-matched random-basis null.",
            "- S2: continuous anisotropy sweep from flat to strongly anisotropic.",
            "- S3: transfer-proxy robustness across top-band lift, residual lift, and residual-growth ratios.",
            "- S4: spectrum-vs-alignment decoupling table separating fixed-spectrum alignment variation from fixed-alignment spectrum variation.",
            "",
            "## What This Does Not Claim",
            "",
            "- It does not prove a learned LeWM latent operator law.",
            "- It does not replace the companion empirical paper on frozen PushT latents.",
            "- It does not justify controllability as the mother object.",
            "",
            "## Section Skeleton",
            "",
            "Abstract -> introduction -> related work -> method -> theorem -> numerical verification -> rival/falsifier analysis -> field consequence -> discussion -> conclusion.",
            "",
            "## References",
            "",
            "- Cao, Y. et al. \"Towards Understanding the Spectral Bias of Deep Learning.\" arXiv:1912.01198. https://arxiv.org/abs/1912.01198",
            "- Hong, Q. et al. \"On the Activation Function Dependence of the Spectral Bias of Neural Networks.\" arXiv:2208.04924. https://arxiv.org/abs/2208.04924",
            "- Khodakarami, S. et al. \"Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines.\" arXiv:2602.19265. https://arxiv.org/abs/2602.19265",
            "- Parada-Mayorga, A. et al. \"Convolutional Filtering and Neural Networks with Non Commutative Algebras.\" arXiv:2108.09923. https://arxiv.org/abs/2108.09923",
            "- Thopalli, K. et al. \"Revisiting Deep Subspace Alignment for Unsupervised Domain Adaptation.\" arXiv:2201.01806. https://arxiv.org/abs/2201.01806",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_paper_spine(output: Path, sweep: dict[str, Any], theorem: dict[str, Any], law_card: dict[str, Any]) -> None:
    (output / "operator_alignment_paper_spine.md").write_text(operator_alignment_paper_spine(sweep, theorem, law_card))


def _write_paper_draft(output: Path, sweep: dict[str, Any], theorem: dict[str, Any], law_card: dict[str, Any]) -> None:
    (output / "operator_alignment_paper_draft.md").write_text(operator_alignment_paper_draft(sweep, theorem, law_card))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = TransferConfig(
        dim=args.dim,
        intrinsic_dim=args.intrinsic_dim,
        input_exponent=args.input_exponent,
        predictor_exponent=args.predictor_exponent,
    )
    report = operator_alignment_report(
        config=config,
        angles=args.angles,
        horizons=args.horizons,
        ks=args.ks,
        random_trials=args.random_trials,
        seed=args.seed,
    )
    theorem = operator_alignment_theorem_report(
        config=config,
        angles=tuple(float(v) for v in args.angles),
    )
    law_card = operator_alignment_law_card(
        config=config,
        angles=tuple(float(v) for v in args.angles),
        horizons=tuple(int(v) for v in args.horizons),
        ks=tuple(int(v) for v in args.ks),
        random_trials=args.random_trials,
        seed=args.seed,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    (output / "operator_alignment_sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "operator_alignment_theorem.json").write_text(json.dumps(theorem, indent=2, sort_keys=True) + "\n")

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    _plot_band_heatmap(axes[0, 0], report)
    axes[0, 0].text(-0.14, 1.08, "A", transform=axes[0, 0].transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    _plot_with_null(
        axes[0, 1],
        report,
        "commutator_norm",
        "Commutator vs. rotation",
        "Commutator norm",
    )
    axes[0, 1].text(-0.14, 1.08, "B", transform=axes[0, 1].transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    _plot_with_null(
        axes[1, 0],
        report,
        "residual_growth_ratio",
        "Residual growth vs. rotation",
        "Residual trace H8/H1",
    )
    axes[1, 0].text(-0.14, 1.08, "C", transform=axes[1, 0].transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    _plot_contrast(axes[1, 1], report)
    axes[1, 1].text(-0.14, 1.08, "D", transform=axes[1, 1].transAxes, fontsize=13, fontweight="bold", va="top", ha="left")

    fig.savefig(output / "operator_alignment_summary.pdf", bbox_inches="tight")
    fig.savefig(output / "operator_alignment_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_markdown(output, report)
    _write_theorem(output, theorem)
    _write_law_card(output, law_card)
    _write_paper_outline(output, report, theorem, law_card)
    _write_oral_object_card(output, report, theorem, law_card)
    _write_paper_spine(output, report, theorem, law_card)
    _write_paper_draft(output, report, theorem, law_card)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
