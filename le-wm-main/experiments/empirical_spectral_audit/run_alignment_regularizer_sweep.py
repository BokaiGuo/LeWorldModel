"""Synthetic mechanism check for the opt-in operator-alignment regularizer.

The sweep keeps the prediction rule's eigenvalues fixed and changes only its
eigenbasis.  It also includes a flat-covariance control, which is essential:
the commutator cannot identify orientation when the covariance itself is
isotropic.  This is a mechanism artifact, not a learned LeWM result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

try:
    from experiments.empirical_spectral_audit.regularizers import operator_alignment_loss
except ModuleNotFoundError:  # direct ``python experiments/.../script.py`` invocation
    from regularizers import operator_alignment_loss


def _rotated_operator(dim: int, angle_degrees: float) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    c, s = math.cos(angle), math.sin(angle)
    rotation = torch.eye(dim)
    rotation[0, 0] = c
    rotation[0, 1] = -s
    rotation[1, 0] = s
    rotation[1, 1] = c
    spectrum = torch.linspace(0.9, 0.2, dim)
    return rotation @ torch.diag(spectrum) @ rotation.T


def _row(record: dict[str, float | int | str]) -> dict[str, float | int | str]:
    return {key: (float(value) if isinstance(value, float) else value) for key, value in record.items()}


def run_sweep(samples: int = 4096, dim: int = 8, seed: int = 0) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    angles = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
    rows: list[dict[str, float | int | str]] = []
    anisotropic_scales = torch.linspace(4.0, 0.5, dim)
    for covariance_name, scales in (
        ("anisotropic", anisotropic_scales),
        ("isotropic_control", torch.ones(dim)),
    ):
        source = torch.randn(samples, dim, generator=generator) * scales
        for angle in angles:
            operator = _rotated_operator(dim, angle)
            predicted = source @ operator.T
            loss = operator_alignment_loss(
                source,
                predicted,
                mode="global",
                min_samples=32,
                ridge=1e-5,
            )
            rows.append(
                _row(
                    {
                        "covariance": covariance_name,
                        "angle_degrees": angle,
                        "prediction_mse": float((predicted - source).square().mean()),
                        "alignment_loss": float(loss),
                    }
                )
            )
    anisotropic = [row for row in rows if row["covariance"] == "anisotropic"]
    isotropic = [row for row in rows if row["covariance"] == "isotropic_control"]
    peak = max(anisotropic, key=lambda row: float(row["alignment_loss"]))
    return {
        "protocol": {
            "kind": "synthetic_mechanism_smoke",
            "seed": seed,
            "samples": samples,
            "dim": dim,
            "angles_degrees": list(angles),
            "operator_eigenvalues_fixed": True,
            "learned_training": False,
            "interpretation": "regularizer mechanism and isotropy identifiability boundary only",
        },
        "rows": rows,
        "summary": {
            "anisotropic_peak_angle_degrees": peak["angle_degrees"],
            "anisotropic_peak_alignment_loss": peak["alignment_loss"],
            "anisotropic_endpoint_mean": float(
                sum(float(row["alignment_loss"]) for row in anisotropic if row["angle_degrees"] in {0.0, 90.0}) / 2.0
            ),
            "isotropic_max_alignment_loss": max(float(row["alignment_loss"]) for row in isotropic),
        },
    }


def write_markdown(report: dict[str, object], output: Path) -> None:
    summary = report["summary"]
    rows = report["rows"]
    lines = [
        "# Alignment-Regularized LeWM: synthetic mechanism smoke",
        "",
        "This artifact tests the opt-in training regularizer with fixed predictor eigenvalues. It is not a learned LeWM result.",
        "",
        "## Protocol",
        "",
        f"- samples: `{report['protocol']['samples']}`; dimension: `{report['protocol']['dim']}`; seed: `{report['protocol']['seed']}`",
        "- operator eigenvalues are fixed while the eigenspace is rotated in the first two coordinates",
        "- prediction MSE is reported only to verify that the spectral rotation is not being treated as a training comparison",
        "- the isotropic control is an identifiability boundary, not a negative result about learned models",
        "",
        "## Summary",
        "",
        f"- anisotropic peak angle: `{summary['anisotropic_peak_angle_degrees']} degrees`",
        f"- anisotropic peak alignment loss: `{summary['anisotropic_peak_alignment_loss']:.6g}`",
        f"- anisotropic endpoint mean: `{summary['anisotropic_endpoint_mean']:.6g}`",
        f"- isotropic-control maximum: `{summary['isotropic_max_alignment_loss']:.6g}`",
        "",
        "## Alignment loss by covariance condition",
        "",
        "| covariance | angle (degrees) | prediction MSE | normalized alignment loss |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['covariance']} | {row['angle_degrees']:.0f} | "
            f"{row['prediction_mse']:.6g} | {row['alignment_loss']:.6g} |"
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_sweep(samples=args.samples, dim=args.dim, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
