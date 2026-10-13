"""Summarize cross-horizon transfer across independently fitted source bases."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.summarize_external_cross_horizon_robustness import (
    _contrast,
    _finite,
)


PATTERN = re.compile(r"external_cross_horizon_(?P<split>\d+)_(?P<normalization>[^_]+(?:_[^_]+)?)_w(?P<windows>\d+)\.json$")
FORWARD_CELLS = ("h1_to_h4", "h1_to_h8", "h4_to_h8")


def run_report(paths: list[Path], output: Path) -> dict[str, Any]:
    if len(paths) != 12:
        raise ValueError("source-split robustness requires exactly twelve reports")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in paths:
        match = PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected report name: {path.name}")
        split = match.group("split")
        normalization = match.group("normalization")
        windows_per_episode = int(match.group("windows"))
        report = json.loads(path.read_text())
        key = (split, normalization, windows_per_episode)
        if key in seen:
            raise ValueError(f"duplicate cell: {key}")
        seen.add(key)
        if report["status"] != "complete" or report["model"]["device"] != "cuda":
            raise ValueError(f"report is not a complete CUDA run: {path}")
        if report["source"]["episodes"] != 21 or report["source"]["windows"] not in (42, 164):
            raise ValueError(f"unexpected external sample count: {path}")
        if report["protocol"]["bootstrap_repeats"] != 2000:
            raise ValueError(f"unexpected bootstrap protocol: {path}")
        paired = report["crossfit"]["paired_contrasts"]
        row: dict[str, Any] = {
            "report": str(path),
            "source_split": split,
            "normalization": normalization,
            "windows_per_episode": windows_per_episode,
            "episodes": int(report["source"]["episodes"]),
            "windows": int(report["source"]["windows"]),
            "device": report["model"]["device"],
            "bootstrap_repeats": int(report["protocol"]["bootstrap_repeats"]),
            "dynamics_basis_sha256": report["source"]["dynamics_basis_sha256"],
            "residual_basis_sha256": report["source"]["residual_basis_sha256"],
            "contrasts": {},
        }
        for cell in FORWARD_CELLS:
            row["contrasts"][f"{cell}/dynamics_minus_random"] = _contrast(
                report, cell, "dynamics_band_increment", "dynamics_minus_random"
            )
        row["contrasts"]["h4_to_h8/dynamics_minus_covariance"] = _contrast(
            report,
            "h4_to_h8",
            "dynamics_band_increment",
            "dynamics_minus_covariance",
        )
        row["contrasts"]["h4_to_h8/residual_minus_dynamics"] = _contrast(
            report,
            "h4_to_h8",
            "residual_band_increment",
            "residual_minus_dynamics",
        )
        rows.append(row)

    expected = {
        (split, normalization, windows)
        for split in ("20261006", "20261007")
        for normalization in ("public", "val_zscore", "raw_div100")
        for windows in (2, 8)
    }
    if seen != expected:
        raise ValueError(f"incomplete source-split matrix: {sorted(seen)}")
    result = {
        "artifact": "external_frozen_lewm_cross_horizon_source_split_robustness",
        "status": "complete",
        "protocol": {
            "comparison": "same external episodes, frozen model, leave-one-episode-out readout, rank, ridge, horizons, normalization arms, and episode-cluster bootstrap; only the independently fitted source split and window density vary",
            "source_splits": ["20261006", "20261007"],
            "normalizations": ["public", "val_zscore", "raw_div100"],
            "windows_per_episode": [2, 8],
            "forward_cells": list(FORWARD_CELLS),
            "rank": 8,
            "episodes": 21,
            "bootstrap_repeats": 2000,
        },
        "rows": rows,
        "claim_boundary": [
            "Positive paired contrasts across both independently fitted source splits support source-fit robustness of the frozen cross-horizon latent readout.",
            "This remains predictive association in encoded latent targets; it does not establish physical causality, causal action effects, online planning, task success, retraining improvement, or a universal world-model property.",
            "The source-split check is complementary to, and not pooled with, the normalization-density matrix based on the canonical source basis.",
        ],
    }
    if not _finite(result):
        raise ValueError("source-split summary contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_markdown(result, output.with_suffix(".md"))
    return result


def _format(value: dict[str, Any]) -> str:
    lo, hi = value["ci95"]
    return f"{value['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Source-split robustness of cross-horizon latent readout",
        "",
        "The source dynamics and residual bases are independently fitted on split 20261006 or 20261007. External episodes, readout protocol, and bootstrap are held fixed; values are paired means with 95% episode-cluster intervals.",
        "",
        "| Source split | Normalization | Windows/episode | Episodes × windows | D--R 1→4 | D--R 1→8 | D--R 4→8 | D--C 4→8 | R--D 4→8 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(result["rows"], key=lambda item: (item["source_split"], item["normalization"], item["windows_per_episode"])):
        c = row["contrasts"]
        lines.append(
            f"| {row['source_split']} | {row['normalization']} | {row['windows_per_episode']} | {row['episodes']} × {row['windows']} | "
            f"{_format(c['h1_to_h4/dynamics_minus_random'])} | "
            f"{_format(c['h1_to_h8/dynamics_minus_random'])} | "
            f"{_format(c['h4_to_h8/dynamics_minus_random'])} | "
            f"{_format(c['h4_to_h8/dynamics_minus_covariance'])} | "
            f"{_format(c['h4_to_h8/residual_minus_dynamics'])} |"
        )
    lines.extend(["", "## Interpretation boundary", "", *[f"- {item}" for item in result["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs=12, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_report(args.reports, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
