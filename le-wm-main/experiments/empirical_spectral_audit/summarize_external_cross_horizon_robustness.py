"""Summarize normalization and window-density robustness of cross-horizon readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


FORWARD_CELLS = ("h1_to_h4", "h1_to_h8", "h4_to_h8")
TARGETS = ("dynamics_band_increment", "residual_band_increment")
CONTRASTS = ("dynamics_minus_random", "dynamics_minus_covariance", "residual_minus_dynamics")


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def _ci(value: dict[str, Any]) -> list[float]:
    return [float(value["ci95"][0]), float(value["ci95"][1])]


def _contrast(report: dict[str, Any], cell: str, target: str, contrast: str) -> dict[str, Any]:
    key = f"{cell}/{target}/k8/{contrast}"
    value = report["crossfit"]["paired_contrasts"][key]
    return {
        "mean": float(value["mean"]),
        "ci95": _ci(value),
        "n_windows": int(value["n_windows"]),
        "clusters": int(value["clusters"]),
    }


def _readout(report: dict[str, Any], cell: str, target: str, basis: str) -> dict[str, Any]:
    key = f"{cell}/{target}/{basis}/k8"
    value = report["crossfit"]["summary"][key]["state_incremental_r2_over_action"]
    return {"mean": float(value["mean"]), "ci95": _ci(value)}


def run_report(paths: list[Path], output: Path) -> dict[str, Any]:
    if len(paths) != 9:
        raise ValueError("the robustness matrix requires exactly nine reports")
    reports = [json.loads(path.read_text()) for path in paths]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for path, report in zip(paths, reports):
        protocol = report["protocol"]
        normalization = protocol["action_normalization"]
        mode = normalization if isinstance(normalization, str) else normalization["mode"]
        windows = int(report["source"]["windows"])
        episodes = int(report["source"]["episodes"])
        # Every record is repeated over targets, bases, and ranks; recover the
        # window density from unique episode/start pairs rather than dividing
        # the total by 21 (short episodes contribute fewer than eight windows).
        starts_by_episode: dict[int, set[int]] = {}
        for record in report["crossfit"]["records"]:
            starts_by_episode.setdefault(int(record["episode_id"]), set()).add(
                int(record["start_observation"])
            )
        key = (str(mode), max(len(starts) for starts in starts_by_episode.values()))
        if key in seen:
            raise ValueError(f"duplicate matrix cell: {key}")
        seen.add(key)
        if report["status"] != "complete" or report["model"]["device"] != "cuda":
            raise ValueError(f"report is not a complete CUDA run: {path}")
        if episodes != 21 or int(protocol["bootstrap_repeats"]) != 2000:
            raise ValueError(f"protocol mismatch in {path}")
        row: dict[str, Any] = {
            "report": str(path),
            "normalization": str(mode),
            "windows_per_episode": key[1],
            "episodes": episodes,
            "windows": windows,
            "device": report["model"]["device"],
            "bootstrap_repeats": int(protocol["bootstrap_repeats"]),
            "contrasts": {},
            "readouts": {},
        }
        for cell in FORWARD_CELLS:
            for target in TARGETS:
                row["readouts"][f"{cell}/{target}/dynamics"] = _readout(
                    report, cell, target, "dynamics_basis"
                )
                row["readouts"][f"{cell}/{target}/residual"] = _readout(
                    report, cell, target, "residual_basis"
                )
                for contrast in CONTRASTS:
                    row["contrasts"][f"{cell}/{target}/{contrast}"] = _contrast(
                        report, cell, target, contrast
                    )
        rows.append(row)

    expected = {(mode, windows) for mode in ("public", "val_zscore", "raw_div100") for windows in (2, 4, 8)}
    if seen != expected:
        raise ValueError(f"incomplete matrix: {sorted(seen)}")
    result = {
        "artifact": "external_frozen_lewm_cross_horizon_normalization_density_robustness",
        "status": "complete",
        "protocol": {
            "comparison": "same frozen source bases, external validation episodes, leave-one-episode-out fits, ridge, horizons, ranks, and episode-cluster bootstrap; action normalization and windows per episode vary",
            "normalizations": ["public", "val_zscore", "raw_div100"],
            "windows_per_episode": [2, 4, 8],
            "forward_cells": list(FORWARD_CELLS),
            "targets": list(TARGETS),
            "contrasts": list(CONTRASTS),
            "rank": 8,
            "episodes": 21,
            "bootstrap_repeats": 2000,
        },
        "rows": rows,
        "claim_boundary": [
            "Across the declared normalization and window-density matrix, the result supports robust cross-horizon predictive transfer of source-fitted latent subspaces.",
            "The evidence is a frozen, cross-fitted latent readout diagnostic and does not establish physical causality, causal action effects, online planning, task success, retraining improvement, or a universal world-model property.",
            "Residual specificity is target-dependent: residual coordinates preferentially predict residual-band increments, while dynamics coordinates preferentially predict dynamics-band increments.",
        ],
    }
    if not _finite(result):
        raise ValueError("robustness summary contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_markdown(result, output.with_suffix(".md"))
    return result


def _format(value: dict[str, Any]) -> str:
    lo, hi = value["ci95"]
    return f"{value['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Cross-horizon normalization and window-density robustness",
        "",
        "All rows use frozen source bases, leave-one-episode-out fits, rank 8, 21 episodes, and 2,000 episode-cluster bootstrap replicates. Values are means with 95% bootstrap intervals.",
        "",
        "| Normalization | Windows/episode | Episodes × windows | $1\\to4$ dynamics−random | $1\\to8$ dynamics−random | $4\\to8$ dynamics−random | $4\\to8$ residual−dynamics |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(result["rows"], key=lambda item: (item["normalization"], item["windows_per_episode"])):
        c = row["contrasts"]
        lines.append(
            f"| {row['normalization']} | {row['windows_per_episode']} | {row['episodes']} × {row['windows']} | "
            f"{_format(c['h1_to_h4/dynamics_band_increment/dynamics_minus_random'])} | "
            f"{_format(c['h1_to_h8/dynamics_band_increment/dynamics_minus_random'])} | "
            f"{_format(c['h4_to_h8/dynamics_band_increment/dynamics_minus_random'])} | "
            f"{_format(c['h4_to_h8/residual_band_increment/residual_minus_dynamics'])} |"
        )
    lines.extend(["", "## Interpretation boundary", "", *[f"- {item}" for item in result["claim_boundary"]]])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs=9, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_report(args.reports, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
