"""Summarize compact versus richer action-history controls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FORWARD_CELLS = ("h1_to_h4", "h1_to_h8", "h4_to_h8")
CONTROLS = (
    "compact6",
    "block_mean_history_padded",
    "normalized_time_dct16",
)


def _parse(path: Path, source_split: str | None) -> dict[str, Any]:
    match = re.match(
        r"(?:canonical_|split_(?P<split>\d+)_)(?P<normalization>[^_]+(?:_[^_]+)?)_w(?P<windows>\d+)\.json$",
        path.name,
    )
    if not match:
        raise ValueError(f"unexpected artifact name: {path.name}")
    payload = json.loads(path.read_text())
    crossfit = payload["crossfit"]
    row: dict[str, Any] = {
        "artifact": str(path),
        "source_split": source_split or "canonical",
        "normalization": match.group("normalization"),
        "windows_per_episode": int(match.group("windows")),
        "episodes": int(payload["source"]["episodes"]),
        "windows": int(payload["source"]["windows"]),
        "device": payload["model"]["device"],
        "bootstrap_repeats": int(payload["protocol"]["bootstrap_repeats"]),
        "controls": {},
        "condition_summary": crossfit["condition_summary"],
    }
    for control in CONTROLS:
        values: dict[str, Any] = {}
        for cell in FORWARD_CELLS:
            prefix = f"{control}/{cell}/dynamics_band_increment"
            values[cell] = {
                "dynamics_minus_random": crossfit["paired_contrasts"][
                    f"{prefix}/k8/dynamics_minus_random"
                ],
                "dynamics_minus_covariance": crossfit["paired_contrasts"][
                    f"{prefix}/k8/dynamics_minus_covariance"
                ],
                "dynamics_incremental_r2": crossfit["summary"][
                    f"{prefix}/dynamics_basis/k8"
                ]["state_incremental_r2_over_action"],
            }
        row["controls"][control] = values
    row["rich_minus_compact"] = {
        cell: crossfit["rich_minus_compact"][
            f"{cell}/dynamics_band_increment/dynamics_basis/k8/rich_minus_compact"
        ]
        for cell in FORWARD_CELLS
    }
    row["control_minus_compact"] = {
        control: {
            cell: crossfit["control_minus_compact"][
                f"{control}/{cell}/dynamics_band_increment/dynamics_basis/k8/rich_minus_compact"
            ]
            for cell in FORWARD_CELLS
        }
        for control in CONTROLS
        if control != "compact6"
    }
    return row


def _all_lower(rows: list[dict[str, Any]], control: str, contrast: str) -> float:
    return min(
        row["controls"][control][cell][contrast]["ci95"][0]
        for row in rows
        for cell in FORWARD_CELLS
    )


def summarize(canonical_dir: Path, source_dir: Path, output: Path) -> dict[str, Any]:
    canonical = [
        _parse(path, None)
        for path in sorted(canonical_dir.glob("canonical_*.json"))
    ]
    source = [
        _parse(path, re.match(r"split_(\d+)_", path.name).group(1))
        for path in sorted(source_dir.glob("split_*.json"))
    ]
    rows = canonical + source
    expected_canonical = {
        ("canonical", normalization, windows)
        for normalization in ("public", "val_zscore", "raw_div100")
        for windows in (2, 8)
    }
    expected_source = {
        (split, normalization, windows)
        for split in ("20261006", "20261007")
        for normalization in ("public", "val_zscore", "raw_div100")
        for windows in (2, 8)
    }
    observed = {
        (row["source_split"], row["normalization"], row["windows_per_episode"])
        for row in rows
    }
    expected = expected_canonical | expected_source
    if observed != expected:
        raise ValueError(f"incomplete matrix: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    for row in rows:
        if row["device"] != "cuda" or row["episodes"] != 21 or row["bootstrap_repeats"] != 2000:
            raise ValueError(f"invalid protocol metadata: {row}")
    summary = {
        "artifact": "external_frozen_lewm_cross_horizon_action_control_robustness",
        "status": "complete",
        "canonical_rows": canonical,
        "source_split_rows": source,
        "rows": rows,
        "matrix": {
            "canonical_cells": len(canonical),
            "source_split_cells": len(source),
            "total_cells": len(rows),
            "forward_cells": list(FORWARD_CELLS),
            "controls": list(CONTROLS),
            "all_dynamics_minus_random_lower": {
                control: _all_lower(rows, control, "dynamics_minus_random")
                for control in CONTROLS
            },
            "all_dynamics_minus_covariance_lower": {
                control: _all_lower(rows, control, "dynamics_minus_covariance")
                for control in CONTROLS
            },
            "all_control_minus_compact_lower": {
                control: {
                    cell: min(
                        row["control_minus_compact"][control][cell]["ci95"][0]
                        for row in rows
                    )
                    for cell in FORWARD_CELLS
                }
                for control in CONTROLS
                if control != "compact6"
            },
        },
        "claim_boundary": [
            "The richer control retains every observed per-block action mean and zero-pads the suffix to a common h=8 feature dimension.",
            "Positive latent incremental R2 is predictive association beyond the declared observed-action history, not a causal action effect.",
            "The audit does not establish physical counterfactual prediction, online planning, task success, retraining improvement, or a universal world model.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Cross-horizon action-history control robustness",
        "",
        "The matrix contains six canonical cells and twelve independent source-fit cells. Each row uses CUDA, 21 external episodes, leave-one-episode-out fitting, and 2,000 episode-cluster bootstrap replicates. Forward cells are h1->h4, h1->h8, and h4->h8.",
        "",
        "| source | normalization | windows | compact6 D--R | padded-history D--R | normalized-DCT D--R | normalized-DCT D--C |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        compact = min(row["controls"]["compact6"][cell]["dynamics_minus_random"]["ci95"][0] for cell in FORWARD_CELLS)
        padded = min(row["controls"]["block_mean_history_padded"][cell]["dynamics_minus_random"]["ci95"][0] for cell in FORWARD_CELLS)
        dct = min(row["controls"]["normalized_time_dct16"][cell]["dynamics_minus_random"]["ci95"][0] for cell in FORWARD_CELLS)
        dct_dc = min(row["controls"]["normalized_time_dct16"][cell]["dynamics_minus_covariance"]["ci95"][0] for cell in FORWARD_CELLS)
        lines.append(f"| {row['source_split']} | {row['normalization']} | {row['windows_per_episode']} | {compact:+.4f} | {padded:+.4f} | {dct:+.4f} | {dct_dc:+.4f} |")
    lines.extend([
        "",
        "## Matrix-level minima",
        "",
        f"- Compact6 dynamics-minus-random minimum lower bound: {summary['matrix']['all_dynamics_minus_random_lower']['compact6']:+.4f}.",
        f"- Padded-history dynamics-minus-random minimum lower bound: {summary['matrix']['all_dynamics_minus_random_lower']['block_mean_history_padded']:+.4f}.",
        f"- Normalized-time DCT dynamics-minus-random minimum lower bound: {summary['matrix']['all_dynamics_minus_random_lower']['normalized_time_dct16']:+.4f}.",
        f"- Normalized-time DCT dynamics-minus-covariance minimum lower bound: {summary['matrix']['all_dynamics_minus_covariance_lower']['normalized_time_dct16']:+.4f}.",
        "- Richer-history minus compact6 is a paired sensitivity diagnostic; it is not required to be positive because the purpose is to test whether the latent contrast survives a stronger control.",
        "",
        "## Interpretation boundary",
        "",
        "The richer action history is an observed-action predictive control. These results support robustness of frozen latent predictive specificity beyond that control, not causal action influence or end-to-end world-model performance.",
    ])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.canonical_dir, args.source_dir, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
