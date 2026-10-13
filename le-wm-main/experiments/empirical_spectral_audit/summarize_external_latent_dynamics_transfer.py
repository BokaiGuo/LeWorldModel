"""Summarize cross-normalization latent-dynamics transfer reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(value: dict[str, Any]) -> str:
    return f"{value['mean']:.6g} [{value['ci95'][0]:.6g}, {value['ci95'][1]:.6g}]"


def _verdict(value: dict[str, Any]) -> str:
    lower, upper = value["ci95"]
    if lower > 0:
        return "supported"
    if upper < 0:
        return "reversed"
    return "inconclusive"


def run_report(inputs: list[Path], output: Path) -> dict[str, Any]:
    if len(inputs) != 3:
        raise ValueError("expected public, validation-zscore, and raw/100 reports")
    reports = {path.stem.rsplit("_", 1)[-1]: json.loads(path.read_text()) for path in inputs}
    # Normalize the public filename, whose suffix is the experiment name.
    reports = {
        ("public" if "_val_zscore" not in path.name and "_raw_div100" not in path.name else
         "val_zscore" if "_val_zscore" in path.name else "raw_div100"): json.loads(path.read_text())
        for path in inputs
    }
    rows: list[dict[str, Any]] = []
    for normalization, report in reports.items():
        contrasts = report["crossfit"]["basis_contrasts"]
        summary = report["crossfit"]["summary"]["by_basis_band_horizon_feature"]
        for horizon in (1, 4, 8):
            for control in ("covariance_basis", "random_basis"):
                key = f"dynamics_basis_minus_{control}/top8/h{horizon}/rollout_increment_only"
                value = contrasts[key]
                pred_key = f"dynamics_basis/top8/h{horizon}/rollout_increment_only"
                rows.append(
                    {
                        "normalization": normalization,
                        "horizon": horizon,
                        "contrast": f"dynamics minus {control.replace('_basis', '')}",
                        "contrast_value": value,
                        "predicted_r2": summary[pred_key]["predicted_r2"],
                        "verdict": _verdict(value),
                    }
                )
    report = {
        "artifact": "external_latent_dynamics_transfer_cross_normalization_summary",
        "status": "complete",
        "inputs": [str(path) for path in inputs],
        "protocol": {
            "external_episodes": 21,
            "external_windows": 164,
            "basis": "internal frozen dynamics basis; no external refitting",
            "target": "future latent increment in transferred top-8 band",
            "evaluation": "leave-one-episode-out with episode-cluster bootstrap",
        },
        "rows": rows,
        "claim_matrix": [
            {
                "claim": "The transferred dynamics basis transports latent increments better than covariance and random top-8 controls.",
                "evidence": "paired predicted-R2 contrasts across horizons and normalization arms",
                "verdict": "supported in public normalization; partial under validation z-score; not supported under raw/100",
            },
            {
                "claim": "The dynamics basis gives normalization-invariant external rollout evidence.",
                "evidence": "raw/100 contrasts include zero at h=1,4,8",
                "verdict": "not supported",
            },
            {
                "claim": "The result identifies physical causal dynamics or planning benefit.",
                "evidence": "latent predictive transport only",
                "verdict": "not tested; no such claim",
            },
        ],
        "claim_boundary": [
            "This is a frozen external latent predictive/transport diagnostic.",
            "The basis is transferred from an independent internal export and never refit on external windows.",
            "Action normalization is not fully matched to the original LeWM training statistics.",
            "The result does not establish physical counterfactual prediction, causal control, online planning, task success, or retraining improvement.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Cross-normalization latent-dynamics transfer summary",
        "",
        "The same external 21-episode, 164-window protocol was evaluated under public mean/std, validation-split z-score, and raw/100 action normalization. Values are paired predicted-R2 differences for the transferred dynamics top-8 band versus a dimension-matched control.",
        "",
        "| normalization | horizon | contrast | predicted-R2 difference | verdict |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['normalization']} | {row['horizon']} | {row['contrast']} | "
            f"{_fmt(row['contrast_value'])} | {row['verdict']} |"
        )
    lines.extend(["", "## Claim matrix", ""])
    for item in report["claim_matrix"]:
        lines.append(f"- **{item['claim']}** {item['verdict']} ({item['evidence']}).")
    lines.extend(["", "## Boundary", "", *[f"- {item}" for item in report["claim_boundary"]]])
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_report(args.input, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
