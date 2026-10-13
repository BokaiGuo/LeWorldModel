"""Summarize the leakage-controlled external forward-dynamics readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete report: {path}")
    return payload


def _bootstrap(values: np.ndarray, seed: int, repeats: int = 2000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(repeats, values.size))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "episodes": int(values.size),
        "bootstrap_unit": "held-out episode",
        "bootstrap_repeats": repeats,
    }


def _value(report: dict[str, Any], basis: str, band: str, horizon: int, feature: str) -> dict[str, Any]:
    return report["crossfit"]["summary"][f"{basis}/{band}/h{horizon}/{feature}"]


def _paired_basis_contrast(
    report: dict[str, Any], basis_a: str, basis_b: str, band: str, horizon: int, feature: str
) -> dict[str, Any]:
    records = report["crossfit"]["records"]
    a = {
        int(row["episode_id"]): float(row["r2"])
        for row in records
        if row["basis"] == basis_a and row["band"] == band
        and int(row["horizon"]) == horizon and row["feature"] == feature
    }
    b = {
        int(row["episode_id"]): float(row["r2"])
        for row in records
        if row["basis"] == basis_b and row["band"] == band
        and int(row["horizon"]) == horizon and row["feature"] == feature
    }
    keys = sorted(set(a).intersection(b))
    if len(keys) < 2:
        raise ValueError("paired basis contrast needs at least two common episodes")
    return _bootstrap(np.asarray([a[k] - b[k] for k in keys]), 20261300 + horizon)


def summarize(input_dir: Path, output: Path) -> dict[str, Any]:
    matched = {
        mode: _load(input_dir / f"external_forward_dynamics_readout_matched_band_{mode}.json")
        for mode in ("public", "val_zscore", "raw_div100")
    }
    fixed = _load(input_dir / "external_forward_dynamics_target_public.json")
    ridge_reports = {
        "1e-3": _load(input_dir / "external_forward_dynamics_target_ridge1e-3.json"),
        "1e-2": fixed,
        "1e-1": _load(input_dir / "external_forward_dynamics_target_ridge1e-1.json"),
    }
    full = _load(input_dir / "external_forward_dynamics_readout_public.json")

    matched_top8_h8 = {}
    for mode, report in matched.items():
        matched_top8_h8[mode] = {
            basis: {
                "joint": _value(report, basis, "top8", 8, "state_plus_action"),
                "gain_over_action": _value(report, basis, "top8", 8, "state_plus_action_minus_action_future"),
            }
            for basis in report["protocol"]["bases"]
        }

    fixed_top8_h8 = {
        basis: {
            "action": _value(fixed, basis, "top8", 8, "action_future"),
            "joint": _value(fixed, basis, "top8", 8, "state_plus_action"),
            "gain_over_action": _value(fixed, basis, "top8", 8, "state_plus_action_minus_action_future"),
        }
        for basis in fixed["protocol"]["bases"]
    }
    fixed_contrasts = {
        f"dynamics_top8_minus_{basis}": _paired_basis_contrast(
            fixed, "dynamics_basis", basis, "top8", 8, "state_plus_action"
        )
        for basis in ("covariance_basis", "residual_basis", "random_basis")
    }
    ridge_sensitivity = {
        ridge: {
            "dynamics_joint": _value(report, "dynamics_basis", "top8", 8, "state_plus_action"),
            "dynamics_gain_over_action": _value(report, "dynamics_basis", "top8", 8, "state_plus_action_minus_action_future"),
            "covariance_joint": _value(report, "covariance_basis", "top8", 8, "state_plus_action"),
        }
        for ridge, report in ridge_reports.items()
    }
    summary = {
        "artifact": "forward_dynamics_readout_summary",
        "status": "complete",
        "source_reports": sorted(path.name for path in input_dir.glob("external_forward_dynamics*.json")),
        "protocol": {
            "external_episodes": fixed["source"]["episodes"],
            "external_windows": fixed["source"]["windows"],
            "target_modes": ["full", "matched_band", "dynamics_target"],
            "controls": ["past_action", "deranged_action", "random_basis", "covariance_basis", "residual_basis"],
        },
        "full_latent_negative_control": {
            "dynamics_top8_h8_joint": _value(full, "dynamics_basis", "top8", 8, "state_plus_action"),
            "dynamics_top8_h8_gain_over_action": _value(full, "dynamics_basis", "top8", 8, "state_plus_action_minus_action_future"),
        },
        "matched_band_top8_h8": matched_top8_h8,
        "fixed_dynamics_target_top8_h8": fixed_top8_h8,
        "fixed_dynamics_target_paired_contrasts": fixed_contrasts,
        "ridge_sensitivity": ridge_sensitivity,
        "claim_matrix": [
            {
                "claim": "A source-fitted dynamics top-8 state representation predicts the same source-defined dynamics target out of episode.",
                "evidence": "positive h=8 state-plus-action readout with paired basis contrasts and ridge sensitivity",
                "verdict": "supported as frozen predictive latent evidence",
            },
            {
                "claim": "The result establishes physical action controllability or planning gain.",
                "evidence": "targets are encoded latent increments and no environment intervention or task-success endpoint is evaluated",
                "verdict": "not supported",
            },
            {
                "claim": "The dynamics subspace predicts the complete external latent transition.",
                "evidence": "full 192-dimensional target readout has no stable positive state-plus-action gain",
                "verdict": "not supported; retain as a bounded band-level result",
            },
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, output.with_suffix(".md"))
    return summary


def _fmt(value: dict[str, Any]) -> str:
    return f"{value['mean']:.4f} [{value['ci95'][0]:.4f}, {value['ci95'][1]:.4f}]"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Forward readout summary for frozen latent dynamics",
        "",
        "This summary keeps the full-latent negative control separate from the fixed-target band-level result. Every interval is episode-bootstrap based and each probe leaves one complete episode out.",
        "",
        "## Fixed dynamics target (top-8), h=8",
        "",
        "| input basis | action R2 | state+action R2 | state+action minus action | dynamics-basis contrast |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    fixed = summary["fixed_dynamics_target_top8_h8"]
    contrasts = summary["fixed_dynamics_target_paired_contrasts"]
    for basis, value in fixed.items():
        contrast = (
            "--"
            if basis == "dynamics_basis"
            else _fmt(contrasts[f"dynamics_top8_minus_{basis}"])
            if f"dynamics_top8_minus_{basis}" in contrasts
            else "--"
        )
        lines.append(f"| {basis} | {_fmt(value['action'])} | {_fmt(value['joint'])} | {_fmt(value['gain_over_action'])} | {contrast} |")
    lines += [
        "",
        "## Normalization robustness: matched target band, top-8, h=8",
        "",
        "| normalization | dynamics joint R2 | dynamics gain | residual joint R2 | residual gain |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, values in summary["matched_band_top8_h8"].items():
        lines.append(
            f"| {mode} | {_fmt(values['dynamics_basis']['joint'])} | {_fmt(values['dynamics_basis']['gain_over_action'])} | "
            f"{_fmt(values['residual_basis']['joint'])} | {_fmt(values['residual_basis']['gain_over_action'])} |"
        )
    lines += [
        "",
        "## Ridge sensitivity: fixed dynamics target, top-8, h=8",
        "",
        "| ridge | dynamics joint R2 | dynamics gain | covariance joint R2 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for ridge, values in summary["ridge_sensitivity"].items():
        lines.append(f"| {ridge} | {_fmt(values['dynamics_joint'])} | {_fmt(values['dynamics_gain_over_action'])} | {_fmt(values['covariance_joint'])} |")
    lines += [
        "",
        "## Boundary",
        "",
        "- The supported result is a source-defined, frozen latent predictive readout; it is not a causal or physical-control result.",
        "- The complete 192-dimensional forward target remains a negative control and is not promoted to a universal transition-prediction claim.",
        "- Past-action and deranged-action controls test temporal alignment but do not establish online planning or task success.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input_dir, args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
