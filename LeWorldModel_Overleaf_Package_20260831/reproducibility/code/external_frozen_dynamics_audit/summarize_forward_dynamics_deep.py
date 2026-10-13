"""Build a multi-horizon, control-complete summary of the forward readout.

This is deliberately a post-processing artifact: the expensive cross-fitted
readout is already stored in the source reports.  The summary exposes the
parts that are easy to hide in a single h=8 table: the horizon profile,
future/past/deranged action controls, the matched-band comparison, the
complete-latent negative control, and the separate physical-target negative
control.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HORIZONS = (1, 4, 8)
BASELINE_BASIS = ("dynamics_basis", "covariance_basis", "residual_basis", "random_basis")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete report: {path}")
    return payload


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def _bootstrap(values: Iterable[float], seed: int, repeats: int = 2000) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("bootstrap requires at least two finite values")
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, array.size, size=(repeats, array.size))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "episodes": int(array.size),
        "bootstrap_repeats": int(repeats),
        "bootstrap_unit": "held-out episode",
    }


def _records(report: dict[str, Any], basis: str, band: str, horizon: int, feature: str) -> dict[int, float]:
    rows = {
        int(row["episode_id"]): float(row["r2"])
        for row in report["crossfit"]["records"]
        if row["basis"] == basis
        and row["band"] == band
        and int(row["horizon"]) == int(horizon)
        and row["feature"] == feature
    }
    if len(rows) < 2:
        raise ValueError(f"not enough records for {basis}/{band}/h{horizon}/{feature}")
    return rows


def _metric(report: dict[str, Any], basis: str, band: str, horizon: int, feature: str) -> dict[str, Any]:
    values = _records(report, basis, band, horizon, feature)
    return _bootstrap(values.values(), seed=20262000 + horizon)


def _contrast(
    report: dict[str, Any],
    basis: str,
    band: str,
    horizon: int,
    feature_a: str,
    feature_b: str,
    seed: int,
) -> dict[str, Any]:
    a = _records(report, basis, band, horizon, feature_a)
    b = _records(report, basis, band, horizon, feature_b)
    keys = sorted(set(a).intersection(b))
    return _bootstrap((a[key] - b[key] for key in keys), seed=seed)


def _profile(report: dict[str, Any], basis: str, band: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        result[str(horizon)] = {
            "action": _metric(report, basis, band, horizon, "action_future"),
            "state_only": _metric(report, basis, band, horizon, "state_only"),
            "state_plus_action": _metric(report, basis, band, horizon, "state_plus_action"),
            "state_plus_action_gain": _contrast(
                report, basis, band, horizon, "state_plus_action", "action_future", 20262100 + horizon
            ),
            "past_action": _metric(report, basis, band, horizon, "action_past"),
            "deranged_action": _metric(report, basis, band, horizon, "action_deranged"),
            "future_minus_past": _contrast(
                report, basis, band, horizon, "action_future", "action_past", 20262200 + horizon
            ),
            "future_minus_deranged": _contrast(
                report, basis, band, horizon, "action_future", "action_deranged", 20262300 + horizon
            ),
            "state_plus_future_minus_state_plus_past": _contrast(
                report, basis, band, horizon, "state_plus_action", "state_plus_past", 20262500 + horizon
            ),
            "state_plus_future_minus_state_plus_deranged": _contrast(
                report, basis, band, horizon, "state_plus_action", "state_plus_deranged", 20262600 + horizon
            ),
        }
    return result


def _physical_negative(physical_report: dict[str, Any]) -> dict[str, Any]:
    key = "dynamics_basis/top8/h8"
    try:
        entry = physical_report["crossfit"]["summary"]["by_basis_band_horizon"][key]
        return {
            "target": "normalized physical displacement",
            "basis": "dynamics_basis/top8",
            "horizon": 8,
            "predicted_incremental_r2_over_action": entry["predicted_incremental_r2"],
            "predicted_increment_r2": entry["predicted_increment_r2"],
            "action_only_r2": entry["action_only_r2"],
        }
    except KeyError as exc:
        raise KeyError(f"physical report missing {key}") from exc


def summarize(
    input_dir: Path,
    output: Path,
    physical_report: Path | None = None,
) -> dict[str, Any]:
    fixed = _load(input_dir / "external_forward_dynamics_target_public.json")
    matched = _load(input_dir / "external_forward_dynamics_readout_matched_band_public.json")
    full = _load(input_dir / "external_forward_dynamics_readout_public.json")

    summary: dict[str, Any] = {
        "artifact": "forward_dynamics_deep_summary",
        "status": "complete",
        "source_reports": [
            fixed["source"].get("basis_artifact", ""),
            "external_forward_dynamics_target_public.json",
            "external_forward_dynamics_readout_matched_band_public.json",
            "external_forward_dynamics_readout_public.json",
        ],
        "protocol": {
            "episodes": int(fixed["source"]["episodes"]),
            "windows": int(fixed["source"]["windows"]),
            "horizons": list(HORIZONS),
            "bootstrap_unit": "held-out episode",
            "controls": ["action_only", "state_only", "future_action", "past_action", "deranged_action"],
            "target_levels": ["fixed dynamics top-8", "matched band", "full 192-dimensional latent", "physical displacement"],
        },
        "fixed_dynamics_target_horizon_profile": {
            basis: _profile(fixed, basis, "top8") for basis in BASELINE_BASIS
        },
        "matched_band_dynamics_horizon_profile": {
            band: _profile(matched, "dynamics_basis", band) for band in ("top8", "middle8_32")
        },
        "full_latent_negative_control": {
            str(horizon): {
                "state_plus_action": _metric(full, "dynamics_basis", "top8", horizon, "state_plus_action"),
                "state_plus_action_gain": _contrast(
                    full, "dynamics_basis", "top8", horizon, "state_plus_action", "action_future", 20262400 + horizon
                ),
            }
            for horizon in HORIZONS
        },
        "claim_matrix": [
            {
                "claim": "The source-fitted dynamics band contains forward-predictive information for an encoded latent target.",
                "evidence": "leave-one-episode-out horizon profile, action-only/state-only controls, and future/past/deranged controls",
                "verdict": "supported as a frozen latent diagnostic",
            },
            {
                "claim": "The effect is a universal full-latent transition model.",
                "evidence": "the complete 192-dimensional target is included as a negative control",
                "verdict": "not supported",
            },
            {
                "claim": "The effect is additional physical-motion predictability beyond actions.",
                "evidence": "the separate physical-displacement probe is action-dominated and has a non-positive incremental readout",
                "verdict": "not supported by this audit",
            },
        ],
    }
    if physical_report is not None:
        summary["physical_target_negative_control"] = _physical_negative(_load(physical_report))
        summary["source_reports"].append(str(physical_report))
    if not _finite(summary):
        raise ValueError("deep summary contains non-finite values")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, output.with_suffix(".md"))
    return summary


def _fmt(metric: dict[str, Any]) -> str:
    return f"{metric['mean']:+.3f} [{metric['ci95'][0]:+.3f},{metric['ci95'][1]:+.3f}]"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Deep forward-dynamics evidence summary",
        "",
        "All probes are fit with a complete held-out episode, and the future latent target is never an input. This report separates encoded-latent predictive evidence from physical-motion and full-latent negative controls.",
        "",
        "## Fixed dynamics target: horizon profile",
        "",
        "| basis | h | action | state-only | state+action | gain over action | future-past (joint) | future-deranged (joint) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    profile = summary["fixed_dynamics_target_horizon_profile"]
    for basis in BASELINE_BASIS:
        for horizon in HORIZONS:
            row = profile[basis][str(horizon)]
            lines.append(
                f"| {basis} | {horizon} | {_fmt(row['action'])} | {_fmt(row['state_only'])} | "
                f"{_fmt(row['state_plus_action'])} | {_fmt(row['state_plus_action_gain'])} | "
                f"{_fmt(row['state_plus_future_minus_state_plus_past'])} | "
                f"{_fmt(row['state_plus_future_minus_state_plus_deranged'])} |"
            )
    lines += [
        "",
        "## Matched-band specificity",
        "",
        "| input band | h=1 gain | h=4 gain | h=8 gain |",
        "| --- | ---: | ---: | ---: |",
    ]
    for band, values in summary["matched_band_dynamics_horizon_profile"].items():
        lines.append(
            f"| {band} | {_fmt(values['1']['state_plus_action_gain'])} | "
            f"{_fmt(values['4']['state_plus_action_gain'])} | {_fmt(values['8']['state_plus_action_gain'])} |"
        )
    lines += [
        "",
        "## Full-latent negative control",
        "",
        "| target | h | state+action R2 | gain over action |",
        "| --- | ---: | ---: | ---: |",
    ]
    for horizon, values in summary["full_latent_negative_control"].items():
        lines.append(
            f"| full 192-d latent increment | {horizon} | {_fmt(values['state_plus_action'])} | "
            f"{_fmt(values['state_plus_action_gain'])} |"
        )
    if "physical_target_negative_control" in summary:
        physical = summary["physical_target_negative_control"]
        lines += [
            "",
            "## Physical-target negative control",
            "",
            f"The dynamics top-8 predicted increment has incremental $R^2$ over the compact action summary of {_fmt(physical['predicted_incremental_r2_over_action'])} for the normalized physical displacement target at $h=8$.",
        ]
    lines += [
        "",
        "## Claim boundary",
        "",
        "- The supported object is a source-defined, band-level predictive diagnostic for encoded latent motion.",
        "- Future/past/deranged controls test temporal alignment; they do not establish a causal action effect.",
        "- The full latent and physical-target controls prevent promotion to a universal transition or physical-control claim.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-report", type=Path)
    args = parser.parse_args()
    summarize(args.input_dir, args.output, args.physical_report)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
