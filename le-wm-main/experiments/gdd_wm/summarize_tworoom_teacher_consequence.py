"""Summarize a paired CUDA TwoRoom teacher consequence without test tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["frozen_lewm"]
    if len(result["seeds"]) != len(result["episode_successes"]):
        raise ValueError(f"{path}: seeds and successes have different lengths")
    return result


def summarize(baseline_path: Path, intervention_path: Path) -> dict:
    baseline = _load(baseline_path)
    intervention = _load(intervention_path)
    if baseline["seeds"] != intervention["seeds"]:
        raise ValueError("baseline and intervention must use the same ordered seeds")
    b = [bool(value) for value in baseline["episode_successes"]]
    i = [bool(value) for value in intervention["episode_successes"]]
    baseline_only = sum(x and not y for x, y in zip(b, i))
    intervention_only = sum(y and not x for x, y in zip(b, i))
    discordant = baseline_only + intervention_only
    return {
        "schema": "gdd_wm_paired_tworoom_consequence_summary.v0.1",
        "claim_boundary": (
            "TwoRoom cross-environment consequence is not an established gain "
            "unless the predeclared paired interval excludes zero"
        ),
        "n": len(b),
        "protocol": {
            "baseline": str(baseline_path.resolve()),
            "intervention": str(intervention_path.resolve()),
            "same_seed_panel": True,
            "outcome_used_for_selection": False,
        },
        "success": {
            "baseline": sum(b),
            "intervention": sum(i),
            "baseline_only": baseline_only,
            "intervention_only": intervention_only,
            "paired_delta_count": sum(i) - sum(b),
            "paired_delta_rate": (sum(i) - sum(b)) / max(len(b), 1),
            "mcnemar_exact_p": float(
                binomtest(min(baseline_only, intervention_only), discordant, p=0.5).pvalue
                if discordant
                else 1.0
            ),
        },
        "runtime_diagnostics": intervention.get("runtime_diagnostics", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--intervention", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.baseline, args.intervention)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
