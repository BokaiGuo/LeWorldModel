"""Summarize a paired TwoRoom intervention against a frozen baseline.

The intervention key can be either ``frozen_lewm`` (a teacher readout) or
``hybrid_rgb_fallback`` (a verified-local repair arm).  The script never
selects an arm from held-out outcomes; it only computes the paired consequence
for already-frozen artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest


def _load(path: Path, key: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get(key), dict):
        raise ValueError(f"{path}: missing result key {key!r}")
    result = payload[key]
    seeds = result.get("seeds", [])
    successes = result.get("episode_successes", [])
    if len(seeds) != len(successes):
        raise ValueError(f"{path}: seed and success lengths differ for {key}")
    return result


def summarize(baseline_path: Path, intervention_path: Path,
              intervention_key: str, baseline_key: str = "frozen_lewm") -> dict:
    baseline = _load(baseline_path, baseline_key)
    intervention = _load(intervention_path, intervention_key)
    if baseline["seeds"] != intervention["seeds"]:
        raise ValueError("baseline and intervention must use the same ordered seeds")
    base = [bool(value) for value in baseline["episode_successes"]]
    arm = [bool(value) for value in intervention["episode_successes"]]
    baseline_only = sum(x and not y for x, y in zip(base, arm))
    intervention_only = sum(y and not x for x, y in zip(base, arm))
    discordant = baseline_only + intervention_only
    return {
        "schema": "gdd_wm_paired_tworoom_repair_consequence.v0.1",
        "protocol": {
            "baseline": str(baseline_path.resolve()),
            "intervention": str(intervention_path.resolve()),
            "intervention_key": intervention_key,
            "baseline_key": baseline_key,
            "same_seed_panel": True,
            "outcome_used_for_selection": False,
        },
        "n": len(base),
        "success": {
            "baseline": sum(base),
            "intervention": sum(arm),
            "baseline_only": baseline_only,
            "intervention_only": intervention_only,
            "paired_delta_count": sum(arm) - sum(base),
            "paired_delta_rate": (sum(arm) - sum(base)) / max(len(base), 1),
            "mcnemar_exact_p": float(
                binomtest(min(baseline_only, intervention_only), discordant,
                          p=0.5).pvalue if discordant else 1.0
            ),
        },
        "runtime_diagnostics": intervention.get(
            "runtime_diagnostics", intervention.get("risk_gate", {})
        ),
        "claim_boundary": (
            "paired positive direction is not a universal repair claim; the "
            "held-out interval is reported without outcome-based selection"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--intervention", type=Path, required=True)
    parser.add_argument("--intervention-key", required=True)
    parser.add_argument(
        "--baseline-key",
        default="frozen_lewm",
        help="result arm from the matched baseline report (default: frozen_lewm)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.baseline, args.intervention, args.intervention_key, args.baseline_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
