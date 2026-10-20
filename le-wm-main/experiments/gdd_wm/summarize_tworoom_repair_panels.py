"""Combine independent paired TwoRoom verified-local repair panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest


def summarize(paths: list[Path], output: Path) -> dict:
    if len(paths) < 2:
        raise ValueError("at least two independent panels are required")
    panels = []
    baseline = intervention = baseline_only = intervention_only = total = 0
    gate_calls = fallback_count = validation_fallback_count = effect_policy_count = 0
    phase_counts = {"door": 0, "goal": 0}
    weighted_cosine_sum = 0.0
    cosine_weight = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "gdd_wm_paired_tworoom_repair_consequence.v0.1":
            raise ValueError(f"unexpected schema: {path}")
        success = payload["success"]
        n = int(payload["n"])
        total += n
        baseline += int(success["baseline"])
        intervention += int(success["intervention"])
        baseline_only += int(success["baseline_only"])
        intervention_only += int(success["intervention_only"])
        diagnostics = payload.get("runtime_diagnostics", {})
        gate_calls += int(diagnostics.get("gate_calls", 0))
        fallback_count += int(diagnostics.get("fallback_count", 0))
        validation_fallback_count += int(
            diagnostics.get("effect_policy_validation_fallback_count", 0)
        )
        effect_policy_count += int(diagnostics.get("effect_policy_count", 0))
        for phase, count in diagnostics.get("phase_counts", {}).items():
            phase_counts[phase] = phase_counts.get(phase, 0) + int(count)
        if diagnostics.get("cosine_mean") is not None and diagnostics.get("gate_calls", 0):
            weighted_cosine_sum += float(diagnostics["cosine_mean"]) * int(
                diagnostics["gate_calls"]
            )
            cosine_weight += int(diagnostics["gate_calls"])
        panels.append({
            "path": str(path.resolve()),
            "n": n,
            "baseline": int(success["baseline"]),
            "intervention": int(success["intervention"]),
            "baseline_only": int(success["baseline_only"]),
            "intervention_only": int(success["intervention_only"]),
            "intervention_key": payload["protocol"]["intervention_key"],
            "outcome_used_for_selection": bool(
                payload["protocol"]["outcome_used_for_selection"]
            ),
            "fallback_rate": diagnostics.get("fallback_rate"),
            "gate_calls": int(diagnostics.get("gate_calls", 0)),
            "effect_policy_count": int(diagnostics.get("effect_policy_count", 0)),
            "effect_policy_validation_fallback_count": int(
                diagnostics.get("effect_policy_validation_fallback_count", 0)
            ),
            "cosine_mean": diagnostics.get("cosine_mean"),
            "cosine_p10": diagnostics.get("cosine_p10"),
            "phase_counts": diagnostics.get("phase_counts", {}),
        })
    discordant = baseline_only + intervention_only
    report = {
        "schema": "gdd_wm_paired_tworoom_repair_panels.v0.1",
        "protocol": {
            "independent_root_panels": True,
            "same_seed_within_panel": True,
            "outcome_used_for_selection": False,
        },
        "panels": panels,
        "combined": {
            "n": total,
            "baseline_successes": baseline,
            "intervention_successes": intervention,
            "baseline_only": baseline_only,
            "intervention_only": intervention_only,
            "success_delta": intervention - baseline,
            "exact_mcnemar_p": float(
                binomtest(min(baseline_only, intervention_only), discordant,
                          p=0.5).pvalue if discordant else 1.0
            ),
            "runtime_diagnostics": {
                "gate_calls": gate_calls,
                "fallback_count": fallback_count,
                "fallback_rate": fallback_count / max(gate_calls, 1),
                "effect_policy_count": effect_policy_count,
                "effect_policy_validation_fallback_count": validation_fallback_count,
                "phase_counts": phase_counts,
                "weighted_cosine_mean": weighted_cosine_sum / max(cosine_weight, 1),
            },
        },
        "claim_boundary": (
            "positive point estimate is a TwoRoom held-out repair result, not a "
            "universal transfer claim; no outcome-based panel selection was used"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.panel, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
