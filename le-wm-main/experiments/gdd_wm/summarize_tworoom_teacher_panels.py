"""Combine predeclared paired TwoRoom teacher-consequence panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest


def summarize(paths: list[Path], output: Path) -> dict:
    if len(paths) < 2:
        raise ValueError("at least two independent paired panels are required")
    panels = []
    baseline = intervention = intervention_only = baseline_only = 0
    total = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "gdd_wm_paired_tworoom_consequence_summary.v0.1":
            raise ValueError(f"unexpected paired-summary schema: {path}")
        success = payload["success"]
        n = int(payload["n"])
        total += n
        baseline += int(success["baseline"])
        intervention += int(success["intervention"])
        intervention_only += int(success["intervention_only"])
        baseline_only += int(success["baseline_only"])
        panels.append({
            "path": str(path.resolve()),
            "n": n,
            "baseline": int(success["baseline"]),
            "intervention": int(success["intervention"]),
            "intervention_only": int(success["intervention_only"]),
            "baseline_only": int(success["baseline_only"]),
            "outcome_used_for_selection": bool(
                payload["protocol"]["outcome_used_for_selection"]
            ),
        })
    discordant = intervention_only + baseline_only
    exact_p = float(
        binomtest(
            min(intervention_only, baseline_only),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )
    report = {
        "schema": "gdd_wm_paired_tworoom_teacher_panels.v0.1",
        "panels": panels,
        "protocol": {
            "same_frozen_teacher_artifact": True,
            "independent_root_panels": True,
            "outcome_used_for_selection": False,
        },
        "combined": {
            "n": total,
            "baseline_successes": baseline,
            "intervention_successes": intervention,
            "intervention_only": intervention_only,
            "baseline_only": baseline_only,
            "success_delta": intervention - baseline,
            "exact_mcnemar_p": exact_p,
        },
        "claim_boundary": (
            "positive point estimate across independent panels, but no established "
            "closed-loop gain unless the predeclared paired interval excludes zero"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
