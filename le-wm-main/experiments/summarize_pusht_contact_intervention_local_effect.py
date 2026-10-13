"""Summarize contact-intervention training with local-effect orientation control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.summarize_pusht_local_effect_repair import (
    bootstrap,
    load,
    paired,
    phase_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()

    rows = load(args.input)
    models = ["state_action", "object_centric", "history", "block_only", "block_only_specialist", "block_agent_hybrid"]
    baseline = "state_action"
    result = {}
    for index, model in enumerate(models):
        subset = [row for row in rows if row["model"] == model]
        result[model] = {
            metric: bootstrap(
                np.asarray([
                    float(row["success"]) if metric == "success" else float(row[metric])
                    for row in subset
                ]),
                args.draws,
                args.seed + index * 100 + offset,
            )
            for offset, metric in enumerate(("success", "block_xy_distance", "angle_error", "agent_xy_distance"))
        }
        if model != baseline:
            result[model]["paired_minus_baseline"] = {
                metric: paired(
                    rows, model, baseline, metric, args.draws,
                    args.seed + 1000 + index * 100 + offset,
                )
                for offset, metric in enumerate(("success", "block_xy_distance", "angle_error", "agent_xy_distance"))
            }

    protocols = []
    for path in args.input:
        report = json.loads(Path(path).read_text())
        protocols.append(report["protocol"])
    report = {
        "claim_boundary": (
            "Privileged simulator-state learned-adapter diagnostic. Training uses contact interventions and "
            "near-goal local-effect probes use current-state simulator successors; this is not frozen-LeWM "
            "visual planning or general PushT success."
        ),
        "protocol": {
            "inputs": args.input,
            "training_protocol": "contact_intervention",
            "planning_objective": "staged_contact_local_effect",
            "cases": len([row for row in rows if row["model"] == baseline]),
            "seeds": sorted({row["seed"] for row in rows}),
            "bootstrap_draws": args.draws,
            "paired_unit": "seed and episode index",
            "source_protocols": protocols,
        },
        "by_model": result,
        "phase_audit": phase_audit(rows),
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(float(row[metric]))
                for row in rows
                for metric in ("block_xy_distance", "angle_error", "agent_xy_distance")
            )),
            "models_present": sorted({row["model"] for row in rows}) == sorted(models),
            "same_case_count": len({sum(row["model"] == model for row in rows) for model in models}) == 1,
        },
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "contact_intervention_local_effect_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# Contact-intervention plus local-effect summary",
        "",
        "| model | success | block XY | angle | agent XY |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in models:
        item = result[model]
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] |"
            % (
                model,
                item["success"]["estimate"], item["success"]["ci95_low"], item["success"]["ci95_high"],
                item["block_xy_distance"]["estimate"], item["block_xy_distance"]["ci95_low"], item["block_xy_distance"]["ci95_high"],
                item["angle_error"]["estimate"], item["angle_error"]["ci95_low"], item["angle_error"]["ci95_high"],
                item["agent_xy_distance"]["estimate"], item["agent_xy_distance"]["ci95_low"], item["agent_xy_distance"]["ci95_high"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "contact_intervention_local_effect_summary.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
