"""Summarize the matched learned torque-effect proposal/verification experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.summarize_pusht_local_effect_repair import bootstrap, load, paired


MODELS = [
    "state_action",
    "object_centric",
    "history",
    "block_only",
    "block_only_specialist",
    "block_agent_hybrid",
    "block_only_torque_readout",
]
METRICS = ("success", "block_xy_distance", "angle_error", "agent_xy_distance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    rows = load(args.input)
    result = {}
    for model_index, model in enumerate(MODELS):
        subset = [row for row in rows if row["model"] == model]
        result[model] = {
            metric: bootstrap(
                np.asarray([
                    float(row["success"]) if metric == "success" else float(row[metric])
                    for row in subset
                ]),
                args.draws,
                args.seed + model_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        if model != "state_action":
            result[model]["paired_minus_state_action"] = {
                metric: paired(
                    rows,
                    model,
                    "state_action",
                    metric,
                    args.draws,
                    args.seed + 1000 + model_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }
    protocols = [json.loads(Path(path).read_text())["protocol"] for path in args.input]
    report = {
        "claim_boundary": (
            "Privileged simulator-state learned-adapter diagnostic. The added arm predicts one-step "
            "block dx, dy, and dtheta from contact-intervention data, proposes a local action fan, "
            "and verifies the top-k proposals with one current-state simulator step. It is not "
            "frozen-LeWM visual planning or general task-success evidence."
        ),
        "protocol": {
            "inputs": args.input,
            "models": MODELS,
            "paired_unit": "seed and episode index",
            "bootstrap_draws": args.draws,
            "source_protocols": protocols,
        },
        "by_model": result,
        "sanity_checks": {
            "all_finite": bool(all(
                np.isfinite(float(row[metric]))
                for row in rows
                for metric in ("block_xy_distance", "angle_error", "agent_xy_distance")
            )),
            "models_present": sorted({row["model"] for row in rows}) == sorted(MODELS),
            "same_case_count": len({
                sum(row["model"] == model for row in rows) for model in MODELS
            }) == 1,
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "torque_effect_readout_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# Learned torque-effect readout summary",
        "",
        "| model | success | block XY | angle | agent XY |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
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
    (output / "torque_effect_readout_summary.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
