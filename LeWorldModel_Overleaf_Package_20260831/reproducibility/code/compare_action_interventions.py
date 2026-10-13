"""Create a side-by-side cross-episode versus within-episode action audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross", type=Path, required=True)
    parser.add_argument("--within", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cross = json.loads(args.cross.read_text())
    within = json.loads(args.within.read_text())
    if set(cross["models"]) != set(within["models"]):
        raise ValueError("cross and within reports must contain the same models")
    output = {
        "protocol": {
            "comparison": "cross_episode_derangement_vs_within_episode_future_block_derangement",
            "interpretation": (
                "Within-episode intervention preserves episode identity and future-action block multiset; "
                "cross-episode intervention additionally changes episode/action marginal context."
            ),
        },
        "models": {},
        "evidence_boundary": (
            "Sensitivity comparison for fixed-checkpoint replay diagnostics; not causal task performance "
            "or a universal learned-world-model claim."
        ),
    }
    for label in cross["models"]:
        cm = cross["models"][label]
        wm = within["models"][label]
        rows = {}
        for horizon in cm["horizons"]:
            c = cm["horizons"][horizon]
            w = wm["horizons"][horizon]
            rows[horizon] = {
                "cross_episode": {
                    "seed_mean_delta": c["seed_mean_delta"],
                    "seed_range": [c["seed_min_delta"], c["seed_max_delta"]],
                    "all_seed_deltas_positive": c["all_seed_deltas_positive"],
                },
                "within_episode": {
                    "seed_mean_delta": w["seed_mean_delta"],
                    "seed_range": [w["seed_min_delta"], w["seed_max_delta"]],
                    "all_seed_deltas_positive": w["all_seed_deltas_positive"],
                },
            }
        output["models"][label] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    markdown = [
        "# Action intervention comparison",
        "",
        "Cross-episode derangement changes episode/action context. Within-episode "
        "future-block derangement keeps episode identity and the exact future-action "
        "block multiset, changing only temporal order. Seeds are not pooled as "
        "independent episodes.",
        "",
        "| Model | Horizon | Cross-episode mean ΔMSE [range] | Within-episode mean ΔMSE [range] |",
        "|---|---:|---:|---:|",
    ]
    for label, rows in output["models"].items():
        for horizon, values in rows.items():
            c, w = values["cross_episode"], values["within_episode"]
            markdown.append(
                f"| {label} | {horizon} | {c['seed_mean_delta']:+.6f} "
                f"[{c['seed_range'][0]:+.6f},{c['seed_range'][1]:+.6f}] | "
                f"{w['seed_mean_delta']:+.6f} "
                f"[{w['seed_range'][0]:+.6f},{w['seed_range'][1]:+.6f}] |"
            )
    markdown.extend(["", "## Evidence boundary", "", output["evidence_boundary"]])
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
