"""Summarize temporal replay stability across reconstructed split seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.input) != len(args.label):
        raise ValueError("Provide one label per input")
    output = {
        "protocol": {
            "unit": "reconstructed split seed",
            "aggregation": "descriptive range across split seeds; no pooling as independent episodes",
            "split_ids": [],
        },
        "models": {},
        "evidence_boundary": (
            "Split-provenance robustness for fixed-checkpoint temporal replay. "
            "It does not certify the original training split or establish a universal theorem."
        ),
    }
    for label, path in zip(args.label, args.input):
        payload = json.loads(path.read_text())
        protocol = payload.get("protocol", {})
        seed = protocol.get("split_seed")
        split_id = str(seed) if seed is not None else protocol.get("split", path.stem)
        episodes = protocol.get("episodes_used", [])
        rows = {
            "source": str(path),
            "split_id": split_id,
            "episode_count": len(episodes),
            "episodes_used": episodes,
            "latent_prediction_error_by_horizon": {
                horizon: values["mean"]
                for horizon, values in payload["latent_prediction_error_by_horizon"].items()
            },
            "covariance_operator_commutator_by_horizon": payload.get(
                "covariance_operator_commutator_by_horizon", {}
            ),
        }
        output["protocol"]["split_ids"].append(split_id)
        output["models"].setdefault(label, []).append(rows)

    summary = {}
    for label, rows in output["models"].items():
        horizons = sorted(rows[0]["latent_prediction_error_by_horizon"], key=int)
        summary[label] = {"splits": len(rows), "horizons": {}}
        for horizon in horizons:
            values = [row["latent_prediction_error_by_horizon"][horizon] for row in rows]
            comm_values = [
                row["covariance_operator_commutator_by_horizon"].get(horizon)
                for row in rows
                if horizon in row["covariance_operator_commutator_by_horizon"]
            ]
            summary[label]["horizons"][horizon] = {
                "mse_mean": sum(values) / len(values),
                "mse_range": [min(values), max(values)],
                "commutator_range": [min(comm_values), max(comm_values)] if comm_values else None,
            }
        overlaps = []
        for i, left in enumerate(rows):
            left_set = set(left["episodes_used"])
            for right in rows[i + 1 :]:
                right_set = set(right["episodes_used"])
                union = left_set | right_set
                overlaps.append({
                    "split_a": left["split_id"],
                    "split_b": right["split_id"],
                    "intersection": len(left_set & right_set),
                    "jaccard": len(left_set & right_set) / len(union) if union else None,
                })
        summary[label]["holdout_overlap"] = overlaps
    output["summary"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    markdown = [
        "# PointMaze split robustness",
        "",
        "Each row is a fixed-checkpoint replay on a different reconstructed "
        "split seed. These are descriptive repeats and are not pooled as "
        "independent episodes.",
        "",
        "| Model | Horizon | Mean MSE across splits | Split range | Commutator range |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in summary.items():
        for horizon, row in values["horizons"].items():
            comm = row["commutator_range"]
            comm_text = "n/a" if comm is None else f"[{comm[0]:.6f}, {comm[1]:.6f}]"
            markdown.append(
                f"| {label} | {horizon} | {row['mse_mean']:.6f} | "
                f"[{row['mse_range'][0]:.6f}, {row['mse_range'][1]:.6f}] | {comm_text} |"
            )
    markdown.extend(["", "## Evidence boundary", "", output["evidence_boundary"]])
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
