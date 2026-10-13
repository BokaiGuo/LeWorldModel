"""Summarize cross-fitted operator controls across split seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--split-seed", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (len(args.input) == len(args.label) == len(args.split_seed)):
        raise ValueError("Each input needs a label and split seed")
    output = {
        "protocol": {
            "primary_setting": {"rank": 8, "ridge": 1e-2},
            "aggregation": "descriptive range across reconstructed split seeds; no pooling as independent episodes",
        },
        "models": {},
        "evidence_boundary": (
            "Split-level robustness of cross-fitted operator controls on released checkpoints; "
            "the original training split remains uncertified where no serialized manifest ships."
        ),
    }
    for path, label, split_seed in zip(args.input, args.label, args.split_seed):
        payload = json.loads(path.read_text())
        model = next(iter(payload["models"].values()))
        rows = output["models"].setdefault(label, [])
        horizons = {}
        for horizon, result in model["horizons"].items():
            primary = result["primary"]
            horizons[horizon] = {
                "paired_r2": primary["paired_r2"]["mean"],
                "identity_r2": primary["identity_r2"]["mean"],
                "shuffled_r2": primary["shuffled_r2"]["mean"],
                "spectral_null_r2": primary["spectral_null_r2"]["mean"],
                "paired_minus_shuffled_r2": primary["paired_minus_shuffled_r2"]["mean_difference"],
                "paired_minus_spectral_null_r2": primary["paired_minus_spectral_null_r2"]["mean_difference"],
            }
        rows.append({"source": str(path), "split_seed": split_seed, "horizons": horizons})

    summary = {}
    for label, rows in output["models"].items():
        horizons = sorted(rows[0]["horizons"], key=int)
        summary[label] = {"splits": len(rows), "horizons": {}}
        for horizon in horizons:
            summary[label]["horizons"][horizon] = {}
            for metric in rows[0]["horizons"][horizon]:
                values = [row["horizons"][horizon][metric] for row in rows]
                summary[label]["horizons"][horizon][metric] = {
                    "mean": sum(values) / len(values),
                    "range": [min(values), max(values)],
                }
    output["summary"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    markdown = [
        "# PointMaze split-seed operator-control robustness",
        "",
        "Primary cross-fitted setting: rank 8, ridge 1e-2. Values are descriptive "
        "ranges across reconstructed split seeds; split seeds are not pooled as "
        "independent episodes.",
        "",
        "| Model | Horizon | Paired R² range | Identity R² range | Shuffled R² range | Paired−shuffled range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, model in summary.items():
        for horizon, values in model["horizons"].items():
            def fmt(name: str) -> str:
                row = values[name]["range"]
                return f"[{row[0]:.3f}, {row[1]:.3f}]"
            markdown.append(
                f"| {label} | {horizon} | {fmt('paired_r2')} | {fmt('identity_r2')} | "
                f"{fmt('shuffled_r2')} | {fmt('paired_minus_shuffled_r2')} |"
            )
    markdown.extend(["", "## Evidence boundary", "", output["evidence_boundary"]])
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
