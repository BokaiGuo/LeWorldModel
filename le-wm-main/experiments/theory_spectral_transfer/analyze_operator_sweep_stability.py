"""Audit whether operator-control conclusions survive rank/ridge choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def margin_summary(values: list[float]) -> dict:
    return {
        "mean": sum(values) / len(values),
        "range": [min(values), max(values)],
        "positive_settings": sum(value > 0 for value in values),
        "settings": len(values),
        "all_positive": all(value > 0 for value in values),
    }


def main() -> None:
    args = parse_args()
    requested = set(args.model)
    found = {}
    protocols = []
    for path in args.input:
        payload = json.loads(path.read_text())
        protocols.append({"source": str(path), "protocol": payload["protocol"]})
        for label, model in payload["models"].items():
            if label in requested:
                found[label] = model
    missing = requested - set(found)
    if missing:
        raise ValueError(f"missing requested models: {sorted(missing)}")

    models = {}
    for label in args.model:
        horizons = {}
        for horizon, result in found[label]["horizons"].items():
            paired_shuffled = []
            paired_spectral = []
            paired_identity = []
            settings = []
            for setting, metrics in result["sweep"].items():
                paired = metrics["paired_r2"]["mean"]
                shuffled = metrics["shuffled_r2"]["mean"]
                spectral = metrics["spectral_null_r2"]["mean"]
                identity = metrics["identity_r2"]["mean"]
                paired_shuffled.append(paired - shuffled)
                paired_spectral.append(paired - spectral)
                paired_identity.append(paired - identity)
                settings.append({
                    "setting": setting,
                    "paired_r2": paired,
                    "identity_r2": identity,
                    "pair_shuffled_r2": shuffled,
                    "spectral_control_r2": spectral,
                })
            horizons[horizon] = {
                "paired_minus_pair_shuffled": margin_summary(paired_shuffled),
                "paired_minus_spectral_control": margin_summary(paired_spectral),
                "paired_minus_identity": margin_summary(paired_identity),
                "settings": settings,
            }
        models[label] = {"horizons": horizons}

    output = {
        "protocol": {
            "sources": protocols,
            "interpretation": {
                "paired_minus_pair_shuffled": "episode correspondence beyond a destroyed-pairing control",
                "paired_minus_spectral_control": "basis-specific structure beyond a spectrum-preserving random basis",
                "paired_minus_identity": "predictive structure beyond latent persistence",
            },
        },
        "models": models,
        "evidence_boundary": (
            "Rank/ridge robustness of held-out linear diagnostics. A positive paired-vs-shuffled margin does not "
            "by itself imply action causality; identity and spectral controls remain separate falsifiers."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

    markdown = [
        "# Operator rank/ridge stability audit",
        "",
        "Each cell summarizes every available rank/ridge setting in the source control sweep.",
        "",
        "| Model | Horizon | Paired>shuffled | Margin range | Paired>spectral | Paired>identity |",
        "|---|---:|:---:|---:|:---:|:---:|",
    ]
    for label, model in models.items():
        for horizon, values in model["horizons"].items():
            correspondence = values["paired_minus_pair_shuffled"]
            spectral = values["paired_minus_spectral_control"]
            identity = values["paired_minus_identity"]
            markdown.append(
                f"| {label} | {horizon} | "
                f"{correspondence['positive_settings']}/{correspondence['settings']} | "
                f"[{correspondence['range'][0]:+.3f},{correspondence['range'][1]:+.3f}] | "
                f"{spectral['positive_settings']}/{spectral['settings']} | "
                f"{identity['positive_settings']}/{identity['settings']} |"
            )
    markdown.extend(["", "## Evidence boundary", "", output["evidence_boundary"]])
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n")
    print(json.dumps({"json": str(args.output), "markdown": str(args.output.with_suffix('.md'))}, indent=2))


if __name__ == "__main__":
    main()
