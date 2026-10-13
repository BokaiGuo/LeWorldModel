"""SIGReg-by-alignment phase diagram on the learned synthetic pixel world."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

try:
    from experiments.empirical_spectral_audit.run_pixel_world_model_synthetic import (
        PixelBenchmarkConfig,
        run_one,
    )
except ModuleNotFoundError:  # direct script invocation
    from run_pixel_world_model_synthetic import PixelBenchmarkConfig, run_one


def _finite(value: Any) -> bool:
    import math

    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def _mean_std(values: list[float]) -> dict[str, float]:
    import torch

    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=len(values) > 1)),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"sigreg={record['sigreg_weight']}:mode={record['mode']}:alignment={record['alignment_weight']}"
        groups.setdefault(key, []).append(record)
    result: dict[str, Any] = {}
    for key, rows in groups.items():
        result[key] = {
            "n_seeds": len(rows),
            "sigreg_weight": float(rows[0]["sigreg_weight"]),
            "mode": rows[0]["mode"],
            "alignment_weight": float(rows[0]["alignment_weight"]),
            "test_one_step_mse": _mean_std([float(row["test_one_step_mse"]) for row in rows]),
            "rollout_h8_mse": _mean_std([float(row["rollout_mse"]["8"]) for row in rows]),
            "composition_error": _mean_std([float(row["action_composition_error_mean"]) for row in rows]),
            "global_commutator": _mean_std([float(row["diagnostics"]["global_commutator"]) for row in rows]),
            "conditional_commutator": _mean_std([float(row["diagnostics"]["conditional_commutator"]) for row in rows]),
            "stable_rank": _mean_std([float(row["diagnostics"]["stable_rank"]) for row in rows]),
        }
    return result


def run_phase(
    sigreg_weights: list[float],
    seeds: list[int],
    alignment_weights: list[float],
    epochs: int = 120,
) -> dict[str, Any]:
    base_config = PixelBenchmarkConfig(epochs=epochs)
    records: list[dict[str, Any]] = []
    started = time.time()
    for sigreg_weight in sigreg_weights:
        config = replace(base_config, sigreg_weight=float(sigreg_weight))
        conditions = [("none", 0.0)]
        conditions.extend(("global", float(weight)) for weight in alignment_weights)
        conditions.extend(("conditional", float(weight)) for weight in alignment_weights)
        for mode, alignment_weight in conditions:
            for seed in seeds:
                print(
                    f"[pixel-phase] sigreg={sigreg_weight} mode={mode} "
                    f"alignment={alignment_weight} seed={seed}"
                )
                record = run_one(mode, alignment_weight, seed, config)
                record["sigreg_weight"] = float(sigreg_weight)
                records.append(record)
    report = {
        "protocol": {
            **asdict(base_config),
            "sigreg_weights": [float(value) for value in sigreg_weights],
            "alignment_weights": [float(value) for value in alignment_weights],
            "seeds": [int(seed) for seed in seeds],
            "modes": ["none", "global", "conditional"],
            "same_generated_protocol": True,
            "learned_encoder_and_predictor": True,
            "synthetic_pixel_world_only": True,
            "elapsed_seconds": time.time() - started,
        },
        "records": records,
        "summary": _summary(records),
        "claim_boundary": [
            "phase diagram measures diagnostic and predictive trade-offs in a synthetic learned pixel world",
            "lower commutator is not assumed to imply better rollout or composition",
            "no nonlinear LeWM, PushT, online planning, or causal-control evidence",
        ],
    }
    if not _finite(report):
        raise ValueError("phase report contains non-finite values")
    return report


def _write_markdown(report: dict[str, Any], output: Path) -> None:
    lines = [
        "# SIGReg × alignment phase diagram in the learned pixel world",
        "",
        "Every cell uses the same generated protocol and random seeds across conditions. The table reports mean ± standard deviation over seeds.",
        "",
        "| SIGReg | mode | alignment weight | seeds | one-step MSE | rollout h=8 | composition | global commutator | conditional commutator | stable rank |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for values in report["summary"].values():
        fmt = lambda name: f"{values[name]['mean']:.6g} ± {values[name]['std']:.2g}"
        lines.append(
            f"| {values['sigreg_weight']:.6g} | {values['mode']} | {values['alignment_weight']:.6g} | "
            f"{values['n_seeds']} | {fmt('test_one_step_mse')} | {fmt('rollout_h8_mse')} | "
            f"{fmt('composition_error')} | {fmt('global_commutator')} | "
            f"{fmt('conditional_commutator')} | {fmt('stable_rank')} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This phase diagram tests the SIGReg/isotropy confound in a learned pixel-world bridge. It does not establish that any cell improves nonlinear LeWM or PushT planning.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigreg-weights", type=float, nargs="+", default=[0.0, 0.01, 0.09])
    parser.add_argument("--alignment-weights", type=float, nargs="+", default=[0.01, 0.1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    report = run_phase(args.sigreg_weights, args.seeds, args.alignment_weights, args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, args.output.with_suffix(".md"))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
