"""Aggregate root-disjoint paired-effect PushT control runs.

The panel is frozen after the three paired-probe runs.  It reads closed-loop
outcomes only for post-hoc evaluation, never for choosing a controller or
threshold.  Both root-level and seed-level CUDA bootstrap intervals are
reported so an effect-prediction improvement cannot be mistaken for planning
success.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_orientation_response import require_cuda


VARIANTS = (
    "global_one_step",
    "mode_aware_lookahead",
    "contact_chart_lookahead",
    "uncertainty_gated",
    "safe_fallback",
    "adaptive_online",
)
METRICS = (
    "pose_success",
    "contact_loss_rate",
    "center_distance_loss_rate",
    "final_angle",
    "predicted_observed_effect_error",
    "wrong_sign_events",
    "online_identified_rate",
    "identification_probe_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261023)
    return parser.parse_args()


def bootstrap(values: list[float], device: torch.device, draws: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n": 0}
    tensor = torch.as_tensor(array, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    indices = torch.randint(array.size, (int(draws), array.size), generator=generator, device=device)
    means = tensor[indices].mean(dim=1)
    result = {
        "estimate": float(tensor.mean().item()),
        "ci95_low": float(torch.quantile(means, 0.025).item()),
        "ci95_high": float(torch.quantile(means, 0.975).item()),
        "n": int(array.size),
    }
    del indices, means, tensor
    torch.cuda.synchronize()
    return result


def raw_root_metrics(loop: dict[str, Any]) -> dict[str, list[float]]:
    result = {metric: [] for metric in METRICS}
    for row in loop.get("rows", []):
        trace = list(row.get("trace", []))
        effect_values = [
            float(item["predicted_observed_effect_error"])
            for item in trace if item.get("predicted_observed_effect_error") is not None
        ]
        result["pose_success"].append(float(bool(row.get("pose_success", False))))
        # The formal panel uses boundary-contact roots.  Centre distance is
        # not a valid contact predicate there: a legal edge contact can be
        # more than 80 px from the block centre.  Prefer the protocol-correct
        # boundary metric, while retaining the legacy centre predicate as an
        # explicit diagnostic.
        result["contact_loss_rate"].append(float(any(
            float(item.get("boundary_distance_after", 0.0)) > 70.0 for item in trace
        )))
        result["center_distance_loss_rate"].append(float(any(
            float(item.get("contact_distance_after", 0.0)) > 80.0 for item in trace
        )))
        result["final_angle"].append(float(row.get("final_angle", np.nan)))
        result["predicted_observed_effect_error"].append(
            float(np.mean(effect_values)) if effect_values else np.nan
        )
        result["wrong_sign_events"].append(float(row.get("wrong_sign_events", 0.0)))
        result["online_identified_rate"].append(float(np.mean([
            bool(item.get("online_identified_used", False)) for item in trace
        ])) if trace else 0.0)
        result["identification_probe_rate"].append(float(np.mean([
            bool(item.get("identification_probe", False)) for item in trace
        ])) if trace else 0.0)
    return result


def main() -> None:
    args = parse_args()
    device = require_cuda()
    if len(args.artifact) < 2:
        raise ValueError("at least two independent formal artifacts are required")

    reports = [json.loads(path.read_text()) for path in args.artifact]
    per_seed: list[dict[str, Any]] = []
    raw_by_variant: dict[str, dict[str, list[float]]] = {
        variant: {metric: [] for metric in METRICS} for variant in VARIANTS
    }
    for index, (path, report) in enumerate(zip(args.artifact, reports)):
        closed_loop = report.get("closed_loop", {})
        missing = [variant for variant in VARIANTS if variant not in closed_loop]
        if missing:
            raise ValueError(f"artifact missing variants {missing}: {path}")
        seed = int(report["protocol"]["train_seed"]) - 1000
        seed_entry: dict[str, Any] = {
            "seed": seed,
            "artifact": str(path.resolve()),
            "held_out_roots": int(report["protocol"]["held_out_roots"]),
            "root_disjoint": bool(report["sanity_checks"]["root_disjoint"]),
            "variants": {},
        }
        for variant in VARIANTS:
            raw = raw_root_metrics(closed_loop[variant])
            seed_entry["variants"][variant] = {
                metric: bootstrap(values, device, int(args.bootstrap), args.seed + index * 1000 + metric_index)
                for metric_index, (metric, values) in enumerate(raw.items())
            }
            for metric, values in raw.items():
                raw_by_variant[variant][metric].extend(values)
        per_seed.append(seed_entry)

    pooled: dict[str, Any] = {}
    for variant_index, variant in enumerate(VARIANTS):
        pooled[variant] = {
            metric: bootstrap(values, device, int(args.bootstrap), args.seed + 20000 + variant_index * 100)
            for metric, values in raw_by_variant[variant].items()
        }

    seed_level: dict[str, Any] = {}
    for variant_index, variant in enumerate(VARIANTS):
        seed_level[variant] = {}
        for metric_index, metric in enumerate(METRICS):
            values = [
                float(item["variants"][variant][metric]["estimate"])
                for item in per_seed
                if item["variants"][variant][metric].get("estimate") is not None
            ]
            seed_level[variant][metric] = bootstrap(
                values, device, int(args.bootstrap), args.seed + 30000 + variant_index * 100 + metric_index
            )

    paired: dict[str, Any] = {}
    for variant in ("mode_aware_lookahead", "uncertainty_gated", "adaptive_online"):
        paired[variant] = {}
        for metric_index, metric in enumerate(METRICS):
            deltas: list[float] = []
            for report in reports:
                candidate = raw_root_metrics(report["closed_loop"][variant])[metric]
                baseline = raw_root_metrics(report["closed_loop"]["global_one_step"])[metric]
                n = min(len(candidate), len(baseline))
                deltas.extend(float(candidate[i] - baseline[i]) for i in range(n))
            paired[variant][f"{metric}_delta_vs_global"] = bootstrap(
                deltas, device, int(args.bootstrap), args.seed + 40000 + metric_index
            )

    report = {
        "artifact": "pusht_paired_effect_control_panel",
        "claim_boundary": (
            "Root-disjoint paired local-effect control panel. Lower effect error is a "
            "mechanism result only; planning success requires independent pose and "
            "contact-retention improvement. No test outcome was used to choose the "
            "paired-probe protocol."
        ),
        "protocol": {
            "analysis_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "n_independent_seeds": len(reports),
            "outcome_used_for_selection": False,
            "root_level_bootstrap": True,
            "seed_level_bootstrap": True,
            "fixed_protocol": {
                "root_protocol": "boundary_contact",
                "train_roots": 24,
                "held_out_roots": 16,
                "lookahead": 2,
                "paired_effect_probes": True,
                "online_effect_correction": True,
            },
        },
        "per_seed": per_seed,
        "pooled_root_level": pooled,
        "seed_level": seed_level,
        "paired_delta_vs_global": paired,
        "sanity_checks": {
            "all_root_disjoint": bool(all(item["root_disjoint"] for item in per_seed)),
            "same_variant_set": True,
            "cuda_analysis_completed": True,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "paired_effect_panel.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.4f}"

    lines = [
        "# PushT paired local-effect control panel", "",
        f"- Seeds: `{len(reports)}`; analysis device: `{report['protocol']['analysis_device']}` ({report['protocol']['cuda_device_name']}).", "",
        "| variant | pose success | contact loss | centre-distance diagnostic | effect error | wrong-sign events |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        values = pooled[variant]
        lines.append(
            f"| {variant} | {fmt(values['pose_success']['estimate'])} | "
            f"{fmt(values['contact_loss_rate']['estimate'])} | "
            f"{fmt(values['center_distance_loss_rate']['estimate'])} | "
            f"{fmt(values['predicted_observed_effect_error']['estimate'])} | "
            f"{fmt(values['wrong_sign_events']['estimate'])} |"
        )
    lines += ["", report["claim_boundary"], ""]
    (args.output / "paired_effect_panel.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "pooled_root_level": pooled, "seed_level": seed_level, "paired_delta_vs_global": paired}, indent=2))


if __name__ == "__main__":
    main()
