"""Aggregate the frozen large-action PushT transfer panel.

This aggregator is evaluation-only.  It does not select a seed, action bank,
or controller from held-out outcomes.  It reports pooled root-level and
seed-level uncertainty plus within-artifact paired deltas against the global
one-step controller.
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


BASE_VARIANTS = (
    "global_one_step",
    "staged_translation_then_orientation",
    "staged_phase_gate_two_pulse_macro",
    "shielded_phase_gate_two_pulse_macro",
    "orientation_then_translation_two_pulse_macro",
    "joint_two_pulse_macro",
    "uncertainty_gated",
    "safe_fallback",
)
EXTRA_VARIANTS = ("orientation_ranker_two_pulse_macro",)
EXTRA_DELTA_VARIANTS = (
    "orientation_delta_ranker_two_pulse_macro",
    "orientation_delta_ranker_shielded_two_pulse_macro",
)
EXTRA_DUAL_VARIANTS = ("orientation_dual_ranker_two_pulse_macro",)
EXTRA_PAIR_VARIANTS = (
    "orientation_pair_ranker_two_pulse_macro",
    "orientation_pair_xy_translation_two_pulse_macro",
    "orientation_pair_reverse_phase_gate_two_pulse_macro",
    "orientation_pair_reverse_phase_gate_shielded_two_pulse_macro",
)
METRICS = (
    "pose_success",
    "final_xy",
    "final_angle",
    "contact_loss_rate",
    "predicted_observed_effect_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261042)
    parser.add_argument(
        "--include-orientation-ranker", action="store_true",
        help="include the new train-only orientation-effect ranker arm",
    )
    parser.add_argument(
        "--include-orientation-delta-ranker", action="store_true",
        help="include the signed after-minus-before orientation-delta ranker arm",
    )
    parser.add_argument(
        "--include-orientation-dual-ranker", action="store_true",
        help="include the frozen equal-weight absolute-plus-delta ranker arm",
    )
    parser.add_argument(
        "--include-orientation-pair-ranker", action="store_true",
        help="include the mirrored-action angular-contrast ranker arm",
    )
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


def raw_metrics(loop: dict[str, Any]) -> dict[str, list[float]]:
    result = {metric: [] for metric in METRICS}
    for row in loop.get("rows", []):
        trace = list(row.get("trace", []))
        errors = [
            float(item["predicted_observed_effect_error"])
            for item in trace if item.get("predicted_observed_effect_error") is not None
        ]
        result["pose_success"].append(float(bool(row.get("pose_success", False))))
        result["final_xy"].append(float(row.get("final_xy", np.nan)))
        result["final_angle"].append(float(row.get("final_angle", np.nan)))
        result["contact_loss_rate"].append(float(any(
            float(item.get("boundary_distance_after", 0.0)) > 70.0 for item in trace
        )))
        result["predicted_observed_effect_error"].append(
            float(np.mean(errors)) if errors else np.nan
        )
    return result


def main() -> None:
    args = parse_args()
    variants = BASE_VARIANTS
    if args.include_orientation_ranker:
        variants += EXTRA_VARIANTS
    if args.include_orientation_delta_ranker:
        variants += EXTRA_DELTA_VARIANTS
    if args.include_orientation_dual_ranker:
        variants += EXTRA_DUAL_VARIANTS
    if args.include_orientation_pair_ranker:
        variants += EXTRA_PAIR_VARIANTS
    device = require_cuda()
    if len(args.artifact) < 3:
        raise ValueError("at least three independent CUDA artifacts are required")
    reports = [json.loads(path.read_text()) for path in args.artifact]
    reference = reports[0]["protocol"]
    raw_by_variant = {variant: {metric: [] for metric in METRICS} for variant in variants}
    per_seed: list[dict[str, Any]] = []
    paired_by_variant = {
        variant: {metric: [] for metric in METRICS} for variant in variants[1:]
    }
    for index, (path, report) in enumerate(zip(args.artifact, reports)):
        protocol = report["protocol"]
        if protocol.get("resolved_device") != "cuda":
            raise ValueError(f"artifact is not CUDA-resolved: {path}")
        for key in ("root_protocol", "train_roots", "held_out_roots", "sequence_steps", "pulse_steps", "macro_pulse_steps", "tangent_magnitudes", "contact_normal_magnitudes", "xy_shield_tolerance"):
            expected = 2.0 if key == "xy_shield_tolerance" else reference.get(key)
            actual = 2.0 if key == "xy_shield_tolerance" and key not in protocol else protocol.get(key)
            if actual != expected:
                raise ValueError(f"protocol mismatch for {key}: {path}")
        closed_loop = report["closed_loop"]
        missing = [variant for variant in variants if variant not in closed_loop]
        if missing:
            raise ValueError(f"artifact missing variants {missing}: {path}")
        raw_seed = {variant: raw_metrics(closed_loop[variant]) for variant in variants}
        seed_entry: dict[str, Any] = {
            "seed": int(protocol["train_seed"]) - 1000,
            "artifact": str(path.resolve()),
            "held_out_roots": int(protocol["held_out_roots"]),
            "root_disjoint": bool(report["sanity_checks"]["root_disjoint"]),
            "variants": {},
        }
        for variant_index, variant in enumerate(variants):
            seed_entry["variants"][variant] = {
                metric: bootstrap(values, device, args.bootstrap, args.seed + index * 1000 + variant_index * 100 + metric_index)
                for metric_index, (metric, values) in enumerate(raw_seed[variant].items())
            }
            for metric, values in raw_seed[variant].items():
                raw_by_variant[variant][metric].extend(values)
        for variant in variants[1:]:
            for metric in METRICS:
                candidate = raw_seed[variant][metric]
                baseline = raw_seed["global_one_step"][metric]
                if len(candidate) != len(baseline):
                    raise ValueError(f"root count mismatch for paired metric {metric}: {path}")
                paired_by_variant[variant][metric].extend(
                    float(candidate[i] - baseline[i]) for i in range(len(candidate))
                )
        per_seed.append(seed_entry)

    pooled = {
        variant: {
            metric: bootstrap(values, device, args.bootstrap, args.seed + 20000 + variant_index * 100 + metric_index)
            for metric_index, (metric, values) in enumerate(raw_by_variant[variant].items())
        }
        for variant_index, variant in enumerate(variants)
    }
    seed_level = {
        variant: {
            metric: bootstrap(
                [
                    float(item["variants"][variant][metric]["estimate"])
                    for item in per_seed
                    if item["variants"][variant][metric].get("estimate") is not None
                ],
                device, args.bootstrap, args.seed + 30000 + variant_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }
        for variant_index, variant in enumerate(variants)
    }
    paired = {
        variant: {
            f"{metric}_delta_vs_global": bootstrap(
                values, device, args.bootstrap, args.seed + 40000 + variant_index * 100 + metric_index,
            )
            for metric_index, (metric, values) in enumerate(metrics.items())
        }
        for variant_index, (variant, metrics) in enumerate(paired_by_variant.items())
    }
    output = {
        "artifact": "pusht_transfer_macro_panel",
        "claim_boundary": (
            "Three-seed root-disjoint CUDA panel. Large action support is an active "
            "intervention; pooled positive point estimates do not establish universal "
            "pose closure or frozen RGB planning without cross-environment replication."
        ),
        "protocol": {
            "analysis_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "n_independent_seeds": len(reports),
            "outcome_used_for_selection": False,
            "root_level_bootstrap": True,
            "seed_level_bootstrap": True,
            "fixed_protocol": {key: (2.0 if key == "xy_shield_tolerance" and key not in reference else reference.get(key)) for key in (
                "root_protocol", "train_roots", "held_out_roots", "sequence_steps",
                "pulse_steps", "macro_pulse_steps", "tangent_magnitudes",
                "contact_normal_magnitudes",
                "xy_shield_tolerance",
            )},
        },
        "per_seed": per_seed,
        "pooled_root_level": pooled,
        "seed_level": seed_level,
        "paired_delta_vs_global": paired,
        "sanity_checks": {
            "all_cuda": True,
            "all_root_disjoint": bool(all(item["root_disjoint"] for item in per_seed)),
            "same_variant_set": True,
            "same_fixed_protocol": True,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transfer_macro_panel.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    lines = [
        "# PushT large-action transfer macro panel", "",
        f"- Seeds: `{len(reports)}`; device: `{device}` ({torch.cuda.get_device_name(device)}).", "",
        "| variant | pose success | final XY | final angle | contact loss | effect error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.4f}"

    for variant in variants:
        values = pooled[variant]
        lines.append(
            f"| {variant} | {fmt(values['pose_success']['estimate'])} | "
            f"{fmt(values['final_xy']['estimate'])} | {fmt(values['final_angle']['estimate'])} | "
            f"{fmt(values['contact_loss_rate']['estimate'])} | "
            f"{fmt(values['predicted_observed_effect_error']['estimate'])} |"
        )
    lines += ["", output["claim_boundary"], ""]
    (args.output / "transfer_macro_panel.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "pooled_root_level": pooled, "seed_level": seed_level, "paired_delta_vs_global": paired}, indent=2))


if __name__ == "__main__":
    main()
