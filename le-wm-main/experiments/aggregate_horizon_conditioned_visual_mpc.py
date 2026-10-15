"""Aggregate frozen-protocol horizon-conditioned visual MPC artifacts.

The input runs are independent seeds of the same h=4 short-pulse visual MPC
protocol.  This utility has no model-selection authority: it validates the
protocol, reports seed-level and episode-level uncertainty, and computes the
paired repaired-minus-base consequence on the same held-out episode indices.
Bootstrap sampling is deliberately performed on CUDA so the aggregation does
not silently fall back to CPU in a GPU-first experiment.
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

from experiments.audit_pusht_orientation_response import require_cuda  # noqa: E402


POLICIES = ("base_mpc", "repaired_mpc", "heuristic", "residual_gated_mpc")
METRICS = (
    "success",
    "contact_reached",
    "block_xy_distance",
    "final_distance",
    "return",
    "mean_selected_endpoint_pose_error",
)
PROTOCOL_KEYS = (
    "environment",
    "train_cases",
    "held_out_cases",
    "episodes",
    "prediction_horizon",
    "action_block",
    "candidates",
    "state_source",
    "planning_cost",
    "bootstrap_mode",
    "candidate_mode",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261115)
    return parser.parse_args()


def bootstrap(values: list[float], device: torch.device, draws: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n": 0}
    tensor = torch.as_tensor(array, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    indices = torch.randint(array.size, (int(draws), array.size), generator=generator, device=device)
    means = tensor[indices].mean(dim=1)
    result = {
        "estimate": float(tensor.mean().item()),
        "ci95_low": float(torch.quantile(means, 0.025).item()),
        "ci95_high": float(torch.quantile(means, 0.975).item()),
        "n": int(array.size),
    }
    del indices, means, tensor
    torch.cuda.synchronize(device)
    return result


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status") not in {"complete", "completed"}:
        raise ValueError(f"artifact is not complete: {path}")
    protocol = payload.get("protocol", {})
    missing = [key for key in PROTOCOL_KEYS if key not in protocol]
    if missing:
        raise ValueError(f"artifact missing protocol keys {missing}: {path}")
    return payload


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric in {"success", "contact_reached"}:
        return float(bool(row.get(metric, False)))
    return float(row.get(metric, np.nan))


def rows_for(report: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    matching = [item for item in report.get("reports", []) if item.get("policy") == policy]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one {policy} report, found {len(matching)}")
    rows = list(matching[0].get("episodes", []))
    if not rows:
        raise ValueError(f"empty episode rows for {policy}")
    return rows


def main() -> None:
    args = parse_args()
    if len(args.artifact) < 2:
        raise ValueError("at least two independent seed artifacts are required")
    device = require_cuda()
    reports = [load(path) for path in args.artifact]
    reference_protocol = {
        key: reports[0]["protocol"][key] for key in PROTOCOL_KEYS
    }
    for path, report in zip(args.artifact, reports):
        protocol = {key: report["protocol"][key] for key in PROTOCOL_KEYS}
        if protocol != reference_protocol:
            raise ValueError(f"protocol mismatch for {path}: {protocol} != {reference_protocol}")
        if not report.get("sanity_checks", {}).get("all_finite", False):
            raise ValueError(f"non-finite artifact rejected: {path}")

    per_seed: list[dict[str, Any]] = []
    pooled: dict[str, dict[str, dict[str, Any]]] = {
        policy: {metric: [] for metric in METRICS} for policy in POLICIES
    }
    paired_rows: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for artifact_index, (path, report) in enumerate(zip(args.artifact, reports)):
        base_rows = rows_for(report, "base_mpc")
        seed_entry: dict[str, Any] = {
            "seed": int(report["protocol"]["seed"]),
            "artifact": str(path.resolve()),
            "n_episodes": len(base_rows),
            "policies": {},
        }
        rows_by_policy = {policy: rows_for(report, policy) for policy in POLICIES}
        episode_ids = [int(row["episode"]) for row in base_rows]
        for policy, rows in rows_by_policy.items():
            if [int(row["episode"]) for row in rows] != episode_ids:
                raise ValueError(f"episode mismatch for {policy}: {path}")
            seed_entry["policies"][policy] = {
                metric: bootstrap(
                    [metric_value(row, metric) for row in rows],
                    device,
                    args.bootstrap,
                    args.seed + artifact_index * 1000 + len(seed_entry["policies"]) * 100 + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            }
            for metric in METRICS:
                pooled[policy][metric].extend(metric_value(row, metric) for row in rows)
        for repaired, base in zip(rows_by_policy["repaired_mpc"], rows_by_policy["base_mpc"]):
            for metric in METRICS:
                paired_rows[metric].append(metric_value(repaired, metric) - metric_value(base, metric))
        per_seed.append(seed_entry)

    pooled_report = {
        policy: {
            metric: bootstrap(values, device, args.bootstrap, args.seed + 20000 + policy_index * 100 + metric_index)
            for metric_index, (metric, values) in enumerate(metrics.items())
        }
        for policy_index, (policy, metrics) in enumerate(pooled.items())
    }
    seed_level: dict[str, dict[str, Any]] = {}
    for policy_index, policy in enumerate(POLICIES):
        seed_level[policy] = {}
        for metric_index, metric in enumerate(METRICS):
            values = [
                float(seed["policies"][policy][metric]["estimate"])
                for seed in per_seed
                if seed["policies"][policy][metric]["estimate"] is not None
            ]
            seed_level[policy][metric] = bootstrap(
                values, device, args.bootstrap,
                args.seed + 30000 + policy_index * 100 + metric_index,
            )
    paired_report = {
        metric: bootstrap(values, device, args.bootstrap, args.seed + 40000 + metric_index)
        for metric_index, (metric, values) in enumerate(paired_rows.items())
    }
    report = {
        "artifact": "horizon_conditioned_visual_mpc_panel",
        "claim_boundary": (
            "Three-seed root/episode-disjoint visual MPC intervention. The h=4 "
            "short-pulse arm is a predictor-repair diagnostic; it does not establish "
            "frozen-LeWM causal physics, universal planning, or task success."
        ),
        "protocol": {
            "analysis_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "n_independent_seeds": len(reports),
            "outcome_used_for_selection": False,
            "frozen_comparison": "base_mpc versus compositional repaired_mpc on identical episode indices",
            "source_protocol": reference_protocol,
        },
        "per_seed": per_seed,
        "pooled_episode_level": pooled_report,
        "seed_level": seed_level,
        "paired_repaired_minus_base": paired_report,
        "sanity_checks": {
            "all_protocols_match": True,
            "all_finite": True,
            "same_episode_indices_within_seed": True,
            "cuda_analysis_completed": True,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "horizon_conditioned_visual_mpc_panel.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.4f}"

    lines = [
        "# Horizon-conditioned visual MPC panel",
        "",
        f"- Seeds: `{len(reports)}`; analysis device: `{device}` ({torch.cuda.get_device_name(device)}).",
        "",
        "| policy | success | contact reached | block XY | final distance |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        values = pooled_report[policy]
        lines.append(
            f"| {policy} | {fmt(values['success']['estimate'])} | "
            f"{fmt(values['contact_reached']['estimate'])} | "
            f"{fmt(values['block_xy_distance']['estimate'])} | "
            f"{fmt(values['final_distance']['estimate'])} |"
        )
    lines += [
        "",
        "## Paired repaired-minus-base consequence",
        "",
        "| metric | delta | 95% CI |",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        item = paired_report[metric]
        lines.append(
            f"| {metric} | {fmt(item['estimate'])} | "
            f"[{fmt(item['ci95_low'])}, {fmt(item['ci95_high'])}] |"
        )
    lines += ["", report["claim_boundary"], ""]
    (args.output / "horizon_conditioned_visual_mpc_panel.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "paired_repaired_minus_base": paired_report}, indent=2))


if __name__ == "__main__":
    main()
