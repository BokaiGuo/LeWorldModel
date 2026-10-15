"""Aggregate calibration-only orientation gates across independent seeds.

This tool reads only orientation-response and calibration-gate artifacts.  It
does not read pose, planning, or terminal outcomes.  Its purpose is to turn
the observed seed heterogeneity into a frozen deployment support set:
authorization frequency, cross-seed intersection, and seed/root bootstrap
uncertainty.  The resulting keys are suitable for the fail-closed control
ablation in ``run_pusht_mode_aware_shared_candidate.py``.
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

from experiments.calibrate_pusht_orientation_gate import (  # noqa: E402
    group_key,
    split_by_stratum,
)
from experiments.audit_pusht_orientation_response import require_cuda  # noqa: E402


METRICS = (
    "sign_consistent",
    "successor_mode_agreement_pm",
    "paired_separation",
    "paired_antisymmetry_rmse",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-artifact", type=Path, action="append", required=True)
    parser.add_argument("--gate-artifact", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261018)
    return parser.parse_args()


def _bootstrap(values: list[float], device: torch.device, draws: int, seed: int) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n": 0}
    tensor = torch.as_tensor(finite, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    indices = torch.randint(finite.size, (int(draws), finite.size), generator=generator, device=device)
    means = tensor[indices].mean(dim=1)
    result = {
        "estimate": float(tensor.mean().item()),
        "ci95_low": float(torch.quantile(means, 0.025).item()),
        "ci95_high": float(torch.quantile(means, 0.975).item()),
        "n": int(finite.size),
    }
    del indices, means, tensor
    torch.cuda.synchronize()
    return result


def _metric_summary(rows: list[dict[str, Any]], device: torch.device, seed: int, draws: int) -> dict[str, Any]:
    return {
        metric: _bootstrap(
            [float(row[metric]) for row in rows], device, draws, seed + index
        )
        for index, metric in enumerate(METRICS)
    }


def main() -> None:
    args = parse_args()
    device = require_cuda()
    if len(args.response_artifact) != len(args.gate_artifact):
        raise ValueError("each response artifact must have one corresponding gate artifact")
    if len(args.response_artifact) < 2:
        raise ValueError("at least two independent seeds are required")

    per_seed: list[dict[str, Any]] = []
    strata: dict[str, list[dict[str, Any]]] = {}
    for index, (response_path, gate_path) in enumerate(zip(args.response_artifact, args.gate_artifact)):
        response = json.loads(response_path.read_text())
        gate = json.loads(gate_path.read_text())
        rows = list(response.get("records", []))
        if not rows:
            raise ValueError(f"response artifact has no records: {response_path}")
        calibration, evaluation = split_by_stratum(
            rows, float(gate["protocol"]["calibration_fraction"])
        )
        groups = gate.get("groups", {})
        if isinstance(groups, list):
            groups = {str(item["key"]): item for item in groups}
        authorized = sorted(
            key for key, value in groups.items() if bool(value.get("authorized", False))
        )
        released = [row for row in evaluation if group_key(row) in authorized]
        seed_value = int(response["protocol"]["seed"])
        per_seed.append({
            "seed": seed_value,
            "response_artifact": str(response_path.resolve()),
            "gate_artifact": str(gate_path.resolve()),
            "total_roots": len(rows),
            "calibration_roots": len(calibration),
            "evaluation_roots": len(evaluation),
            "authorized_strata": authorized,
            "authorized_strata_count": len(authorized),
            "released_evaluation_roots": len(released),
            "release_coverage": float(len(released) / max(len(evaluation), 1)),
            "evaluation_metrics": _metric_summary(released, device, args.seed + 1000 * index, int(args.bootstrap))
            if released else {},
            "root_disjoint": not bool(
                {str(row["root_id"]) for row in calibration}
                & {str(row["root_id"]) for row in evaluation}
            ),
        })
        for key in sorted(set(group_key(row) for row in rows) | set(authorized)):
            bucket = strata.setdefault(key, [])
            bucket.append({
                "seed": seed_value,
                "authorized": key in authorized,
                "evaluation_rows": [row for row in evaluation if group_key(row) == key],
                "released_rows": [row for row in released if group_key(row) == key],
            })

    seed_count = len(per_seed)
    all_authorized = [set(item["authorized_strata"]) for item in per_seed]
    intersection = sorted(set.intersection(*all_authorized)) if all_authorized else []
    union = sorted(set.union(*all_authorized)) if all_authorized else []
    stratum_report: dict[str, Any] = {}
    for key, entries in sorted(strata.items()):
        authorized_values = [float(entry["authorized"]) for entry in entries]
        eval_rows = [row for entry in entries for row in entry["evaluation_rows"]]
        released_rows = [row for entry in entries for row in entry["released_rows"]]
        stratum_report[key] = {
            "authorized_seed_count": int(sum(authorized_values)),
            "seed_count": seed_count,
            "authorization_frequency": float(np.mean(authorized_values)),
            "all_seed_authorized": bool(all(authorized_values)),
            "evaluation_root_count": len(eval_rows),
            "released_root_count": len(released_rows),
            "released_metrics_root_bootstrap": _metric_summary(
                released_rows, device, args.seed + 20000 + len(stratum_report) * 10, int(args.bootstrap)
            ) if released_rows else {},
        }

    # The seed-level uncertainty is deliberately computed from one mean per
    # independent seed, rather than pooling roots and pretending they are
    # exchangeable across generators.
    seed_level = {}
    for metric in METRICS:
        values = [
            float(item["evaluation_metrics"][metric]["estimate"])
            for item in per_seed if item["evaluation_metrics"].get(metric, {}).get("estimate") is not None
        ]
        seed_level[metric] = _bootstrap(values, device, int(args.bootstrap), args.seed + 30000)

    report = {
        "artifact": "pusht_orientation_gate_cross_seed_aggregate",
        "claim_boundary": (
            "Cross-seed aggregation of a calibration-only privileged orientation-effect gate. "
            "The authorized set is a support permission, not a universal certificate, RGB "
            "planner, learned latent causal proof, or pose-closure result."
        ),
        "protocol": {
            "analysis_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "outcome_fields_read": False,
            "seed_level_bootstrap": True,
            "root_level_bootstrap": True,
            "n_independent_seeds": seed_count,
        },
        "per_seed": per_seed,
        "strata": stratum_report,
        "cross_seed": {
            "authorized_union": union,
            "authorized_intersection": intersection,
            "intersection_count": len(intersection),
            "union_count": len(union),
            "intersection_frequency": float(len(intersection) / max(len(union), 1)),
            "seed_level_metrics": seed_level,
        },
        "sanity_checks": {
            "all_root_splits_disjoint": bool(all(item["root_disjoint"] for item in per_seed)),
            "no_planning_outcome_read": True,
            "cuda_analysis_completed": True,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "orientation_gate_cross_seed.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# PushT orientation gate cross-seed aggregate", "",
        f"- Seeds: `{seed_count}`; union strata: `{len(union)}`; intersection strata: `{len(intersection)}`.",
        f"- Analysis device: `{report['protocol']['analysis_device']}` ({report['protocol']['cuda_device_name']}).", "",
        "| stratum | authorized seeds | frequency | eval roots | released roots | all seeds |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, item in sorted(stratum_report.items()):
        lines.append(
            f"| {key} | {item['authorized_seed_count']} | {item['authorization_frequency']:.3f} | "
            f"{item['evaluation_root_count']} | {item['released_root_count']} | {item['all_seed_authorized']} |"
        )
    lines += ["", report["claim_boundary"], ""]
    (args.output / "orientation_gate_cross_seed.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "cross_seed": report["cross_seed"], "per_seed": per_seed}, indent=2))


if __name__ == "__main__":
    main()
