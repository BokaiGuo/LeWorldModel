"""Calibrate a fail-closed PushT orientation-effect support gate.

The input is the frozen, privileged matched-pulse audit artifact.  Roots are
split deterministically within each edge/fraction stratum; only calibration
roots can authorize a stratum.  Evaluation roots are used once, only to report
whether the predeclared gate transfers.  No terminal pose or planning outcome
is read by this tool.

This is a mechanism-level calibration audit.  It is not a learned RGB
controller and a permitted stratum is not evidence of pose closure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_pusht_orientation_response import (  # noqa: E402
    gpu_bootstrap,
    require_cuda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--min-calibration-roots", type=int, default=2)
    parser.add_argument("--min-sign-consistency", type=float, default=0.75)
    parser.add_argument("--min-mode-agreement", type=float, default=0.75)
    parser.add_argument("--min-separation", type=float, default=0.02)
    parser.add_argument("--max-antisymmetry-rmse", type=float, default=0.06)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20261014)
    return parser.parse_args()


def group_key(row: dict[str, Any]) -> str:
    return f"{row['edge_id']}__bin{row['fraction_bin']}"


def split_by_stratum(rows: list[dict[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration-fraction must lie strictly between zero and one")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)
    calibration: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = sorted(grouped[key], key=lambda row: (int(row["source_index"]), int(row["root_id"])))
        cut = max(1, min(len(values) - 1, int(np.ceil(len(values) * fraction))))
        calibration.extend(values[:cut])
        evaluation.extend(values[cut:])
    return calibration, evaluation


def mean_metric(rows: list[dict[str, Any]], name: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([float(row[name]) for row in rows]))


def authorize(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    stats = {
        "n_calibration_roots": len(rows),
        "sign_consistency": mean_metric(rows, "sign_consistent"),
        "mode_agreement": mean_metric(rows, "successor_mode_agreement_pm"),
        "paired_separation": mean_metric(rows, "paired_separation"),
        "paired_antisymmetry_rmse": mean_metric(rows, "paired_antisymmetry_rmse"),
    }
    checks = {
        "minimum_support": len(rows) >= int(args.min_calibration_roots),
        "sign_consistency": bool(np.isfinite(stats["sign_consistency"]) and stats["sign_consistency"] >= args.min_sign_consistency),
        "mode_agreement": bool(np.isfinite(stats["mode_agreement"]) and stats["mode_agreement"] >= args.min_mode_agreement),
        "paired_separation": bool(np.isfinite(stats["paired_separation"]) and stats["paired_separation"] >= args.min_separation),
        "antisymmetry": bool(np.isfinite(stats["paired_antisymmetry_rmse"]) and stats["paired_antisymmetry_rmse"] <= args.max_antisymmetry_rmse),
    }
    stats["checks"] = checks
    stats["authorized"] = bool(all(checks.values()))
    return bool(stats["authorized"]), stats


def evaluate(rows: list[dict[str, Any]], device: Any, args: argparse.Namespace) -> dict[str, Any]:
    metrics = ("sign_consistent", "successor_mode_agreement_pm", "paired_separation", "paired_antisymmetry_rmse")
    return {
        metric: gpu_bootstrap(
            np.asarray([float(row[metric]) for row in rows], dtype=np.float64),
            device,
            int(args.bootstrap),
            int(args.seed + 100 + index),
        )
        for index, metric in enumerate(metrics)
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer, bool, str)) or value is None:
        return value
    return value


def main() -> None:
    args = parse_args()
    if args.min_calibration_roots < 1:
        raise ValueError("min-calibration-roots must be positive")
    device = require_cuda()
    payload = json.loads(args.artifact.read_text())
    rows = list(payload.get("records", []))
    if not rows:
        raise ValueError("orientation artifact contains no records")
    calibration, evaluation = split_by_stratum(rows, args.calibration_fraction)
    groups: dict[str, dict[str, Any]] = {}
    calibration_by_group: dict[str, list[dict[str, Any]]] = {}
    evaluation_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in calibration:
        calibration_by_group.setdefault(group_key(row), []).append(row)
    for row in evaluation:
        evaluation_by_group.setdefault(group_key(row), []).append(row)
    for key in sorted(set(calibration_by_group) | set(evaluation_by_group)):
        authorized, stats = authorize(calibration_by_group.get(key, []), args)
        heldout = evaluation_by_group.get(key, [])
        groups[key] = {
            "calibration": stats,
            "authorized": authorized,
            "evaluation_root_count": len(heldout),
            "evaluation_metrics": evaluate(heldout, device, args) if authorized and heldout else {},
        }
    authorized_keys = [key for key, value in groups.items() if value["authorized"]]
    released = [row for row in evaluation if group_key(row) in authorized_keys]
    report = {
        "artifact": "pusht_orientation_calibrated_gate",
        "claim_boundary": (
            "Calibration-only privileged orientation-effect support gate. It does not use held-out "
            "planning or pose outcomes and does not establish RGB control or pose closure."
        ),
        "protocol": {
            "source_artifact": str(args.artifact.resolve()),
            "calibration_fraction": float(args.calibration_fraction),
            "min_calibration_roots": int(args.min_calibration_roots),
            "min_sign_consistency": float(args.min_sign_consistency),
            "min_mode_agreement": float(args.min_mode_agreement),
            "min_separation": float(args.min_separation),
            "max_antisymmetry_rmse": float(args.max_antisymmetry_rmse),
            "evaluation_outcome_used": False,
            "analysis_device": str(device),
            "cuda_device_name": torch_name(device),
        },
        "split": {
            "total_roots": len(rows),
            "calibration_roots": len(calibration),
            "evaluation_roots": len(evaluation),
            "authorized_strata": authorized_keys,
            "released_evaluation_roots": len(released),
        },
        "groups": groups,
        "released_evaluation_metrics": evaluate(released, device, args) if released else {},
        "sanity_checks": {
            "root_disjoint": not bool(set(row["root_id"] for row in calibration) & set(row["root_id"] for row in evaluation)),
            "no_outcome_field_read": True,
            "cuda_analysis_completed": True,
        },
    }
    report = json_safe(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "orientation_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# PushT calibrated orientation gate", "",
        f"- Authorized strata: `{len(authorized_keys)}`; released evaluation roots: `{len(released)}`.",
        f"- Analysis device: `{report['protocol']['analysis_device']}` ({report['protocol']['cuda_device_name']}).", "",
        "| stratum | calibration roots | authorized | evaluation roots | evaluation sign | evaluation mode |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in sorted(groups.items()):
        metrics = value["evaluation_metrics"]
        lines.append(
            f"| {key} | {value['calibration']['n_calibration_roots']} | {value['authorized']} | "
            f"{value['evaluation_root_count']} | "
            f"{metrics.get('sign_consistent', {}).get('estimate', 'NA')} | "
            f"{metrics.get('successor_mode_agreement_pm', {}).get('estimate', 'NA')} |"
        )
    lines += ["", report["claim_boundary"], ""]
    (args.output / "orientation_gate.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "split": report["split"], "groups": report["groups"]}, indent=2))


def torch_name(device: Any) -> str:
    # Kept separate so the JSON construction remains easy to test without
    # instantiating an environment or reading any planning artifact.
    import torch

    return str(torch.cuda.get_device_name(device))


if __name__ == "__main__":
    main()
