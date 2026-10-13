"""Cross-environment transfer-risk diagnostic with a strict claim gate.

This is a reader-facing wrapper around the matched learned benchmark.  It
keeps per-environment matching visible, reports the worst identifiable
environment, computes leave-one-environment-out audits, and refuses a
cross-environment claim when any required environment lacks an identifiable
matched clique or independent external evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .run_matched_learned_transfer_benchmark import _load, audit
except ImportError:  # direct script execution
    from run_matched_learned_transfer_benchmark import _load, audit  # type: ignore[no-redef]


def _record_paths(roots: list[Path], records: list[Path]) -> list[Path]:
    paths = list(records)
    for root in roots:
        paths.extend(sorted(root.rglob("checkpoint_record.json")))
    return sorted(set(paths))


def _load_all(paths: list[Path], *, require_external: bool) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    loaded: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            loaded.append(_load(path, require_external=require_external))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"path": str(path), "error": str(exc)})
    return loaded, failures


def _short_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    delta = summary["primary_delta_risk_mse_minus_full"]
    return {
        "status": summary["status"],
        "claim_ready_for_positive_delta": bool(summary["claim_ready_for_positive_delta"]),
        "identifiable_strata": int(summary["identifiable_strata"]),
        "identifiable_environments": list(summary["identifiable_environments"]),
        "environment_count": int(summary["environment_count"]),
        "architecture_count": int(summary["architecture_count"]),
        "seed_count": int(summary["seed_count"]),
        "primary_delta": delta,
    }


def diagnostic(
    records: list[dict[str, Any]],
    *,
    mse_tolerance: float = 0.05,
    spectrum_tolerance: float = 0.05,
    action_risk_tolerance: float = 0.05,
    bootstrap_seed: int = 20261013,
    require_external: bool = True,
) -> dict[str, Any]:
    if len(records) < 2:
        raise ValueError("at least two valid records are required")
    global_report = audit(
        records,
        mse_tolerance=mse_tolerance,
        spectrum_tolerance=spectrum_tolerance,
        action_risk_tolerance=action_risk_tolerance,
        bootstrap_seed=bootstrap_seed,
        require_external=require_external,
    )
    by_environment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_environment[record["_environment"]].append(record)

    per_environment: dict[str, Any] = {}
    for index, environment in enumerate(sorted(by_environment)):
        subset = by_environment[environment]
        if len(subset) < 2:
            per_environment[environment] = {
                "status": "gated_fewer_than_two_records",
                "record_count": len(subset),
                "identifiable": False,
                "reason": "fewer_than_two_records",
            }
            continue
        report = audit(
            subset,
            mse_tolerance=mse_tolerance,
            spectrum_tolerance=spectrum_tolerance,
            action_risk_tolerance=action_risk_tolerance,
            bootstrap_seed=bootstrap_seed + index + 1,
            require_external=require_external,
        )
        short = _short_summary(report)
        identifiable = bool(short["identifiable_strata"] > 0)
        per_environment[environment] = {
            **short,
            "record_count": len(subset),
            "identifiable": identifiable,
            "matched_strata": [row for row in report["strata"] if row["identifiable"]],
            "gating_reasons": [
                {"training_stage": row["training_stage"], "reason": row["reason_if_gated"]}
                for row in report["strata"] if row["reason_if_gated"]
            ],
        }

    leave_one_environment_out: dict[str, Any] = {}
    environments = sorted(by_environment)
    for index, heldout in enumerate(environments):
        subset = [record for record in records if record["_environment"] != heldout]
        if len(subset) < 2:
            leave_one_environment_out[heldout] = {
                "status": "not_estimable_after_exclusion",
                "reason": "fewer_than_two_records",
            }
            continue
        report = audit(
            subset,
            mse_tolerance=mse_tolerance,
            spectrum_tolerance=spectrum_tolerance,
            action_risk_tolerance=action_risk_tolerance,
            bootstrap_seed=bootstrap_seed + 100 + index,
            require_external=require_external,
        )
        leave_one_environment_out[heldout] = _short_summary(report)

    identifiable_envs = [name for name, item in per_environment.items() if item.get("identifiable")]
    missing_envs = [name for name, item in per_environment.items() if not item.get("identifiable", False)]
    worst_environment = None
    delta_values = [
        (name, item["primary_delta"]["mean"])
        for name, item in per_environment.items()
        if item.get("identifiable") and item.get("primary_delta", {}).get("mean") is not None
    ]
    if delta_values:
        worst_environment = min(delta_values, key=lambda item: float(item[1]))[0]
    global_summary = global_report["summary"]
    cross_environment_gate = bool(
        global_summary["claim_ready_for_positive_delta"]
        and len(identifiable_envs) >= 2
        and not missing_envs
        and all(item.get("architecture_count", 0) >= 2 and item.get("seed_count", 0) >= 2
                for item in per_environment.values())
        and require_external
    )
    return {
        "schema": "cross_environment_transfer_diagnostic.v1",
        "protocol": {
            "matching": "delegated to strict real learned matched benchmark",
            "replication_unit": "environment-level matched stratum",
            "leave_one_environment_out": True,
            "worst_environment_reported": True,
            "require_independent_external_evidence": bool(require_external),
            "positive_claim_requires": [
                "global paired positive Delta R gate",
                "at least two identifiable environments",
                "each included environment has at least two architectures and seeds",
                "independent held-out action/planning consequence",
            ],
        },
        "summary": {
            "status": "cross_environment_positive_claim_ready" if cross_environment_gate else "gated_cross_environment_claim",
            "cross_environment_positive_gate": cross_environment_gate,
            "global": _short_summary(global_report),
            "identifiable_environments": identifiable_envs,
            "missing_identifiable_environments": missing_envs,
            "worst_environment": worst_environment,
        },
        "per_environment": per_environment,
        "leave_one_environment_out": leave_one_environment_out,
        "global_audit": global_report,
        "claim_boundary": (
            "The tool can report a cross-environment positive transfer-risk result "
            "only when all declared gates pass. Missing or negative environments "
            "remain visible and fail closed."
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Cross-environment transfer diagnostic",
        "",
        f"**Status:** `{summary['status']}`  ",
        f"**Cross-environment positive gate:** `{summary['cross_environment_positive_gate']}`  ",
        f"**Identifiable environments:** {', '.join(summary['identifiable_environments']) or 'none'}  ",
        f"**Worst environment:** {summary['worst_environment'] or 'not estimable'}",
        "",
        "| Environment | Status | Identifiable strata | Architectures | Seeds | Delta R mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for environment, item in report["per_environment"].items():
        delta = item.get("primary_delta", {}).get("mean")
        lines.append(
            f"| {environment} | {item.get('status')} | {item.get('identifiable_strata', 0)} | "
            f"{item.get('architecture_count', 0)} | {item.get('seed_count', 0)} | "
            f"{'NA' if delta is None else f'{delta:.6g}'} |"
        )
    lines.extend(["", "## Claim boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="append", type=Path)
    parser.add_argument("--record-root", action="append", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mse-tolerance", type=float, default=0.05)
    parser.add_argument("--spectrum-tolerance", type=float, default=0.05)
    parser.add_argument("--action-risk-tolerance", type=float, default=0.05)
    parser.add_argument("--allow-missing-independent-external", action="store_true")
    args = parser.parse_args()
    paths = _record_paths(args.record_root or [], args.record or [])
    if not paths:
        parser.error("provide --record or --record-root")
    records, failures = _load_all(paths, require_external=not args.allow_missing_independent_external)
    if len(records) < 2:
        raise SystemExit(f"fewer than two valid records; failures={len(failures)}")
    report = diagnostic(
        records,
        mse_tolerance=args.mse_tolerance,
        spectrum_tolerance=args.spectrum_tolerance,
        action_risk_tolerance=args.action_risk_tolerance,
        require_external=not args.allow_missing_independent_external,
    )
    report["input"] = {"paths": [str(path) for path in paths], "load_failures": failures}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cross_environment_transfer_diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "cross_environment_transfer_diagnostic.md").write_text(markdown(report))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
