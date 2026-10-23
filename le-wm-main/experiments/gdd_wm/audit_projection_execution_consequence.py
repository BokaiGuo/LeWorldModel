"""Root-paired held-out execution audit for a frozen D2 projection diagnostic.

The D2 projection is trained before the outcome split is evaluated. This audit
uses the identity-bound held-out group traces emitted by D2 to quantify the
real-task regret difference between the fitted projection and its frozen
baseline. It cannot select a projection, repair gate, or threshold after the
fact; it only evaluates the frozen result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "gdd_wm_projection_execution_consequence_audit.v0.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(metric: dict[str, Any], name: str) -> dict[tuple[str, str], dict[str, Any]]:
    values = metric.get("per_group")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} lacks identity-bound per_group traces; regenerate D2 before auditing")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in values:
        if not isinstance(row, dict):
            raise ValueError(f"{name} per_group row must be an object")
        key = (str(row.get("instance_id")), str(row.get("teacher_set_id")))
        regret = row.get("real_task_regret")
        if "None" in key or not np.isfinite(float(regret)) or key in result:
            raise ValueError(f"{name} has malformed or duplicate group identity")
        result[key] = row
    return result


def audit(
    report: dict[str, Any], *, alpha: float = 0.05, draws: int = 10_000, seed: int = 202610382,
    minimum_changed_root_fraction: float = 0.10,
) -> dict[str, Any]:
    if not 0.0 < alpha < 1.0 or draws < 100 or not 0.0 < minimum_changed_root_fraction <= 1.0:
        raise ValueError("alpha, draws, and minimum changed-root coverage violate the audit contract")
    if report.get("schema") != "gdd_wm_d2_projection_distillation.v0.1":
        raise ValueError("report is not a D2 projection diagnostic")
    held_out = report.get("metrics", {}).get("held_out", {}).get("outcome")
    if not isinstance(held_out, dict):
        raise ValueError("report lacks held-out outcome metrics")
    fitted = _rows(held_out.get("fitted_projection", {}), "fitted_projection")
    baseline = _rows(held_out.get("frozen_baseline", {}), "frozen_baseline")
    if set(fitted) != set(baseline):
        raise ValueError("fitted and baseline outcome groups must match exactly")
    by_root: dict[str, list[float]] = defaultdict(list)
    changed = 0
    changed_roots: set[str] = set()
    for key in sorted(fitted):
        delta = float(fitted[key]["real_task_regret"]) - float(baseline[key]["real_task_regret"])
        by_root[key[0]].append(delta)
        did_change = fitted[key]["predicted_candidate_id"] != baseline[key]["predicted_candidate_id"]
        changed += int(did_change)
        if did_change:
            changed_roots.add(key[0])
    root_delta = np.asarray([np.mean(values) for _, values in sorted(by_root.items())], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(root_delta), size=(int(draws), len(root_delta)))
    upper = float(np.quantile(root_delta[indices].mean(axis=1), 1.0 - alpha))
    mean = float(root_delta.mean())
    changed_root_fraction = float(len(changed_roots) / len(root_delta))
    # A zero upper bound only establishes non-inferiority under resampling;
    # it is insufficient for an improvement claim.  Likewise, a gain confined
    # to one root can be a tie-break artifact rather than useful repair.
    passed = bool(
        mean < 0.0 and upper < 0.0 and changed_root_fraction >= minimum_changed_root_fraction
    )
    return {
        "schema": SCHEMA,
        "status": "passed_heldout_regret_improvement" if passed else "not_confirmed_heldout_regret_improvement",
        "passed": passed,
        "claim_boundary": "Held-out execution consequence of one frozen D2 diagnostic; not a model-selection rule, prediction-preservation result, online planning result, or universal GDD claim.",
        "protocol": {
            "outcome_split_only": True, "projection_and_baseline_frozen_before_outcome": True,
            "one_sided_root_bootstrap_alpha": float(alpha), "bootstrap_draws": int(draws), "bootstrap_seed": int(seed),
            "post_outcome_configuration_change_forbidden": True,
            "minimum_changed_root_fraction": float(minimum_changed_root_fraction),
        },
        "outcome": {
            "root_count": len(root_delta), "group_count": len(fitted),
            "changed_candidate_group_count": int(changed),
            "changed_root_count": len(changed_roots),
            "changed_root_fraction": changed_root_fraction,
            "mean_fitted_minus_baseline_regret": mean,
            "upper_95_fitted_minus_baseline_regret": upper,
        },
    }


def run(*, report_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite consequence audit: {output}")
    result = audit(json.loads(report_path.read_text(encoding="utf-8")))
    result["provenance"] = {"report": str(report_path.resolve()), "report_sha256": _sha256(report_path)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(report_path=args.report, output=args.output)
    print(json.dumps({"status": result["status"], "outcome": result["outcome"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
