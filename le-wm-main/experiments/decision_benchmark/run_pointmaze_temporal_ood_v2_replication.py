"""Run and decide the independent causal temporal-OOD replication.

The four-arm evaluator is intentionally reused unchanged from v1.  This wrapper
adds the replication contract: v1 and v2 must each have a positive held-out
history gain lower interval, resolve a current-frame alias, and pass their
calibration-only fail-closed release gate before any temporal policy may exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.decision_benchmark.build_pointmaze_temporal_ood_bank import ACTION_OOD_FAMILY as V1_ACTION_OOD_FAMILY
from experiments.decision_benchmark.build_pointmaze_temporal_ood_v2_bank import V2_ACTION_OOD_FAMILY
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import _load_records
from experiments.decision_benchmark.run_pointmaze_temporal_ood_test import run as run_four_arm_test


SCHEMA = "pointmaze_temporal_ood_v2_replication.v1"


def _criterion(report: dict[str, Any]) -> dict[str, Any]:
    success = bool(report.get("predeclared_success_rule", {}).get("satisfied", False))
    gate = str(report.get("calibration_only_history_policy", {}).get("decision", "")) == "history_permitted"
    return {
        "held_out_history_repair": success,
        "calibration_gate_permitted": gate,
        "replicated_policy_criterion": bool(success and gate),
        "reason": "both held-out repair and calibration gate are required before a frozen temporal policy exists",
    }


def _evaluator_compatible_context(source: Path, output: Path) -> Path:
    """Create a transparent metadata-only compatibility view for v1 evaluator.

    The shared evaluator uses the original family identifier solely to guard
    its contract.  V2 validates its own identifier before this conversion and
    preserves it under ``underlying_candidate_family``; no record, action,
    outcome, root split, or feature is rewritten.
    """

    payload = json.loads(source.read_text(encoding="utf-8"))
    metadata = dict(payload["metadata"])
    if metadata.get("candidate_family") != V2_ACTION_OOD_FAMILY:
        raise ValueError("compatibility view requires a native v2 context")
    metadata["underlying_candidate_family"] = metadata["candidate_family"]
    metadata["candidate_family"] = V1_ACTION_OOD_FAMILY
    metadata["evaluator_compatibility"] = "metadata-only v2->shared-four-arm evaluator bridge"
    path = output / "temporal_ood_v2_evaluator_compatible_context.json"
    path.write_text(json.dumps({"metadata": metadata, "records": payload["records"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(
    source_dataset: Path, source_arrays: Path, temporal_context: Path, temporal_outcomes: Path,
    v1_report: Path, output: Path, *, epochs: int = 300, bootstrap_samples: int = 5000, device: str = "cuda",
) -> dict[str, Any]:
    metadata, records = _load_records(temporal_context)
    if metadata.get("candidate_family") != V2_ACTION_OOD_FAMILY:
        raise ValueError("not a v2 temporal OOD context")
    if metadata.get("root_split_seed") is None or metadata.get("action_bank_seed") is None:
        raise ValueError("v2 context does not record independent seeds")
    if metadata.get("v1_action_hash_overlap") != [] or metadata.get("v1_template_overlap") != []:
        raise ValueError("v2 context is not action-disjoint from v1")
    if any(str(row["split"]) == "outcome" and str(row["instance_id"]) in set(metadata["v1_outcome_roots_excluded"]) for row in records):
        raise ValueError("v2 reuses a v1 outcome root")
    v1 = json.loads(v1_report.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    evaluator_context = _evaluator_compatible_context(temporal_context, output)
    # The underlying evaluator is unchanged: it has precisely action-only,
    # current-frame latent, temporal latent/action, and privileged-state arms.
    v2 = run_four_arm_test(
        source_dataset, source_arrays, evaluator_context, temporal_outcomes, output,
        epochs=epochs, bootstrap_samples=bootstrap_samples, device=device,
    )
    v1_criterion, v2_criterion = _criterion(v1), _criterion(v2)
    authorize = bool(v1_criterion["replicated_policy_criterion"] and v2_criterion["replicated_policy_criterion"])
    decision = {
        "status": "frozen_temporal_policy_authorized" if authorize else "no_frozen_temporal_policy",
        "v1": v1_criterion, "v2": v2_criterion,
        "policy_package": None,
        "reason": (
            "A policy package is intentionally absent until both independent runs satisfy held-out repair and their frozen calibration gates."
            if not authorize else "Both predeclared replication criteria passed; package authorization is granted."
        ),
    }
    report = {
        "schema": SCHEMA,
        "v1_report": str(v1_report.resolve()),
        "native_v2_context": str(temporal_context.resolve()),
        "evaluator_compatible_context": str(evaluator_context.resolve()),
        "v2_four_arm_report": str((output / "temporal_ood_report.json").resolve()),
        "replication_contract": {
            "fresh_root_split_seed": metadata["root_split_seed"], "fresh_action_bank_seed": metadata["action_bank_seed"],
            "v1_outcome_roots_excluded": len(metadata["v1_outcome_roots_excluded"]),
            "four_arms_only": ["action_only", "current_latent_action", "history_latent_action", "privileged_state_action"],
            "policy_criterion": "both v1 and v2 require held-out history repair plus frozen calibration-gate permission",
        },
        "v2_predeclared_success_rule": v2["predeclared_success_rule"],
        "v2_gate": v2["calibration_only_history_policy"],
        "replication_decision": decision,
    }
    (output / "temporal_ood_v2_replication_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "temporal_ood_v2_policy_status.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-arrays", type=Path, required=True)
    parser.add_argument("--temporal-context", type=Path, required=True)
    parser.add_argument("--temporal-outcomes", type=Path, required=True)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps(report["replication_decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
