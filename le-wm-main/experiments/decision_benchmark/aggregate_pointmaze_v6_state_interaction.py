"""Aggregate only sealed, independent V6 state-interaction panels.

This is intentionally narrower than the general V6 summary: a panel is
eligible only when its builder proves `state_interaction_pairs`, every held
action is source-supported, and report/context/builder bindings agree.  It
prevents a post-hoc random-bank result from being pooled with the V6.7/V6.8
mechanism replication.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.audit_unseen_action_bank_contract import sequence_hash


def _read(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _bootstrap(values: np.ndarray, *, seed: int, samples: int) -> dict[str, float | int]:
    if values.ndim != 1 or not len(values) or samples < 1:
        raise ValueError("bootstrap needs non-empty 1D gains and positive samples")
    generator = np.random.default_rng(int(seed))
    draws = np.asarray([generator.choice(values, size=len(values), replace=True).mean() for _ in range(int(samples))])
    return {
        "mean_gain": float(values.mean()),
        "lower": float(np.quantile(draws, .025)),
        "upper": float(np.quantile(draws, .975)),
        "root_count": int(len(values)),
    }


def _records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    panel = _read(path, name="V6 context")
    metadata, records = panel.get("metadata"), panel.get("records")
    if not isinstance(metadata, dict) or metadata.get("schema") not in {
        "pointmaze_temporal_ood_v6_matched_panel.v1",
        "pointmaze_temporal_ood_v6_matched_panel.v2",
    }:
        raise ValueError("context is not a V6 matched panel")
    if not isinstance(records, list) or not records:
        raise ValueError("context has no records")
    return metadata, [dict(row) for row in records]


def _hashes(records: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in records:
        action = np.asarray(row.get("action_sequence"), dtype=np.float64)
        if action.shape != (20, 2) or not np.isfinite(action).all():
            raise ValueError("candidate action is not a finite [20,2] tensor")
        computed = sequence_hash(action)
        if str(row.get("candidate_action_hash", "")) != computed:
            raise ValueError("candidate action hash does not match executable action")
        values.add(computed)
    return values


def _paired_roots(report: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    recurrent, action = report.get("recurrent"), report.get("action_only")
    if not isinstance(recurrent, dict) or not isinstance(action, dict):
        raise ValueError("score report lacks recurrent/action-only arms")
    recurrent_roots, action_roots = recurrent.get("roots"), action.get("roots")
    declared = report.get("outcome_root_ids")
    if not isinstance(recurrent_roots, dict) or not isinstance(action_roots, dict) or not isinstance(declared, list):
        raise ValueError("score report has invalid root bindings")
    roots = sorted(recurrent_roots)
    if roots != sorted(action_roots) or roots != sorted(str(value) for value in declared):
        raise ValueError("score report root bindings disagree")
    try:
        gains = np.asarray([float(action_roots[root]["regret"]) - float(recurrent_roots[root]["regret"]) for root in roots])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("score report has invalid paired regrets") from error
    return roots, gains


def _audit_panel(report_path: Path) -> tuple[dict[str, Any], list[str], np.ndarray, set[str]]:
    report = _read(report_path, name="score report")
    if report.get("schema") != "pointmaze_temporal_ood_v6_interaction_score.v3":
        raise ValueError("report is not a current V6 interaction score")
    suite = report.get("observability_suite")
    if not isinstance(suite, dict) or not {
        "candidate_oracle", "privileged_state_effect", "history_cyclic_shuffle",
        "paired_gain_visual_over_action_only", "paired_gain_visual_over_cyclic_shuffle",
    } <= set(suite):
        raise ValueError("interaction aggregate requires the frozen observability suite")
    build_path = report_path.with_name("build_v6_matched_report.json")
    build = _read(build_path, name="builder report")
    if build.get("candidate_design") != "state_interaction_pairs":
        raise ValueError("refusing non-interaction candidate panel")
    if build.get("candidate_generalization_regime") != "shared_candidate_in_support":
        raise ValueError("interaction panel must use shared candidate support")
    design = build.get("interaction_pair_design")
    if not isinstance(design, dict) or design.get("screening_role") != "design-root terminal-effect interaction only; no goal or task-cost access":
        raise ValueError("builder lacks task-blind interaction screening contract")
    if int(design.get("design_root_count", 0)) != 64 or int(design.get("proposal_count", 0)) != 96:
        raise ValueError("builder does not match the frozen interaction design")
    context_path, source_path = Path(str(build.get("context", ""))), Path(str(build.get("source_dataset", "")))
    metadata, context = _records(context_path)
    source = _read(source_path, name="source dataset").get("records")
    if not isinstance(source, list) or not source:
        raise ValueError("source dataset has no records")
    outcome = [row for row in context if str(row.get("split")) == "outcome"]
    roots, gains = _paired_roots(report)
    if roots != sorted({str(row.get("instance_id")) for row in outcome}):
        raise ValueError("report outcome roots do not bind builder outcome contexts")
    if report.get("outcome_labels_generated_after_frozen_selection") is not True:
        raise ValueError("outcome labels were not generated after frozen selection")
    if report.get("outcome_labels_preexisted") is not False:
        raise ValueError("outcome labels preexisted scoring")
    held_hashes, source_hashes = _hashes(outcome), _hashes([dict(row) for row in source])
    declared_hashes = set(str(value) for value in design.get("held_pair_hashes", []))
    if held_hashes != declared_hashes or not held_hashes <= source_hashes:
        raise ValueError("held action identity/support does not bind the interaction design")
    outcome_ids = {int(str(row.get("start_episode_id"))) for row in outcome}
    design_roots = {int(value) for value in build.get("interaction_design_roots", [])}
    if outcome_ids & design_roots:
        raise ValueError("interaction design roots overlap outcome roots")
    return {
        "report": str(report_path.resolve()), "build": str(build_path.resolve()),
        "outcome_root_count": len(roots), "active_policy": report.get("active_policy"),
        "paired_recurrent_over_action_only": _bootstrap(gains, seed=202610284 + len(roots), samples=10000),
        "outcome_label_sha256": report.get("outcome_labels_sha256"),
        "context_schema": metadata.get("schema"),
    }, roots, gains, held_hashes


def aggregate(report_paths: list[Path], *, samples: int = 10000, seed: int = 202610284) -> dict[str, Any]:
    if len(report_paths) < 2:
        raise ValueError("interaction aggregate requires at least two independent panels")
    panels, roots_seen, actions_seen, values = [], set(), set(), []
    for report_path in report_paths:
        panel, roots, gains, actions = _audit_panel(report_path)
        if overlap := roots_seen & set(roots):
            raise ValueError(f"outcome roots overlap across panels: {sorted(overlap)}")
        if overlap := actions_seen & actions:
            raise ValueError(f"held action hashes overlap across panels: {sorted(overlap)}")
        roots_seen.update(roots); actions_seen.update(actions); values.extend(gains.tolist()); panels.append(panel)
    gains = np.asarray(values, dtype=np.float64)
    return {
        "schema": "pointmaze_temporal_ood_v6_state_interaction_aggregate.v1",
        "panels": panels,
        "outcome_root_count": len(roots_seen),
        "held_action_hash_count": len(actions_seen),
        "pooled_root_level_paired_recurrent_over_action_only": _bootstrap(gains, seed=seed, samples=samples),
        "claim_boundary": "Pooled, sealed, state-interaction PointMaze mechanism evidence; not an authorization or live visual-planning claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    result = aggregate(args.reports, samples=args.bootstrap_samples)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
