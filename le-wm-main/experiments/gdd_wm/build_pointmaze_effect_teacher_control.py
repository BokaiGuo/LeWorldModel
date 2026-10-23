"""Build a train-only PointMaze intervention-effect teacher *control*.

This is deliberately not a D-JEPA preference teacher.  The fit target is the
observed terminal state delta from a declared, executed PointMaze action
sequence.  Source state, goal state, and action sequence are decision-time
causal variables in the simulator; ``real_task_cost`` is excluded from the
teacher inputs, target, fitting, and calibration-stability calculation.

The artifact answers a narrow repairability question: can a train-only,
execution-supervised state-effect model induce stable candidate preferences on
root-disjoint calibration contexts, and do those frozen preferences align with
outcome-root task costs?  The latter is an external evaluation only.  Even a
positive result remains an explicitly labelled state-access control and never
permits D2, distillation, deployment, or a frozen-visual claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gdd_wm.build_pointmaze_frozen_predictive_teacher_package import _action_summary
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "gdd_wm_pointmaze_execution_supervised_effect_control.v0.1"
SPLITS = ("train", "calibration", "outcome")
CONTROL_ID = "pointmaze_train_only_execution_supervised_state_effect_ridge_v1"
DEFAULT_FEATURE_FAMILY = "state_goal_latent_action"
FEATURE_FAMILIES = frozenset({
    "state_goal_latent_action",
    "state_action",
    "latent_action",
    "history_action",
    "latent_history_action",
})


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_hash(row: dict[str, Any]) -> str:
    sequence = row.get("action_sequence")
    if not isinstance(sequence, list) or not sequence:
        raise ValueError(f"{row.get('instance_id')}: action_sequence must be non-empty")
    return _canonical_hash(sequence)


def _load_context(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata, records = payload.get("metadata"), payload.get("records")
    if not isinstance(metadata, dict) or not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected non-empty metadata and records")
    arrays_path = path.parent / "arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"{path}: missing arrays.npz")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        values = {key: np.asarray(arrays[key], dtype=np.float64) for key in arrays.files}
    return dict(metadata), [dict(row) for row in records], values


def _terminal_state(row: dict[str, Any], arrays: dict[str, np.ndarray], source: Path) -> np.ndarray:
    reference = str(row.get("realized_terminal_state_ref", ""))
    if "arrays.npz::" not in reference:
        raise ValueError(f"{source}: {row.get('instance_id')} has no realized terminal-state reference")
    key = reference.split("arrays.npz::", 1)[1]
    if key not in arrays:
        raise ValueError(f"{source}: missing realized terminal-state array {key}")
    value = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
    if value.size != 4 or not np.isfinite(value).all():
        raise ValueError(f"{source}: realized terminal state must be a finite PointMaze state[4]")
    return value


def _source_state(row: dict[str, Any]) -> np.ndarray:
    value = np.asarray(row.get("source_state_raw"), dtype=np.float64).reshape(-1)
    if value.size != 4 or not np.isfinite(value).all():
        raise ValueError(f"{row.get('instance_id')}: source_state_raw must be finite state[4]")
    return value


def _goal_state(row: dict[str, Any]) -> np.ndarray:
    value = np.asarray(row.get("goal_raw"), dtype=np.float64).reshape(-1)
    if value.size != 2 or not np.isfinite(value).all():
        raise ValueError(f"{row.get('instance_id')}: goal_raw must be a finite PointMaze XY goal[2]")
    return value


def _source_latent(row: dict[str, Any]) -> np.ndarray:
    value = np.asarray(row.get("source_latent_model"), dtype=np.float64).reshape(-1)
    if value.size < 1 or not np.isfinite(value).all():
        raise ValueError(f"{row.get('instance_id')}: source_latent_model must be finite and non-empty")
    return value


def _source_history(row: dict[str, Any]) -> np.ndarray:
    """Return the frozen causal history available at decision time."""

    value = np.asarray(row.get("source_history_model"), dtype=np.float64).reshape(-1)
    if value.size < 1 or not np.isfinite(value).all():
        raise ValueError(f"{row.get('instance_id')}: source_history_model must be finite and non-empty")
    return value


def _feature_fields(feature_family: str) -> list[str]:
    if feature_family == "state_goal_latent_action":
        return ["source_state_raw", "goal_raw", "source_latent_model", "action_sequence_summary"]
    if feature_family == "state_action":
        return ["source_state_raw", "action_sequence_summary"]
    if feature_family == "latent_action":
        return ["source_latent_model", "action_sequence_summary"]
    if feature_family == "history_action":
        return ["source_history_model", "action_sequence_summary"]
    if feature_family == "latent_history_action":
        return ["source_latent_model", "source_history_model", "action_sequence_summary"]
    raise ValueError(f"unknown feature family {feature_family!r}")


def _features(row: dict[str, Any], feature_family: str = DEFAULT_FEATURE_FAMILY) -> np.ndarray:
    """Return a declared causal input family without successor/cost/rank/ID.

    The ablation is deliberately inside the same effect-control protocol.  It
    separates a positive state-access control from what the frozen current
    latent or frozen short history actually makes available for action-effect
    prediction.  The goal is intentionally omitted from the visual-only
    families because physical effect does not causally depend on the task goal;
    goal remains only in the fixed post-effect task readout.
    """

    if feature_family == "state_goal_latent_action":
        parts = (_source_state(row), _goal_state(row), _source_latent(row), _action_summary(row))
    elif feature_family == "state_action":
        parts = (_source_state(row), _action_summary(row))
    elif feature_family == "latent_action":
        parts = (_source_latent(row), _action_summary(row))
    elif feature_family == "history_action":
        parts = (_source_history(row), _action_summary(row))
    elif feature_family == "latent_history_action":
        parts = (_source_latent(row), _source_history(row), _action_summary(row))
    else:
        raise ValueError(f"unknown feature family {feature_family!r}")
    return np.concatenate(parts)


def _fit_ridge(features: np.ndarray, target: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ridge <= 0.0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and positive")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    standardized = (features - mean) / scale
    design = np.concatenate((np.ones((len(standardized), 1)), standardized), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target), mean, scale


def _predict(features: np.ndarray, beta: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standardized = (features - mean) / scale
    design = np.concatenate((np.ones((len(standardized), 1)), standardized), axis=1)
    return design @ beta


def _preference_score(row: dict[str, Any], predicted_effect: np.ndarray) -> float:
    """Task-conditioned score from causal current/goal state and predicted effect.

    This is a declared simulator-state control readout, not a task-cost label.
    """

    predicted_terminal = _source_state(row) + np.asarray(predicted_effect, dtype=np.float64)
    goal = _goal_state(row)
    return -float(np.linalg.norm(predicted_terminal[:2] - goal))


@dataclass(frozen=True)
class _Row:
    context_id: str
    row: dict[str, Any]
    source: Path
    terminal_effect: np.ndarray | None


def _execute_train_terminal_state(row: dict[str, Any], model: Any) -> np.ndarray:
    """Execute a declared train-root action sequence for an effect label.

    The returned state is used only as a train target.  Calibration/outcome
    roots never enter this executor: their scores come from the frozen fitted
    effect model, while outcome costs remain a final external evaluation.
    """

    from experiments.decision_benchmark.verify_rgb_successor import _simulate_blocks

    source = _source_state(row)
    sequence = np.asarray(row.get("action_sequence"), dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[1] != 2 or not np.isfinite(sequence).all():
        raise ValueError(f"{row.get('instance_id')}: PointMaze action_sequence must be finite [steps,2]")
    blocks = list(_simulate_blocks(model, source, sequence))
    if not blocks:
        raise ValueError(f"{row.get('instance_id')}: declared action sequence produced no simulator block")
    terminal = np.asarray(blocks[-1], dtype=np.float64).reshape(-1)
    if terminal.shape != source.shape or not np.isfinite(terminal).all():
        raise ValueError("executed terminal PointMaze state is invalid")
    return terminal


def _validate_contexts(contexts: list[tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]]) -> dict[str, str]:
    if len(contexts) < 3:
        raise ValueError("at least three shared-anchor contexts are required for D1 stability")
    split_maps: list[dict[str, str]] = []
    action_maps: list[dict[str, set[str]]] = []
    identities: set[str] = set()
    for path, metadata, rows, _arrays in contexts:
        if str(metadata.get("environment", "")) != "PointMaze":
            raise ValueError(f"{path}: only PointMaze shared-anchor contexts are accepted")
        identity = str(metadata.get("world_model_sha256", metadata.get("checkpoint", "")))
        if not identity:
            raise ValueError(f"{path}: missing frozen world-model identity")
        identities.add(identity)
        split_map: dict[str, str] = {}
        action_map: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            root, split = str(row.get("instance_id", "")), str(row.get("split", ""))
            if not root or split not in SPLITS:
                raise ValueError(f"{path}: every record needs a root-disjoint train/calibration/outcome split")
            if split_map.setdefault(root, split) != split:
                raise ValueError(f"{path}: root {root} occurs in multiple splits")
            action_map[root].add(_action_hash(row))
        split_maps.append(split_map)
        action_maps.append(action_map)
    if len(identities) != 1:
        raise ValueError("all contexts must use the same frozen world model")
    if any(item != split_maps[0] for item in split_maps[1:]):
        raise ValueError("shared-anchor contexts must have identical root split assignments")
    if set(split_maps[0].values()) != set(SPLITS):
        raise ValueError("all three root-disjoint splits must be non-empty")
    for root in split_maps[0]:
        common = set.intersection(*(mapping[root] for mapping in action_maps))
        if len(common) < 2:
            raise ValueError(f"root {root} has fewer than two common action anchors")
    return split_maps[0]


def _d1_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate only calibration-root repeated contexts; outcome costs are untouched."""

    by_root_context: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["split"] != "calibration":
            continue
        root, context, action = row["instance_id"], row["candidate_bank_context_id"], row["action_sequence_sha256"]
        prior = by_root_context[root][context].get(action)
        if prior is not None and not np.isclose(prior, float(row["teacher_score"]), rtol=0.0, atol=1.0e-12):
            raise ValueError(f"conflicting calibration duplicate {root}/{context}/{action}")
        by_root_context[root][context][action] = float(row["teacher_score"])
    stable = unstable = 0
    root_summaries: dict[str, Any] = {}
    for root, contexts in sorted(by_root_context.items()):
        common = set.intersection(*(set(values) for values in contexts.values()))
        root_stable = root_unstable = 0
        ordered = sorted(common)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1:]:
                signs = {int(np.sign(values[left] - values[right])) for values in contexts.values()}
                if 0 in signs:
                    continue
                if len(signs) == 1:
                    stable += 1
                    root_stable += 1
                else:
                    unstable += 1
                    root_unstable += 1
        root_summaries[root] = {
            "context_count": len(contexts),
            "common_anchor_count": len(common),
            "stable_pair_count": root_stable,
            "unstable_pair_count": root_unstable,
        }
    return {
        "split": "calibration",
        "root_count": len(by_root_context),
        "stable_pair_count": stable,
        "unstable_pair_count": unstable,
        "passed": bool(by_root_context and stable > 0 and unstable == 0),
        "per_root": root_summaries,
        "real_task_cost_read": False,
    }


def _held_outcome_alignment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use outcome costs exactly once, after teacher freeze, as external alignment."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "outcome":
            groups[(row["instance_id"], row["candidate_bank_context_id"])].append(row)
    comparable = consistent = 0
    regrets: list[float] = []
    for group in groups.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1:]:
                score_delta = float(left["teacher_score"]) - float(right["teacher_score"])
                cost_delta = float(right["real_task_cost"]) - float(left["real_task_cost"])
                if abs(score_delta) <= 1.0e-10 or abs(cost_delta) <= 1.0e-10:
                    continue
                comparable += 1
                consistent += int(np.sign(score_delta) == np.sign(cost_delta))
        selected = max(group, key=lambda row: (float(row["teacher_score"]), str(row["candidate_id"])))
        oracle = min(group, key=lambda row: (float(row["real_task_cost"]), str(row["candidate_id"])))
        regrets.append(float(selected["real_task_cost"]) - float(oracle["real_task_cost"]))
    return {
        "split": "outcome",
        "root_context_count": len(groups),
        "comparable_pair_count": comparable,
        "pairwise_execution_alignment": None if comparable == 0 else float(consistent / comparable),
        "mean_candidate_regret": None if not regrets else float(np.mean(regrets)),
        "zero_regret_fraction": None if not regrets else float(np.mean(np.asarray(regrets) <= 1.0e-10)),
        "real_task_cost_read": True,
        "use": "post-freeze external outcome evaluation only; never input to teacher, fitting, calibration stability, or D2 authorization",
    }


def build(
    context_paths: list[Path],
    output: Path,
    *,
    ridge: float,
    device: str = "cuda",
    feature_family: str = DEFAULT_FEATURE_FAMILY,
) -> dict[str, Any]:
    """Build the train-only state-effect control and its separated audit."""

    if feature_family not in FEATURE_FAMILIES:
        raise ValueError(f"unknown feature family {feature_family!r}")
    resolved = [path.resolve() for path in context_paths]
    if len(resolved) != len(set(resolved)):
        raise ValueError("context paths must be distinct")
    contexts = [(path, *_load_context(path)) for path in resolved]
    _validate_contexts(contexts)
    cuda = require_cuda_allocation(device)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml
    import mujoco

    execution_model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    rows: list[_Row] = []
    for context_index, (path, _metadata, source_rows, arrays) in enumerate(contexts):
        context_id = f"context_{context_index:02d}_{_file_hash(path)[:12]}"
        for source_row in source_rows:
            row = dict(source_row)
            source = _source_state(row)
            # No calibration/outcome record is executed or used as a label.
            # This preserves the intended train-only intervention supervision.
            terminal_effect = (
                _execute_train_terminal_state(row, execution_model) - source
                if str(row["split"]) == "train"
                else None
            )
            rows.append(_Row(
                context_id=context_id,
                row=row,
                source=path,
                terminal_effect=terminal_effect,
            ))
    features = np.stack([_features(item.row, feature_family) for item in rows])
    train_indices = np.asarray([index for index, item in enumerate(rows) if str(item.row["split"]) == "train"], dtype=np.int64)
    if len(train_indices) < features.shape[1] + 1:
        raise ValueError("insufficient train-only root rows for effect ridge")
    # Repeated shared anchors identify context stability, but must not gain
    # extra fitting weight merely because they appear in three contexts.
    unique_train: dict[tuple[str, str], int] = {}
    for index in train_indices:
        item = rows[int(index)]
        key = (str(item.row["instance_id"]), _action_hash(item.row))
        existing = unique_train.get(key)
        if existing is not None and not np.allclose(rows[existing].terminal_effect, item.terminal_effect, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"repeated train anchor has inconsistent executed effect: {key}")
        unique_train.setdefault(key, int(index))
    fit_indices = np.asarray(sorted(unique_train.values()), dtype=np.int64)
    fit_targets = np.stack([np.asarray(rows[index].terminal_effect, dtype=np.float64) for index in fit_indices])
    beta, mean, scale = _fit_ridge(features[fit_indices], fit_targets, ridge)
    predicted_effect = _predict(features, beta, mean, scale)

    audit_rows: list[dict[str, Any]] = []
    for item, effect in zip(rows, predicted_effect):
        row = item.row
        audit_rows.append({
            "instance_id": str(row["instance_id"]),
            "split": str(row["split"]),
            "candidate_id": str(row["candidate_id"]),
            "candidate_bank_context_id": item.context_id,
            "action_sequence_sha256": _action_hash(row),
            "teacher_score": _preference_score(row, effect),
            "teacher_score_source": "train_only_predicted_execution_effect_to_causal_goal_progress",
            "real_task_cost": float(row["real_task_cost"]),
        })
    d1 = _d1_stability(audit_rows)
    outcome = _held_outcome_alignment(audit_rows)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "effect_teacher_model.npz", beta=beta, mean=mean, scale=scale)
    report = {
        "schema": SCHEMA,
        "status": "completed_execution_supervised_control",
        "teacher_identity": f"{CONTROL_ID}__{feature_family}",
        "claim_boundary": (
            "A simulator-state, execution-supervised intervention-effect control. It is not an independent D-JEPA teacher, "
            "not a frozen-visual interface, and cannot authorize D2, distillation, planning, or deployment."
        ),
        "teacher_contract": {
            "control_label": "execution_supervised_state_effect_control",
            "teacher_input_fields": _feature_fields(feature_family),
            "teacher_target": "executed_terminal_state_minus_source_state",
            "forbidden_fit_fields": ["real_task_cost", "candidate_id", "teacher_rank", "outcome_root_rows"],
            "fit_split": "train",
            "calibration_split": "calibration",
            "outcome_split": "outcome_external_evaluation_only",
            "outcome_roots_excluded_from_fit": True,
            "outcome_roots_excluded_from_calibration_stability": True,
            "execution_supervised": True,
            "deploy_interface_matches_frozen_visual_model": False,
        },
        "fit_provenance": {
            "context_paths": [str(path) for path in resolved],
            "context_sha256": [_file_hash(path) for path in resolved],
            "raw_train_row_count": int(len(train_indices)),
            "deduplicated_train_intervention_count": int(len(fit_indices)),
            "feature_family": feature_family,
            "feature_dimension": int(features.shape[1]),
            "effect_dimension": int(fit_targets.shape[1]),
            "ridge": float(ridge),
            "real_task_cost_read_during_fit": False,
            "outcome_roots_read_during_fit": False,
            "calibration_roots_read_for_effect_target": False,
            "train_effect_target_backend": "legacy_pointmaze_mujoco_declared_action_replay",
            "model_sha256": _file_hash(output / "effect_teacher_model.npz"),
        },
        "d1_calibration_stability": d1,
        "held_out_outcome_alignment": outcome,
        "d2_permission": {
            "status": "fail_closed",
            "reason": "The teacher is explicitly execution-supervised and state-accessed; it is a repairability control, not an independent teacher with a frozen visual deployment interface.",
        },
        "cuda_preflight": cuda,
        "rows": audit_rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--feature-family", choices=sorted(FEATURE_FAMILIES), default=DEFAULT_FEATURE_FAMILY)
    args = parser.parse_args()
    report = build(
        args.contexts,
        args.output,
        ridge=args.ridge,
        device=args.device,
        feature_family=args.feature_family,
    )
    print(json.dumps({
        "status": report["status"],
        "d1_calibration_stability": report["d1_calibration_stability"]["passed"],
        "d2_permission": report["d2_permission"],
        "held_out_outcome_alignment": report["held_out_outcome_alignment"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
