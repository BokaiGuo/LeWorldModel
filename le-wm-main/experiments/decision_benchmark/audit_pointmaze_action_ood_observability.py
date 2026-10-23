"""Diagnose the PointMaze unseen-action-bank failure without choosing on outcomes.

The sealed unseen-action-bank kill test showed that a source-latent/action
effect MLP can lose to an action-only control when all action sequences move
out of distribution.  That result admits two different mechanisms:

* the frozen source latent aliases states with different action effects; or
* state conditioning is simply the wrong inductive bias for this action bank.

This audit keeps the sealed bank, root split, action hashes, model family,
epochs, and ensemble seeds fixed.  It fits four *predeclared* feature families
on train roots only: action-only, frozen source-latent, a privileged physical
state oracle, and a legal temporal history only when the source protocol
actually exports a distinct causal history.  Calibration and outcome labels
are read only after every predictor has been fitted.  They are reported for
diagnosis only; no outcome result chooses a model or threshold.

The physical-state arm is a mechanism control, never a deployable method.  A
positive state-oracle / action-only gap alongside a non-positive latent /
action-only gap supports a representation-observability repair route.  If the
state oracle also cannot beat action-only, the fail-closed route is an
action-only fallback for this action-OOD regime, not more latent conditioning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.audit_unseen_action_bank_contract import sequence_hash
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import (
    _append_outcome_labels,
    _load_records,
    _paired_gain,
    _selection,
)
from experiments.decision_benchmark.train_candidate_effect_adapter import (
    SEEDS,
    _array_key,
    fit_effect_model,
    predict_effect_model,
)
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_action_ood_observability.v0.1"
LATENT_NEAR_QUANTILE = 0.25
EFFECT_DIVERGENT_QUANTILE = 0.75
FEATURE_ORDER = ("action_only", "frozen_source_latent", "physical_state_oracle")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_rows(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return one source context per root and reject candidate-varying context."""

    fields = ("split", "source_latent_model", "source_state_raw", "goal_raw")
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        root = str(row.get("instance_id", ""))
        if not root:
            raise ValueError("record lacks instance_id")
        previous = result.setdefault(root, row)
        for field in fields:
            if field not in row or field not in previous:
                raise ValueError(f"record lacks {field}")
            if field == "split":
                if str(row[field]) != str(previous[field]):
                    raise ValueError(f"root has mixed split: {root}")
            elif not np.allclose(np.asarray(row[field]), np.asarray(previous[field]), rtol=0.0, atol=1e-10):
                raise ValueError(f"root has candidate-conditioned {field}: {root}")
    return result


def temporal_history_status(records: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """Return an explicit availability decision; identity history is not a control."""

    declared = int(metadata.get("history_windows", 1))
    root_rows = _root_rows(records)
    latent_width = len(np.asarray(next(iter(root_rows.values()))["source_latent_model"]).reshape(-1))
    histories = []
    for row in root_rows.values():
        value = row.get("source_history_model")
        if value is None:
            return {
                "status": "unavailable_missing_source_history",
                "declared_history_windows": declared,
                "reason": "candidate bank does not export a causal source history",
            }
        history = np.asarray(value, dtype=np.float64)
        histories.append(history.reshape(-1))
    widths = sorted({int(history.size) for history in histories})
    identity = all(
        history.size == latent_width and np.allclose(history, np.asarray(row["source_latent_model"], dtype=np.float64))
        for history, row in zip(histories, root_rows.values())
    )
    if declared < 2 or identity:
        return {
            "status": "unavailable_identity_history",
            "declared_history_windows": declared,
            "history_widths": widths,
            "latent_width": latent_width,
            "reason": "the exported history is a single current latent, not an independent temporal feature",
        }
    if len(widths) != 1 or widths[0] <= latent_width:
        return {
            "status": "unavailable_invalid_history_shape",
            "declared_history_windows": declared,
            "history_widths": widths,
            "latent_width": latent_width,
            "reason": "history does not provide a consistent multi-frame causal context",
        }
    return {
        "status": "available",
        "declared_history_windows": declared,
        "history_width": widths[0],
        "latent_width": latent_width,
        "reason": "distinct causal source-history feature exported by the sealed protocol",
    }


def _context(row: dict[str, Any], family: str) -> np.ndarray:
    if family == "action_only":
        value = np.empty((0,), dtype=np.float64)
    elif family == "frozen_source_latent":
        value = np.asarray(row["source_latent_model"], dtype=np.float64).reshape(-1)
    elif family == "physical_state_oracle":
        value = np.asarray(row["source_state_raw"], dtype=np.float64).reshape(-1)
    elif family == "temporal_history":
        value = np.asarray(row["source_history_model"], dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"unsupported feature family: {family}")
    action = np.asarray(row["action_sequence"], dtype=np.float64).reshape(-1)
    return np.concatenate((value, action)).astype(np.float32)


def _features_and_effects(
    records: list[dict[str, Any]], arrays: Any, family: str
) -> tuple[np.ndarray, np.ndarray]:
    values, effects = [], []
    for row in records:
        source = np.asarray(row["source_state_raw"], dtype=np.float64)
        terminal = np.asarray(arrays[_array_key(str(row["realized_terminal_state_ref"]))], dtype=np.float64)
        values.append(_context(row, family))
        effects.append(terminal[:2] - source[:2])
    return np.asarray(values, dtype=np.float32), np.asarray(effects, dtype=np.float32)


def _ensemble_predict(
    train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, *, epochs: int, device: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions, fits = [], []
    for seed in SEEDS:
        model, fit = fit_effect_model(train_x, train_y, seed=seed, epochs=epochs, device_name=device)
        predictions.append(predict_effect_model(model, fit, eval_x))
        fits.append({key: value for key, value in fit.items() if key not in {"x_mean", "x_scale", "y_mean", "y_scale"}})
    return np.mean(np.stack(predictions), axis=0), fits


def _costs(records: list[dict[str, Any]], effect: np.ndarray) -> np.ndarray:
    source_xy = np.asarray([row["source_state_raw"][:2] for row in records], dtype=np.float64)
    goals = np.asarray([row["goal_raw"][:2] for row in records], dtype=np.float64)
    return np.linalg.norm(source_xy + np.asarray(effect, dtype=np.float64) - goals, axis=1)


def _partition_selection(records: list[dict[str, Any]], costs: np.ndarray, split: str) -> dict[str, Any]:
    mask = np.asarray([str(row["split"]) == split for row in records])
    if not mask.any():
        raise ValueError(f"no {split} records")
    return _selection([row for row, keep in zip(records, mask) if keep], np.asarray(costs)[mask])


def _effect_maps_from_source(records: list[dict[str, Any]], arrays: Any) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for row in records:
        root = str(row["instance_id"])
        action = sequence_hash(row["action_sequence"])
        source = np.asarray(row["source_state_raw"], dtype=np.float64)
        terminal = np.asarray(arrays[_array_key(str(row["realized_terminal_state_ref"]))], dtype=np.float64)
        result[root][action] = terminal[:2] - source[:2]
    return result


def _effect_maps_from_sealed_outcomes(records: list[dict[str, Any]], outcome_path: Path) -> dict[str, dict[str, np.ndarray]]:
    outcomes = json.loads(outcome_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for row in records:
        key = f"{row['instance_id']}/{int(row['candidate_id'])}"
        outcome = outcomes.get(key)
        if not isinstance(outcome, dict) or "realized_terminal_state" not in outcome:
            raise ValueError(f"sealed outcome lacks realized terminal state: {key}")
        terminal = np.asarray(outcome["realized_terminal_state"], dtype=np.float64)
        source = np.asarray(row["source_state_raw"], dtype=np.float64)
        result[str(row["instance_id"])][sequence_hash(row["action_sequence"])] = terminal[:2] - source[:2]
    return result


def _mean_effect_divergence(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("paired roots have no common action sequences")
    return float(np.mean([np.linalg.norm(left[action] - right[action]) for action in shared]))


def _nearest_train_root(
    root: str, contexts: dict[str, dict[str, Any]], train_roots: list[str], mean: np.ndarray, scale: np.ndarray
) -> tuple[str, float]:
    latent = (np.asarray(contexts[root]["source_latent_model"], dtype=np.float64) - mean) / scale
    candidates = []
    for other in train_roots:
        if other == root:
            continue
        other_latent = (np.asarray(contexts[other]["source_latent_model"], dtype=np.float64) - mean) / scale
        candidates.append((float(np.linalg.norm(latent - other_latent)), other))
    if not candidates:
        raise ValueError("at least two train roots are required for alias diagnostics")
    return min(candidates, key=lambda item: (item[0], item[1]))[1], min(candidates, key=lambda item: (item[0], item[1]))[0]


def alias_conflict_diagnostic(
    source_records: list[dict[str, Any]], source_arrays: Any,
    unseen_records: list[dict[str, Any]], unseen_outcome_path: Path,
) -> dict[str, Any]:
    """Audit near-latent, same-action pairs before interpreting OOD ranking.

    Quantiles are estimated from source-bank train roots only.  The unseen
    bank is inspected after fitting, and its outcome rows never feed a model,
    threshold, or architecture choice.
    """

    source_contexts = _root_rows(source_records)
    train_roots = sorted(root for root, row in source_contexts.items() if str(row["split"]) == "train")
    if len(train_roots) < 2:
        raise ValueError("alias diagnostic requires at least two train roots")
    train_latents = np.asarray([source_contexts[root]["source_latent_model"] for root in train_roots], dtype=np.float64)
    mean, scale = train_latents.mean(axis=0), np.maximum(train_latents.std(axis=0), 1.0e-6)
    source_effects = _effect_maps_from_source(source_records, source_arrays)
    train_distances, train_divergences = [], []
    for root in train_roots:
        other, distance = _nearest_train_root(root, source_contexts, train_roots, mean, scale)
        train_distances.append(distance)
        train_divergences.append(_mean_effect_divergence(source_effects[root], source_effects[other]))
    latent_threshold = float(np.quantile(train_distances, LATENT_NEAR_QUANTILE))
    divergent_threshold = float(np.quantile(train_divergences, EFFECT_DIVERGENT_QUANTILE))

    unseen_contexts = _root_rows(unseen_records)
    unseen_effects = _effect_maps_from_sealed_outcomes(unseen_records, unseen_outcome_path)
    rows = []
    for root, row in sorted(unseen_contexts.items()):
        other, latent_distance = _nearest_train_root(root, unseen_contexts, train_roots, mean, scale)
        state_distance = float(np.linalg.norm(
            np.asarray(row["source_state_raw"], dtype=np.float64) - np.asarray(unseen_contexts[other]["source_state_raw"], dtype=np.float64)
        ))
        divergence = _mean_effect_divergence(unseen_effects[root], unseen_effects[other])
        same_actions = sorted(unseen_effects[root]) == sorted(unseen_effects[other])
        rows.append({
            "root": root,
            "split": str(row["split"]),
            "nearest_train_root": other,
            "same_action_hashes": bool(same_actions),
            "latent_distance_standardized": latent_distance,
            "physical_state_distance": state_distance,
            "mean_real_effect_divergence": divergence,
            "near_latent": bool(latent_distance <= latent_threshold),
            "divergent_real_effect": bool(divergence >= divergent_threshold),
            "alias_conflict": bool(same_actions and latent_distance <= latent_threshold and divergence >= divergent_threshold),
        })
    split_summary = {}
    for split in ("train", "calibration", "outcome"):
        selected = [row for row in rows if row["split"] == split]
        split_summary[split] = {
            "root_count": len(selected),
            "alias_conflict_count": int(sum(row["alias_conflict"] for row in selected)),
            "alias_conflict_fraction": float(np.mean([row["alias_conflict"] for row in selected])) if selected else None,
            "mean_latent_distance": float(np.mean([row["latent_distance_standardized"] for row in selected])) if selected else None,
            "mean_effect_divergence": float(np.mean([row["mean_real_effect_divergence"] for row in selected])) if selected else None,
        }
    return {
        "thresholds_train_only": {
            "latent_near_quantile": LATENT_NEAR_QUANTILE,
            "latent_distance_threshold": latent_threshold,
            "effect_divergent_quantile": EFFECT_DIVERGENT_QUANTILE,
            "effect_divergence_threshold": divergent_threshold,
        },
        "definition": "same shared action sequence, near frozen source latent, and divergent realized XY effect",
        "rows": rows,
        "by_split": split_summary,
    }


def _mechanistic_route(gains: dict[str, dict[str, float]], history: dict[str, Any], alias: dict[str, Any]) -> dict[str, Any]:
    latent = gains["frozen_source_latent_vs_action_only"]
    physical = gains["physical_state_oracle_vs_action_only"]
    conflict = float(alias["by_split"]["outcome"]["alias_conflict_fraction"] or 0.0)
    if physical["lower"] > 0.0 and latent["lower"] <= 0.0:
        route = "representation_or_history_repair_before_reenabling_state_conditioning"
        reason = "privileged state recovers held-bank ranking while frozen source latent does not"
    elif physical["lower"] <= 0.0:
        route = "action_only_fallback_for_this_action_ood_regime"
        reason = "even the privileged source-state control cannot establish a state-conditioning benefit"
    elif latent["lower"] > 0.0:
        route = "retain_latent_conditioning_and_target_action_generalization"
        reason = "frozen latent itself has a held-bank benefit; the remaining failure is not an action-only advantage"
    else:
        route = "inconclusive_more_root_disjoint_evidence_required"
        reason = "neither declared contrast clears its paired uncertainty interval"
    return {
        "route": route,
        "reason": reason,
        "outcome_alias_conflict_fraction": conflict,
        "temporal_history_status": history["status"],
        "interpretation_boundary": (
            "This is a mechanism diagnostic for one sealed PointMaze bank. A physical-state benefit is evidence of information missing from the frozen input, not proof that any particular history encoder will repair it."
        ),
    }


def run(
    source_dataset: Path, source_arrays: Path, unseen_context: Path, unseen_outcomes: Path, output: Path,
    *, epochs: int = 300, bootstrap_samples: int = 5000, device: str = "cuda",
) -> dict[str, Any]:
    if device != "cuda":
        raise ValueError("formal action-OOD observability audit requires --device cuda")
    cuda = require_cuda_allocation(device)
    source_metadata, source_records = _load_records(source_dataset)
    unseen_metadata, unseen_records = _load_records(unseen_context)
    if any("real_task_cost" in row for row in unseen_records):
        raise ValueError("sealed unseen contexts contain outcome labels before fitting")
    if unseen_metadata.get("outcomes_sealed_from_feature_construction") is not True:
        raise ValueError("unseen context does not declare sealed outcome construction")
    arrays = np.load(source_arrays, allow_pickle=False)
    history = temporal_history_status(source_records, source_metadata)
    families = list(FEATURE_ORDER)
    if history["status"] == "available":
        families.append("temporal_history")
    fit_mask = np.asarray([str(row["split"]) == "train" for row in source_records])
    if not fit_mask.any():
        raise ValueError("source dataset has no train records")
    predictions: dict[str, np.ndarray] = {}
    training: dict[str, Any] = {}
    # Everything above and within this loop is outcome blind for the held bank.
    for family in families:
        all_x, all_y = _features_and_effects(source_records, arrays, family)
        unseen_x = np.asarray([_context(row, family) for row in unseen_records], dtype=np.float32)
        predictions[family], training[family] = _ensemble_predict(
            all_x[fit_mask], all_y[fit_mask], unseen_x, epochs=epochs, device=device,
        )

    # This is the first outcome join.  It occurs only after every predeclared
    # predictor has been fitted and no branch below changes a fitted model.
    labeled = _append_outcome_labels(unseen_records, unseen_outcomes)
    selections = {
        family: {
            "calibration_diagnostic": _partition_selection(labeled, _costs(labeled, effect), "calibration"),
            "outcome": _partition_selection(labeled, _costs(labeled, effect), "outcome"),
        }
        for family, effect in predictions.items()
    }
    gains = {}
    action = selections["action_only"]["outcome"]
    for family in families:
        if family == "action_only":
            continue
        gains[f"{family}_vs_action_only"] = _paired_gain(
            selections[family]["outcome"], action,
            seed=2026102100 + len(gains), samples=bootstrap_samples,
        )
    alias = alias_conflict_diagnostic(source_records, arrays, unseen_records, unseen_outcomes)
    report = {
        "schema": SCHEMA,
        "status": "completed_mechanistic_diagnostic_not_model_selection",
        "inputs": {
            "source_dataset": str(source_dataset.resolve()),
            "source_dataset_sha256": _sha256_file(source_dataset),
            "source_arrays": str(source_arrays.resolve()),
            "source_arrays_sha256": _sha256_file(source_arrays),
            "sealed_unseen_context": str(unseen_context.resolve()),
            "sealed_unseen_context_sha256": _sha256_file(unseen_context),
            "sealed_unseen_outcomes": str(unseen_outcomes.resolve()),
            "sealed_unseen_outcomes_sha256": _sha256_file(unseen_outcomes),
        },
        "protocol": {
            "fit_roots": int(sum(str(row["split"]) == "train" for row in _root_rows(source_records).values())),
            "calibration_roots": int(sum(str(row["split"]) == "calibration" for row in _root_rows(unseen_records).values())),
            "outcome_roots": int(sum(str(row["split"]) == "outcome" for row in _root_rows(unseen_records).values())),
            "predeclared_feature_families": families,
            "physical_state_oracle": "source qpos/qvel only; privileged diagnostic control and not deployable",
            "outcome_used_for_fit_threshold_or_feature_selection": False,
            "outcomes_opened_after_all_predictions_frozen": True,
            "epochs": int(epochs), "seeds": list(SEEDS), "cuda_preflight": cuda,
        },
        "temporal_history": history,
        "selection": selections,
        "paired_outcome_regret_gain": gains,
        "alias_conflict": alias,
        "training": training,
        "mechanistic_route": _mechanistic_route(gains, history, alias),
        "claim_boundary": (
            "The action bank and evaluation roots are held out, but the physical-state arm is privileged. "
            "This identifies a repair direction; it is not visual deployment, causal world-model, or universal-transfer evidence."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "action_ood_observability_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--source-arrays", required=True, type=Path)
    parser.add_argument("--unseen-context", required=True, type=Path)
    parser.add_argument("--unseen-outcomes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps({
        "status": report["status"],
        "paired_outcome_regret_gain": report["paired_outcome_regret_gain"],
        "mechanistic_route": report["mechanistic_route"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
