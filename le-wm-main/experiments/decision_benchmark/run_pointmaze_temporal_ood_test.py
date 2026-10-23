"""Test whether causal frozen history repairs PointMaze action-OOD ranking.

The predeclared primary estimand is paired outcome-root regret gain of
``history_latent_action`` over ``action_only``.  The test passes only if its
bootstrap lower confidence bound exceeds zero *and* the causal history removes
at least one current-latent alias conflict.  Current-frame frozen latent and
privileged source state are diagnostic controls, not selectable alternatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.build_pointmaze_temporal_ood_bank import (
    ACTION_OOD_FAMILY,
    HISTORY_WINDOWS,
    temporal_template_fingerprint,
)
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import (
    _append_outcome_labels,
    _bootstrap,
    _load_records,
    _selection,
)
from experiments.decision_benchmark.train_candidate_effect_adapter import (
    SEEDS,
    _array_key,
    fit_effect_model,
    predict_effect_model,
)
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_temporal_ood_test.v1"
FEATURES = ("action_only", "current_latent_action", "history_latent_action", "privileged_state_action")
ALIAS_NEAR_QUANTILE = 0.25
ALIAS_DIVERGENT_QUANTILE = 0.75


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _roots(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    required = ("instance_id", "split", "source_latent_model", "source_history_model", "source_state_raw", "goal_raw")
    for row in records:
        if any(key not in row for key in required):
            raise ValueError("record lacks a required root context field")
        root = str(row["instance_id"])
        previous = values.setdefault(root, row)
        for field in required[1:]:
            if field == "split":
                if str(row[field]) != str(previous[field]):
                    raise ValueError(f"root has candidate-specific {field}: {root}")
            elif not np.allclose(np.asarray(row[field]), np.asarray(previous[field]), rtol=0.0, atol=1e-10):
                raise ValueError(f"root has candidate-specific {field}: {root}")
    return values


def _history_actions(row: dict[str, Any]) -> np.ndarray:
    actions = np.asarray(row.get("history_actions_raw"), dtype=np.float64)
    frames = [int(value) for value in row.get("history_frame_indices", [])]
    source_frame = int(row.get("source_frame", -1))
    if actions.shape != ((HISTORY_WINDOWS - 1) * 5, 2):
        raise ValueError("history action shape is not three complete causal frame intervals")
    if len(frames) != HISTORY_WINDOWS or frames[-1] != source_frame or any(right <= left for left, right in zip(frames, frames[1:])):
        raise ValueError("history frame ledger is not a causal sequence ending at the decision frame")
    if int(row["history_action_interval"][1]) != source_frame or int(row["history_action_interval"][0]) != frames[0]:
        raise ValueError("history action interval does not end before the candidate sequence")
    return actions


def _feature(row: dict[str, Any], family: str) -> np.ndarray:
    candidate = np.asarray(row["action_sequence"], dtype=np.float64).reshape(-1)
    if family == "action_only":
        context = np.empty((0,), dtype=np.float64)
    elif family == "current_latent_action":
        context = np.asarray(row["source_latent_model"], dtype=np.float64).reshape(-1)
    elif family == "history_latent_action":
        context = np.concatenate((
            np.asarray(row["source_history_model"], dtype=np.float64).reshape(-1),
            _history_actions(row).reshape(-1),
        ))
    elif family == "privileged_state_action":
        context = np.asarray(row["source_state_raw"], dtype=np.float64).reshape(-1)
    else:
        raise ValueError(f"unsupported family: {family}")
    return np.concatenate((context, candidate)).astype(np.float32)


def _training_features(records: list[dict[str, Any]], arrays: Any, family: str) -> tuple[np.ndarray, np.ndarray]:
    values, effects = [], []
    for row in records:
        values.append(_feature(row, family))
        source = np.asarray(row["source_state_raw"], dtype=np.float64)
        terminal = np.asarray(arrays[_array_key(str(row["realized_terminal_state_ref"]))], dtype=np.float64)
        effects.append(terminal[:2] - source[:2])
    return np.asarray(values, dtype=np.float32), np.asarray(effects, dtype=np.float32)


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, target_x: np.ndarray, *, epochs: int, device: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values, provenance = [], []
    for seed in SEEDS:
        model, fit = fit_effect_model(train_x, train_y, seed=seed, epochs=epochs, device_name=device)
        values.append(predict_effect_model(model, fit, target_x))
        provenance.append({key: value for key, value in fit.items() if key not in {"x_mean", "x_scale", "y_mean", "y_scale"}})
    return np.mean(np.stack(values), axis=0), provenance


def _costs(records: list[dict[str, Any]], effect: np.ndarray) -> np.ndarray:
    state = np.asarray([row["source_state_raw"][:2] for row in records], dtype=np.float64)
    goal = np.asarray([row["goal_raw"][:2] for row in records], dtype=np.float64)
    return np.linalg.norm(state + np.asarray(effect, dtype=np.float64) - goal, axis=1)


def _partition(records: list[dict[str, Any]], values: np.ndarray, split: str) -> dict[str, Any]:
    mask = np.asarray([str(row["split"]) == split for row in records])
    if not mask.any():
        raise ValueError(f"no {split} roots")
    return _selection([row for row, keep in zip(records, mask) if keep], np.asarray(values)[mask])


def _paired_gain(method: dict[str, Any], control: dict[str, Any], *, seed: int, samples: int) -> dict[str, float]:
    roots = sorted(method["roots"])
    if roots != sorted(control["roots"]):
        raise ValueError("paired methods must evaluate exactly the same roots")
    gain = np.asarray([control["roots"][root]["regret"] - method["roots"][root]["regret"] for root in roots], dtype=np.float64)
    return _bootstrap(gain, roots, seed=seed, samples=samples)


def _effects(records: list[dict[str, Any]], outcomes: Path) -> dict[str, dict[str, np.ndarray]]:
    payload = json.loads(outcomes.read_text(encoding="utf-8"))
    values: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for row in records:
        key = f"{row['instance_id']}/{int(row['candidate_id'])}"
        terminal = np.asarray(payload[key]["realized_terminal_state"], dtype=np.float64)
        values[str(row["instance_id"])][str(row["candidate_action_hash"])] = terminal[:2] - np.asarray(row["source_state_raw"][:2], dtype=np.float64)
    return values


def _mean_divergence(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError("roots have no shared candidate action hash")
    return float(np.mean([np.linalg.norm(left[key] - right[key]) for key in keys]))


def _nearest(context: dict[str, np.ndarray], root: str, train_roots: list[str], mean: np.ndarray, scale: np.ndarray) -> tuple[str, float]:
    value = (context[root] - mean) / scale
    distances = [(float(np.linalg.norm(value - (context[other] - mean) / scale)), other) for other in train_roots if other != root]
    if not distances:
        raise ValueError("alias diagnostic requires multiple train roots")
    return min(distances, key=lambda item: (item[0], item[1]))[1], min(distances, key=lambda item: (item[0], item[1]))[0]


def alias_resolution(records: list[dict[str, Any]], outcomes: Path) -> dict[str, Any]:
    """Use train-only thresholds to test if history separates current aliases."""

    roots = _roots(records)
    train = sorted(root for root, row in roots.items() if str(row["split"]) == "train")
    current = {root: np.asarray(row["source_latent_model"], dtype=np.float64).reshape(-1) for root, row in roots.items()}
    history = {root: np.concatenate((np.asarray(row["source_history_model"], dtype=np.float64).reshape(-1), _history_actions(row).reshape(-1))) for root, row in roots.items()}
    effect = _effects(records, outcomes)

    def thresholds(context: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float, float]:
        matrix = np.asarray([context[root] for root in train], dtype=np.float64)
        mean, scale = matrix.mean(0), np.maximum(matrix.std(0), 1.0e-6)
        distances, divergences = [], []
        for root in train:
            other, distance = _nearest(context, root, train, mean, scale)
            distances.append(distance)
            divergences.append(_mean_divergence(effect[root], effect[other]))
        return mean, scale, float(np.quantile(distances, ALIAS_NEAR_QUANTILE)), float(np.quantile(divergences, ALIAS_DIVERGENT_QUANTILE))

    current_mean, current_scale, current_near, current_divergent = thresholds(current)
    history_mean, history_scale, history_near, history_divergent = thresholds(history)
    rows = []
    for root in sorted(roots):
        current_neighbor, current_distance = _nearest(current, root, train, current_mean, current_scale)
        history_neighbor, history_distance = _nearest(history, root, train, history_mean, history_scale)
        current_effect_divergence = _mean_divergence(effect[root], effect[current_neighbor])
        history_effect_divergence = _mean_divergence(effect[root], effect[history_neighbor])
        current_alias = bool(current_distance <= current_near and current_effect_divergence >= current_divergent)
        history_alias = bool(history_distance <= history_near and history_effect_divergence >= history_divergent)
        rows.append({
            "root": root,
            "split": str(roots[root]["split"]),
            "current_nearest_train_root": current_neighbor,
            "history_nearest_train_root": history_neighbor,
            "current_distance": current_distance,
            "history_distance": history_distance,
            "effect_divergence_against_current_neighbor": current_effect_divergence,
            "effect_divergence_against_history_neighbor": history_effect_divergence,
            "current_alias_conflict": current_alias,
            "history_alias_conflict": history_alias,
            "alias_resolved_by_history": bool(current_alias and not history_alias),
        })
    summary = {}
    for split in ("train", "calibration", "outcome"):
        selected = [row for row in rows if row["split"] == split]
        summary[split] = {
            "root_count": len(selected),
            "current_alias_conflicts": int(sum(row["current_alias_conflict"] for row in selected)),
            "history_alias_conflicts": int(sum(row["history_alias_conflict"] for row in selected)),
            "history_resolved_alias_conflicts": int(sum(row["alias_resolved_by_history"] for row in selected)),
        }
    return {
        "thresholds_from_train_roots_only": {
            "current_near_distance": current_near,
            "history_near_distance": history_near,
            "current_effect_divergence": current_divergent,
            "history_effect_divergence": history_divergent,
            "near_quantile": ALIAS_NEAR_QUANTILE,
            "divergent_quantile": ALIAS_DIVERGENT_QUANTILE,
        },
        "rows": rows,
        "by_split": summary,
    }


def _validate_contract(source_records: list[dict[str, Any]], metadata: dict[str, Any], ood_records: list[dict[str, Any]], ood_metadata: dict[str, Any]) -> dict[str, Any]:
    if int(metadata.get("history_windows", 0)) != HISTORY_WINDOWS:
        raise ValueError("training bank lacks predeclared four-frame history")
    if ood_metadata.get("history_is_causal") is not True or ood_metadata.get("history_actions_are_strictly_pre_candidate") is not True:
        raise ValueError("temporal OOD context does not declare the causal history contract")
    if ood_metadata.get("candidate_family") != ACTION_OOD_FAMILY:
        raise ValueError("wrong temporal OOD candidate family")
    source_hashes = {hashlib.sha256(np.ascontiguousarray(np.asarray(row["action_sequence"], dtype=np.float64)).tobytes()).hexdigest() for row in source_records}
    test_hashes = {str(row["candidate_action_hash"]) for row in ood_records}
    source_templates = {temporal_template_fingerprint(np.asarray(row["action_sequence"], dtype=np.float64)) for row in source_records}
    test_templates = {str(row["candidate_template_fingerprint"]) for row in ood_records}
    overlap_roots = set(str(row["instance_id"]) for row in source_records if str(row["split"]) == "train") & set(str(row["instance_id"]) for row in ood_records if str(row["split"]) == "outcome")
    if overlap_roots or source_hashes & test_hashes or source_templates & test_templates:
        raise ValueError("temporal OOD contract failed: root/action/template overlap")
    for row in ood_records:
        _history_actions(row)
        if "real_task_cost" in row or "realized_terminal_state" in row:
            raise ValueError("outcome leaked into the causal OOD context")
    return {
        "train_outcome_root_overlap": sorted(overlap_roots),
        "source_candidate_action_hash_overlap": sorted(source_hashes & test_hashes),
        "source_template_overlap": sorted(source_templates & test_templates),
        "history_causal": True,
        "context_outcome_free_before_fit": True,
        "root_counts": {split: sum(str(row["split"]) == split for row in _roots(ood_records).values()) for split in ("train", "calibration", "outcome")},
    }


def run(
    source_dataset: Path,
    source_arrays: Path,
    temporal_context: Path,
    temporal_outcomes: Path,
    output: Path,
    *,
    epochs: int = 300,
    bootstrap_samples: int = 5000,
    device: str = "cuda",
) -> dict[str, Any]:
    if device != "cuda":
        raise ValueError("formal temporal OOD test requires --device cuda")
    cuda = require_cuda_allocation(device)
    source_metadata, source_records = _load_records(source_dataset)
    ood_metadata, ood_contexts = _load_records(temporal_context)
    contract = _validate_contract(source_records, source_metadata, ood_contexts, ood_metadata)
    arrays = np.load(source_arrays, allow_pickle=False)
    train_records = [row for row in source_records if str(row["split"]) == "train"]
    if not train_records:
        raise ValueError("source bank has no train roots")

    predictions, training = {}, {}
    for family in FEATURES:
        train_x, train_y = _training_features(train_records, arrays, family)
        eval_x = np.asarray([_feature(row, family) for row in ood_contexts], dtype=np.float32)
        predictions[family], training[family] = _fit_predict(train_x, train_y, eval_x, epochs=epochs, device=device)

    # Sole post-fit join: no candidate outcome was visible while constructing,
    # fitting, or choosing these four predeclared feature families.
    labeled = _append_outcome_labels(ood_contexts, temporal_outcomes)
    selections = {family: {split: _partition(labeled, _costs(labeled, effect), split) for split in ("calibration", "outcome")} for family, effect in predictions.items()}
    gains = {family: _paired_gain(selections[family]["outcome"], selections["action_only"]["outcome"], seed=2026102150 + offset, samples=bootstrap_samples) for offset, family in enumerate(FEATURES) if family != "action_only"}
    calibration_history_gain = _paired_gain(selections["history_latent_action"]["calibration"], selections["action_only"]["calibration"], seed=2026102191, samples=bootstrap_samples)
    # A calibration point estimate is not a deployment authorization.  Keep
    # the runner fail-closed: only a positive paired lower interval may permit
    # history at runtime.  This does not alter the predeclared held-out
    # comparison below, which reports all four arms regardless of release.
    calibration_policy = "history_permitted" if float(calibration_history_gain["lower"]) > 0.0 else "history_rejected_action_only_fallback"
    alias = alias_resolution(labeled, temporal_outcomes)
    primary = gains["history_latent_action"]
    alias_removed = alias["by_split"]["outcome"]["history_resolved_alias_conflicts"] > 0
    status = "passed_causal_history_repair" if float(primary["lower"]) > 0.0 and alias_removed else "failed_causal_history_repair"
    report = {
        "schema": SCHEMA,
        "status": status,
        "inputs": {
            "source_dataset": str(source_dataset.resolve()), "source_dataset_sha256": _sha256(source_dataset),
            "source_arrays": str(source_arrays.resolve()), "source_arrays_sha256": _sha256(source_arrays),
            "temporal_context": str(temporal_context.resolve()), "temporal_context_sha256": _sha256(temporal_context),
            "sealed_outcomes": str(temporal_outcomes.resolve()), "sealed_outcomes_sha256": _sha256(temporal_outcomes),
        },
        "contract": contract,
        "protocol": {
            "train_roots_only_for_fitting": contract["root_counts"]["train"],
            "calibration_roots_only_for_frozen_policy": contract["root_counts"]["calibration"],
            "outcome_roots_held_out": contract["root_counts"]["outcome"],
            "predeclared_feature_families": list(FEATURES),
            "history_input": "four causal frozen frame latents plus executed actions from the three preceding frame intervals",
            "privileged_control": "source qpos/qvel plus candidate action, evaluation-only",
            "outcomes_opened_after_all_predictions_frozen": True,
            "epochs": int(epochs), "seeds": list(SEEDS), "cuda_preflight": cuda,
        },
        "selection": selections,
        "paired_outcome_regret_gain_over_action_only": gains,
        "calibration_only_history_policy": {"decision": calibration_policy, "paired_regret_gain": calibration_history_gain},
        "alias_resolution": alias,
        "predeclared_success_rule": {
            "rule": "history paired held-out regret lower CI > 0 and at least one held-out current-latent alias conflict resolved by history",
            "history_gain_lower": float(primary["lower"]),
            "outcome_aliases_resolved": alias["by_split"]["outcome"]["history_resolved_alias_conflicts"],
            "satisfied": status == "passed_causal_history_repair",
        },
        "training": training,
        "claim_boundary": "A root-disjoint, held-action-bank PointMaze mechanism test. The state arm is privileged; success would support causal frozen-history repair in this bank, not universal transfer or visual closed-loop planning.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "temporal_ood_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-arrays", type=Path, required=True)
    parser.add_argument("--temporal-context", type=Path, required=True)
    parser.add_argument("--temporal-outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps({"status": report["status"], "success_rule": report["predeclared_success_rule"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed_causal_history_repair" else 2


if __name__ == "__main__":
    raise SystemExit(main())
