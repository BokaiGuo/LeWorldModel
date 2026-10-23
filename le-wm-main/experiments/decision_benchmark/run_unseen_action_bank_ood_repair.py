"""Predeclared action-OOD repair for PointMaze intervention-effect ranking.

The preceding unseen-bank audit found that the concatenative latent/action MLP
lost to an action-only model when every candidate sequence was new.  This
runner evaluates a structural repair without changing that held action bank:

1. the original concatenative MLP;
2. the original action-only bank control;
3. an additive source-plus-action-basis ridge head; and
4. a low-rank/state-conditioned action-basis (bilinear) ridge head.

The latter two are fitted on a predeclared action-diversified intervention set
collected solely from train roots.  Architecture family and ridge grid are
fixed below; the single repair choice is made from calibration-root regret on
the already sealed unseen bank.  Outcome roots are opened only after that
choice and all predictions are frozen.

This is an offline effect/ranking repair test.  It cannot establish frozen
world-model dynamics, visual closed-loop control, or universal transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.audit_unseen_action_bank_contract import (
    audit_unseen_action_bank_contract,
    canonical_json_hash,
    sequence_hash,
    template_fingerprint,
)
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import (
    _append_outcome_labels,
    _ensemble_predict,
    _load_records,
    _observed_effect_rmse,
    _paired_gain,
    _selection,
    _selection_disagreement,
    _simulate_sequence,
    _source_rows_by_root,
)
from experiments.decision_benchmark.train_candidate_effect_adapter import (
    _cyclic_within_root,
    _features,
)
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_unseen_action_bank_ood_repair.v0.1"
DIVERSE_FAMILY = "world_fixed_diverse_basis_pulse_v1"
REPAIR_CONFIG_ORDER = (
    "additive_diverse_action_basis",
    "bilinear_diverse_action_basis",
    "bilinear_base_action_basis",
)
RIDGE_GRID = (1.0e-4, 1.0e-2, 1.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_basis(sequence: np.ndarray) -> np.ndarray:
    """Action-equivariant temporal basis, independent of candidate ID.

    Effect changes linearly in these fixed signed action statistics.  The
    bilinear head may change their coefficients with source latent, but it
    cannot assign a free lookup vector to a particular candidate identity.
    """

    actions = np.asarray(sequence, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2 or actions.shape[0] % 4:
        raise ValueError("actions must have shape [steps divisible by four, 2]")
    blocks = np.stack([part.mean(axis=0) for part in np.array_split(actions, 4)]).reshape(-1)
    time = np.linspace(-1.0, 1.0, actions.shape[0], dtype=np.float64)[:, None]
    impulse = np.mean(actions, axis=0)
    temporal_moment = np.mean(time * actions, axis=0)
    # All terms are signed and transform with the action components.  We avoid
    # action energy, which was a flexible candidate-template cue in the MLP.
    return np.concatenate((blocks, impulse, temporal_moment))


def diversified_train_action_sequences(*, count: int = 24, steps: int = 20) -> np.ndarray:
    """A fixed, train-root-only coverage bank distinct from old and held banks."""

    if count < 8 or steps < 8 or steps % 4:
        raise ValueError("count must be >=8 and steps must be divisible by four and >=8")
    width = steps // 4
    sequences: list[np.ndarray] = []
    for index in range(count):
        angle = 0.051 + 2.0 * np.pi * (index + 0.5) / count
        direct = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
        tangent = np.asarray([-direct[1], direct[0]], dtype=np.float64)
        magnitude = 0.18 + 0.045 * (index % 6)
        bend = (-1.0 if index % 2 else 1.0) * (0.08 + 0.025 * (index % 4))
        blocks = np.asarray([
            0.70 * magnitude * direct + bend * tangent,
            1.00 * magnitude * direct + 0.15 * bend * tangent,
            0.45 * magnitude * direct - bend * tangent,
            0.85 * magnitude * direct + 0.55 * bend * tangent,
        ], dtype=np.float64)
        sequence = np.repeat(blocks, width, axis=0)
        if np.max(np.abs(sequence)) > 1.0:
            raise RuntimeError("diversified action left declared PointMaze support")
        sequences.append(sequence)
    bank = np.stack(sequences)
    if len({sequence_hash(sequence) for sequence in bank}) != count:
        raise RuntimeError("diversified training bank has duplicate sequences")
    return bank


def _assert_diverse_train_bank_disjoint(
    train_records: list[dict[str, Any]],
    unseen_records: list[dict[str, Any]],
    sequences: np.ndarray,
) -> dict[str, Any]:
    historical_hashes = {sequence_hash(row["action_sequence"]) for row in train_records}
    held_hashes = {sequence_hash(row["action_sequence"]) for row in unseen_records}
    historical_templates = {template_fingerprint(row["action_sequence"]) for row in train_records}
    held_templates = {template_fingerprint(row["action_sequence"]) for row in unseen_records}
    diverse_hashes = {sequence_hash(sequence) for sequence in sequences}
    diverse_templates = {template_fingerprint(sequence) for sequence in sequences}
    if diverse_hashes & historical_hashes or diverse_hashes & held_hashes:
        raise ValueError("diverse training bank reused an old or held action hash")
    if diverse_templates & historical_templates or diverse_templates & held_templates:
        raise ValueError("diverse training bank reused an old or held action template")
    return {
        "historical_hash_overlap": [],
        "held_hash_overlap": [],
        "historical_template_overlap": [],
        "held_template_overlap": [],
        "diverse_action_hashes": sorted(diverse_hashes),
    }


def _materialize_diverse_train_interventions(
    source_records: list[dict[str, Any]],
    sequences: np.ndarray,
    output: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect intervention labels only from the immutable train-root split."""

    roots = _source_rows_by_root(source_records)
    train_sources = [row for _root, row in sorted(roots.items()) if str(row["split"]) == "train"]
    if not train_sources:
        raise ValueError("source bank has no train roots")
    latent, action_values, effects, audit_rows = [], [], [], []
    for source in train_sources:
        state = np.asarray(source["source_state_raw"], dtype=np.float64)
        for action_index, sequence in enumerate(sequences):
            terminal = _simulate_sequence(state, sequence)
            effect = terminal[:2] - state[:2]
            latent.append(np.asarray(source["source_latent_model"], dtype=np.float64))
            action_values.append(action_basis(sequence))
            effects.append(effect)
            audit_rows.append({
                "instance_id": str(source["instance_id"]),
                "split": "train",
                "diverse_action_id": int(2000 + action_index),
                "action_hash": sequence_hash(sequence),
                "effect": effect.tolist(),
            })
    report = {
        "schema": SCHEMA,
        "family": DIVERSE_FAMILY,
        "train_root_count": len(train_sources),
        "action_count": int(len(sequences)),
        "record_count": len(audit_rows),
        "outcome_or_calibration_roots_used": False,
        "records": audit_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "diverse_train_interventions.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return np.asarray(latent), np.asarray(action_values), np.asarray(effects), report


@dataclass(frozen=True)
class RidgeEffectHead:
    mode: str
    ridge: float
    state_mean: np.ndarray
    state_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray
    coefficients: np.ndarray

    def _design(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        source = (np.asarray(state, dtype=np.float64) - self.state_mean) / self.state_scale
        basis = (np.asarray(action, dtype=np.float64) - self.action_mean) / self.action_scale
        pieces = [np.ones((len(source), 1), dtype=np.float64), basis, source]
        if self.mode == "bilinear":
            pieces.append((source[:, :, None] * basis[:, None, :]).reshape(len(source), -1))
        elif self.mode != "additive":
            raise ValueError(f"unsupported head mode: {self.mode}")
        return np.concatenate(pieces, axis=1)

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self._design(state, action) @ self.coefficients


def fit_ridge_effect_head(
    state: np.ndarray,
    action: np.ndarray,
    effect: np.ndarray,
    *,
    mode: str,
    ridge: float,
) -> RidgeEffectHead:
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    effect = np.asarray(effect, dtype=np.float64)
    if state.ndim != 2 or action.ndim != 2 or effect.shape != (len(state), 2):
        raise ValueError("state/action/effect arrays have incompatible shapes")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")
    state_mean, action_mean = state.mean(axis=0), action.mean(axis=0)
    state_scale = np.maximum(state.std(axis=0), 1.0e-4)
    action_scale = np.maximum(action.std(axis=0), 1.0e-4)
    provisional = RidgeEffectHead(mode, float(ridge), state_mean, state_scale, action_mean, action_scale, np.empty((0, 2)))
    design = provisional._design(state, action)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ effect)
    return RidgeEffectHead(mode, float(ridge), state_mean, state_scale, action_mean, action_scale, coefficients)


def _context_state_action(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([row["source_latent_model"] for row in records], dtype=np.float64),
        np.asarray([action_basis(np.asarray(row["action_sequence"], dtype=np.float64)) for row in records], dtype=np.float64),
    )


def _calibration_choice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose only with calibration regret; stable lexical order breaks ties."""

    if not rows:
        raise ValueError("no repair candidates were evaluated on calibration roots")
    order = {name: index for index, name in enumerate(REPAIR_CONFIG_ORDER)}
    selected = min(
        rows,
        key=lambda row: (float(row["calibration_mean_regret"]), order[row["config"]], float(row["ridge"])),
    )
    return {"rule": "minimum calibration mean regret, then predeclared config order, then ridge", "selected": selected, "candidates": rows}


def run(
    source_dataset: Path,
    source_arrays: Path,
    unseen_context: Path,
    unseen_outcomes: Path,
    output: Path,
    *,
    epochs: int = 300,
    bootstrap_samples: int = 5000,
    device: str = "cuda",
    diverse_action_count: int = 24,
) -> dict[str, Any]:
    if device != "cuda":
        raise ValueError("formal OOD repair requires --device cuda")
    cuda = require_cuda_allocation(device)
    source_metadata, source_records = _load_records(source_dataset)
    _unseen_metadata, unseen_contexts = _load_records(unseen_context)
    if source_metadata.get("shared_world_action_bank") is not True:
        raise ValueError("source bank must declare shared_world_action_bank=true")
    if any("real_task_cost" in row for row in unseen_contexts):
        raise ValueError("unseen contexts contain outcomes before fitting")
    train_calibration_records = [row for row in source_records if str(row["split"]) in {"train", "calibration"}]
    contract = audit_unseen_action_bank_contract(train_calibration_records, unseen_contexts)
    diverse_sequences = diversified_train_action_sequences(count=diverse_action_count)
    diverse_contract = _assert_diverse_train_bank_disjoint(train_calibration_records, unseen_contexts, diverse_sequences)
    diverse_state, diverse_action, diverse_effect, diverse_audit = _materialize_diverse_train_interventions(
        source_records, diverse_sequences, output,
    )

    # Original fixed-bank intervention data, fit roots only.
    arrays = np.load(source_arrays, allow_pickle=False)
    full_x, full_y, _roots, _goals = _features(source_records, arrays, "frozen_latent_action")
    fit_mask = np.asarray([str(row["split"]) == "train" for row in source_records])
    old_train_records = [row for row, keep in zip(source_records, fit_mask) if keep]
    old_state, old_action = _context_state_action(old_train_records)
    old_effect = full_y[fit_mask].astype(np.float64)
    unseen_state, unseen_action = _context_state_action(unseen_contexts)

    # Exact current concatenative adapter and exact action-only control.
    full_effect, full_training = _ensemble_predict(
        full_x[fit_mask], full_y[fit_mask],
        np.asarray([
            np.concatenate((np.asarray(row["source_latent_model"], dtype=np.float32), np.asarray(row["action_sequence"], dtype=np.float32).reshape(-1)))
            for row in unseen_contexts
        ], dtype=np.float32),
        epochs=epochs, device=device,
    )
    latent_width = len(source_records[0]["source_latent_model"])
    bank_effect, bank_training = _ensemble_predict(
        full_x[fit_mask, latent_width:], full_y[fit_mask],
        np.asarray([np.asarray(row["action_sequence"], dtype=np.float32).reshape(-1) for row in unseen_contexts], dtype=np.float32),
        epochs=epochs, device=device,
    )

    # All repair candidates and hyperparameters are declared above.  Building
    # them uses training interventions only; no calibration/outcome labels have
    # yet been opened.
    training_sets = {
        "additive_diverse_action_basis": (np.concatenate((old_state, diverse_state)), np.concatenate((old_action, diverse_action)), np.concatenate((old_effect, diverse_effect)), "additive"),
        "bilinear_diverse_action_basis": (np.concatenate((old_state, diverse_state)), np.concatenate((old_action, diverse_action)), np.concatenate((old_effect, diverse_effect)), "bilinear"),
        "bilinear_base_action_basis": (old_state, old_action, old_effect, "bilinear"),
    }
    candidate_heads: dict[tuple[str, float], RidgeEffectHead] = {}
    candidate_effects: dict[tuple[str, float], np.ndarray] = {}
    for config in REPAIR_CONFIG_ORDER:
        state, action, effect, mode = training_sets[config]
        for ridge in RIDGE_GRID:
            head = fit_ridge_effect_head(state, action, effect, mode=mode, ridge=ridge)
            candidate_heads[(config, ridge)] = head
            candidate_effects[(config, ridge)] = head.predict(unseen_state, unseen_action)

    # Sealed labels are opened at this point.  Calibration roots choose one
    # frozen repair; outcome roots are still unread by the choice rule.
    labeled = _append_outcome_labels(unseen_contexts, unseen_outcomes)
    calibration_mask = np.asarray([str(row["split"]) == "calibration" for row in labeled])
    outcome_mask = np.asarray([str(row["split"]) == "outcome" for row in labeled])
    source_xy = np.asarray([row["source_state_raw"][:2] for row in labeled], dtype=np.float64)
    goals = np.asarray([row["goal_raw"][:2] for row in labeled], dtype=np.float64)
    calibration_records = [row for row, keep in zip(labeled, calibration_mask) if keep]
    outcome_records = [row for row, keep in zip(labeled, outcome_mask) if keep]
    calibration_rows = []
    for config in REPAIR_CONFIG_ORDER:
        for ridge in RIDGE_GRID:
            effect = candidate_effects[(config, ridge)]
            selection = _selection(calibration_records, np.linalg.norm(source_xy[calibration_mask] + effect[calibration_mask] - goals[calibration_mask], axis=1))
            calibration_rows.append({
                "config": config,
                "ridge": float(ridge),
                "calibration_mean_regret": float(selection["mean_regret"]),
                "calibration_zero_regret_fraction": float(selection["zero_regret_fraction"]),
            })
    selection_rule = _calibration_choice(calibration_rows)
    selected_config = str(selection_rule["selected"]["config"])
    selected_ridge = float(selection_rule["selected"]["ridge"])
    repair_effect = candidate_effects[(selected_config, selected_ridge)]

    full_cost = np.linalg.norm(source_xy + full_effect - goals, axis=1)
    bank_cost = np.linalg.norm(source_xy + bank_effect - goals, axis=1)
    repair_cost = np.linalg.norm(source_xy + repair_effect - goals, axis=1)
    full_selection = _selection(outcome_records, full_cost[outcome_mask])
    bank_selection = _selection(outcome_records, bank_cost[outcome_mask])
    repair_selection = _selection(outcome_records, repair_cost[outcome_mask])
    no_dynamics_selection = _selection(outcome_records, np.zeros(int(outcome_mask.sum()), dtype=np.float64))
    shuffled_selection = _selection(outcome_records, _cyclic_within_root(repair_cost[outcome_mask], outcome_records))
    controls = {
        "bank_only": _paired_gain(repair_selection, bank_selection, seed=2026102111, samples=bootstrap_samples),
        "no_dynamics": _paired_gain(repair_selection, no_dynamics_selection, seed=2026102112, samples=bootstrap_samples),
        "shuffled_successor": _paired_gain(repair_selection, shuffled_selection, seed=2026102113, samples=bootstrap_samples),
        "concatenative_adapter": _paired_gain(repair_selection, full_selection, seed=2026102114, samples=bootstrap_samples),
    }
    per_candidate_outcome = {}
    for config in REPAIR_CONFIG_ORDER:
        for ridge in RIDGE_GRID:
            effect = candidate_effects[(config, ridge)]
            select = _selection(outcome_records, np.linalg.norm(source_xy[outcome_mask] + effect[outcome_mask] - goals[outcome_mask], axis=1))
            per_candidate_outcome[f"{config}/ridge={ridge:g}"] = {
                "mean_regret": select["mean_regret"],
                "effect_rmse": _observed_effect_rmse(outcome_records, effect[outcome_mask], unseen_outcomes),
            }
    report = {
        "schema": SCHEMA,
        "status": (
            "passed_ood_generalization_repair"
            if float(controls["bank_only"]["lower"]) > 0.0
            else "failed_ood_generalization_repair"
        ),
        "source": {
            "dataset": str(source_dataset.resolve()), "dataset_sha256": _sha256_file(source_dataset),
            "arrays": str(source_arrays.resolve()), "arrays_sha256": _sha256_file(source_arrays),
            "unseen_context": str(unseen_context.resolve()), "unseen_context_sha256": _sha256_file(unseen_context),
            "unseen_outcomes": str(unseen_outcomes.resolve()), "unseen_outcomes_sha256": _sha256_file(unseen_outcomes),
        },
        "contracts": {"unseen_bank": contract, "diverse_training_bank": diverse_contract},
        "diversified_training": diverse_audit,
        "predeclared_selection": {
            "config_order": list(REPAIR_CONFIG_ORDER), "ridge_grid": list(RIDGE_GRID),
            **selection_rule,
        },
        "protocol": {
            "fit_roots": contract["train_root_counts"]["train"],
            "calibration_roots": contract["unseen_root_counts"]["calibration"],
            "outcome_roots": contract["unseen_root_counts"]["outcome"],
            "outcome_used_for_architecture_hyperparameter_or_model_selection": False,
            "calibration_used_for_repair_selection": True,
            "cuda_preflight": cuda,
            "epochs": int(epochs),
        },
        "selection": {
            "repair_selected": repair_selection, "bank_only": bank_selection,
            "concatenative_adapter": full_selection, "no_dynamics": no_dynamics_selection,
            "shuffled_successor": shuffled_selection,
        },
        "effect_rmse": {
            "repair_selected": _observed_effect_rmse(outcome_records, repair_effect[outcome_mask], unseen_outcomes),
            "bank_only": _observed_effect_rmse(outcome_records, bank_effect[outcome_mask], unseen_outcomes),
            "concatenative_adapter": _observed_effect_rmse(outcome_records, full_effect[outcome_mask], unseen_outcomes),
        },
        "paired_regret_gain": controls,
        "diagnostics": {
            "repair_vs_bank_only": _selection_disagreement(repair_selection, bank_selection),
            "all_predeclared_repair_candidates_outcome": per_candidate_outcome,
        },
        "training": {"concatenative_adapter": full_training, "bank_only": bank_training},
        "claim_boundary": (
            "A calibration-selected, train-intervention action-OOD repair test. A positive result "
            "would establish only fixed-bank held-root action-ranking transfer, not frozen dynamics "
            "or visual closed-loop planning."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "unseen_action_bank_ood_repair_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-arrays", type=Path, required=True)
    parser.add_argument("--unseen-context", type=Path, required=True)
    parser.add_argument("--unseen-outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--diverse-action-count", type=int, default=24)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps({
        "status": report["status"],
        "selected": report["predeclared_selection"]["selected"],
        "effect_rmse": report["effect_rmse"],
        "paired_regret_gain": report["paired_regret_gain"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed_ood_generalization_repair" else 2


if __name__ == "__main__":
    raise SystemExit(main())
