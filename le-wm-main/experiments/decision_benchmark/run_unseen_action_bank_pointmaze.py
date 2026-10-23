"""Run the PointMaze action-sequence generalization kill test.

This experiment attacks the strongest alternative explanation for the existing
intervention-supervised effect-adapter result: an MLP may recognize the small
fixed candidate bank instead of learning a source-latent/action effect map.

The adapter is fitted on the legacy shared world-action bank only.  It is then
asked to rank a new, deterministic, world-coordinate bank whose numeric IDs,
canonical sequence hashes, and coarse temporal templates are all disjoint from
training.  Root partitions remain immutable train/calibration/outcome splits.

The held-bank outcomes are materialized separately and are opened only after
the two models have been fitted and their predictions computed.  This is an
offline action-ranking consequence test, not a frozen-dynamics or closed-loop
planning result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark.audit_unseen_action_bank_contract import (
    audit_unseen_action_bank_contract,
    canonical_json_hash,
    sequence_hash,
)
from experiments.decision_benchmark.train_candidate_effect_adapter import (
    SEEDS,
    _bootstrap,
    _cyclic_within_root,
    _features,
    fit_effect_model,
    predict_effect_model,
)
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
    _legacy_pointmaze_xml,
)
from experiments.gpu_preflight import require_cuda_allocation


SCHEMA = "pointmaze_unseen_action_bank_effect_adapter.v0.1"
UNSEEN_FAMILY = "world_fixed_modulated_turn_pulse_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path}: expected a candidate dataset with records")
    records = [dict(row) for row in payload["records"]]
    if not records:
        raise ValueError(f"{path}: empty candidate dataset")
    return dict(payload.get("metadata", {})), records


def _simulate_sequence(source_state: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    """Run exactly one predeclared sequence from a recorded PointMaze root."""

    import mujoco

    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    data = mujoco.MjData(model)
    state = np.asarray(source_state, dtype=np.float64)
    if state.shape != (4,):
        raise ValueError("PointMaze source_state_raw must have shape [4]")
    data.qpos[:] = state[:2]
    data.qvel[:] = state[2:]
    mujoco.mj_forward(model, data)
    for action in np.asarray(sequence, dtype=np.float64):
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
    return np.concatenate((np.asarray(data.qpos[:2]), np.asarray(data.qvel[:2])))


def unseen_action_sequences(*, count: int = 16, steps: int = 20) -> np.ndarray:
    """Create a predeclared bank unlike paired-order training templates.

    Every candidate contains four distinct five-step pulses with a small phase
    offset.  The bank is world fixed, shared across roots, and does not use a
    source goal or any test outcome.  It is intentionally not a permutation or
    relabeling of the historical paired-order bank.
    """

    if count < 4 or steps < 8 or steps % 4:
        raise ValueError("count must be >=4 and steps must be divisible by four and >=8")
    block = steps // 4
    sequences: list[np.ndarray] = []
    for index in range(count):
        angle = 0.137 + (2.0 * np.pi * index / count)
        direct = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
        tangent = np.asarray([-direct[1], direct[0]], dtype=np.float64)
        magnitude = 0.40 + 0.06 * (index % 5)
        lateral = (-1.0 if (index // 2) % 2 else 1.0) * (0.17 + 0.03 * (index % 3))
        pulses = np.asarray([
            magnitude * direct + lateral * tangent,
            0.55 * magnitude * direct - 0.80 * lateral * tangent,
            0.80 * magnitude * direct + 0.35 * lateral * tangent,
            0.35 * magnitude * direct - 0.45 * lateral * tangent,
        ], dtype=np.float64)
        sequence = np.repeat(pulses, block, axis=0)
        if np.max(np.abs(sequence)) > 1.0:
            raise RuntimeError("predeclared unseen action exceeded PointMaze control support")
        sequences.append(sequence)
    bank = np.stack(sequences, axis=0)
    hashes = [sequence_hash(value) for value in bank]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("predeclared unseen bank contains duplicate action sequences")
    return bank


def _source_rows_by_root(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    required = (
        "instance_id", "split", "source_state_raw", "source_latent_model",
        "goal_raw", "start_episode_id",
    )
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        if any(key not in row for key in required):
            raise ValueError("source candidate record lacks PointMaze context fields")
        root = str(row["instance_id"])
        if root not in result:
            result[root] = row
            continue
        reference = result[root]
        if str(reference["split"]) != str(row["split"]):
            raise ValueError(f"source root has mixed splits: {root}")
        for key in ("source_state_raw", "source_latent_model", "goal_raw"):
            if not np.allclose(np.asarray(reference[key]), np.asarray(row[key]), rtol=0.0, atol=1e-10):
                raise ValueError(f"source root has candidate-conditioned {key}: {root}")
    return result


def materialize_unseen_bank(
    source_records: list[dict[str, Any]],
    output: Path,
    *,
    candidate_id_offset: int = 1000,
    count: int = 16,
    steps: int = 20,
) -> tuple[Path, Path, Path]:
    """Write outcome-blind contexts and separately sealed realized outcomes."""

    roots = _source_rows_by_root(source_records)
    if set(str(row.get("split")) for row in roots.values()) != {"train", "calibration", "outcome"}:
        raise ValueError("source bank must contain root-disjoint train/calibration/outcome roots")
    sequences = unseen_action_sequences(count=count, steps=steps)
    contexts: list[dict[str, Any]] = []
    outcomes: dict[str, dict[str, Any]] = {}
    realized: dict[str, np.ndarray] = {}
    for root, source in sorted(roots.items()):
        source_state = np.asarray(source["source_state_raw"], dtype=np.float64)
        goal = np.asarray(source["goal_raw"], dtype=np.float64)
        for offset, sequence in enumerate(sequences):
            candidate_id = int(candidate_id_offset + offset)
            key = f"{root}/{candidate_id}"
            terminal = _simulate_sequence(source_state, sequence)
            # Contexts intentionally omit real_task_cost and realized state.
            # A later, post-fit outcome join is the first code path that opens
            # this label map.
            context = {
                "instance_id": root,
                "split": str(source["split"]),
                "environment": "PointMaze",
                "start_episode_id": str(source["start_episode_id"]),
                "source_frame": int(source.get("source_frame", 5)),
                "context_length": int(source.get("context_length", 2)),
                "goal_id": str(source.get("goal_id", "legacy_pointmaze_goal_1p2_1p2")),
                "candidate_id": candidate_id,
                "candidate_generator": UNSEEN_FAMILY,
                "candidate_bank_hash": canonical_json_hash([sequence_hash(item) for item in sequences]),
                "action_sequence": sequence.tolist(),
                "total_action_steps": int(steps),
                "source_state_raw": source_state.tolist(),
                "source_latent_model": list(source["source_latent_model"]),
                "goal_raw": goal.tolist(),
                "source_observation_ref": str(source.get("source_observation_ref", "frozen_source")),
                "outcome_label_status": "sealed_until_postfit_join",
            }
            contexts.append(context)
            realized[key] = terminal
            outcomes[key] = {
                "real_task_cost": float(np.linalg.norm(terminal[:2] - goal[:2])),
                "realized_terminal_state": terminal.tolist(),
            }
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "unseen_action_bank_context.json"
    outcome_path = output / "unseen_action_bank_outcomes.json"
    realized_path = output / "unseen_action_bank_realized_states.npz"
    context_path.write_text(
        json.dumps({
            "metadata": {
                "schema": SCHEMA,
                "candidate_family": UNSEEN_FAMILY,
                "outcomes_sealed_from_feature_construction": True,
                "candidate_id_offset": int(candidate_id_offset),
                "shared_world_action_bank": True,
            },
            "records": contexts,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outcome_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(realized_path, **realized)
    return context_path, outcome_path, realized_path


def _append_outcome_labels(records: list[dict[str, Any]], outcome_path: Path) -> list[dict[str, Any]]:
    """Perform the sole post-fit join of unseen real task outcomes."""

    outcomes = json.loads(outcome_path.read_text(encoding="utf-8"))
    labeled: list[dict[str, Any]] = []
    for row in records:
        key = f"{row['instance_id']}/{int(row['candidate_id'])}"
        value = outcomes.get(key)
        if not isinstance(value, dict) or "real_task_cost" not in value:
            raise ValueError(f"missing sealed outcome for {key}")
        labeled.append({**row, "real_task_cost": float(value["real_task_cost"])})
    return labeled


def _selection(records: list[dict[str, Any]], predicted_cost: np.ndarray) -> dict[str, Any]:
    if len(records) != len(predicted_cost):
        raise ValueError("records and prediction costs are misaligned")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[str(row["instance_id"])].append(index)
    roots: dict[str, dict[str, Any]] = {}
    for root, indices_list in sorted(grouped.items()):
        indices = np.asarray(indices_list, dtype=np.int64)
        # Candidate IDs are intentionally unique across train/test.  This tie
        # rule is deterministic and applies to every method including zeros.
        ordered = indices[np.argsort([int(records[index]["candidate_id"]) for index in indices], kind="stable")]
        selected = int(ordered[int(np.argmin(np.asarray(predicted_cost)[ordered]))])
        actual = np.asarray([float(records[index]["real_task_cost"]) for index in indices])
        oracle = int(indices[int(np.argmin(actual))])
        roots[root] = {
            "selected_candidate": int(records[selected]["candidate_id"]),
            "oracle_candidate": int(records[oracle]["candidate_id"]),
            "regret": float(records[selected]["real_task_cost"] - records[oracle]["real_task_cost"]),
            "ranking_correct": bool(selected == oracle),
        }
    regrets = np.asarray([value["regret"] for value in roots.values()], dtype=np.float64)
    return {
        "root_count": len(roots),
        "mean_regret": float(regrets.mean()),
        "zero_regret_fraction": float(np.mean([value["ranking_correct"] for value in roots.values()])),
        "roots": roots,
    }


def _paired_gain(method: dict[str, Any], control: dict[str, Any], *, seed: int, samples: int) -> dict[str, float]:
    roots = sorted(set(method["roots"]) & set(control["roots"]))
    if roots != sorted(method["roots"]) or roots != sorted(control["roots"]):
        raise ValueError("paired controls must cover exactly the same outcome roots")
    gain = np.asarray([
        float(control["roots"][root]["regret"] - method["roots"][root]["regret"])
        for root in roots
    ], dtype=np.float64)
    return _bootstrap(gain, roots, seed=seed, samples=samples)


def _selection_disagreement(full: dict[str, Any], bank: dict[str, Any]) -> dict[str, Any]:
    """Diagnose whether state conditioning helps after actions move OOD."""

    roots = sorted(set(full["roots"]) & set(bank["roots"]))
    if roots != sorted(full["roots"]) or roots != sorted(bank["roots"]):
        raise ValueError("full and action-only selections need identical roots")
    full_better = bank_better = ties = 0
    agreement = 0
    for root in roots:
        full_row, bank_row = full["roots"][root], bank["roots"][root]
        agreement += int(full_row["selected_candidate"] == bank_row["selected_candidate"])
        difference = float(full_row["regret"] - bank_row["regret"])
        if difference < -1.0e-12:
            full_better += 1
        elif difference > 1.0e-12:
            bank_better += 1
        else:
            ties += 1
    return {
        "root_count": len(roots),
        "selected_candidate_agreement_fraction": float(agreement / len(roots)),
        "full_adapter_lower_regret_root_count": full_better,
        "bank_only_lower_regret_root_count": bank_better,
        "tied_regret_root_count": ties,
    }


def _observed_effect_rmse(
    records: list[dict[str, Any]],
    effect: np.ndarray,
    outcome_path: Path,
) -> float:
    """Compute a post-fit physical-effect diagnostic from sealed outcomes."""

    outcomes = json.loads(outcome_path.read_text(encoding="utf-8"))
    observed = []
    for row in records:
        key = f"{row['instance_id']}/{int(row['candidate_id'])}"
        terminal = np.asarray(outcomes[key]["realized_terminal_state"], dtype=np.float64)
        source = np.asarray(row["source_state_raw"], dtype=np.float64)
        observed.append(terminal[:2] - source[:2])
    return float(np.sqrt(np.mean((np.asarray(effect, dtype=np.float64) - np.asarray(observed)) ** 2)))


def _ensemble_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    epochs: int,
    device: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values, metadata = [], []
    for seed in SEEDS:
        model, fit = fit_effect_model(train_x, train_y, seed=seed, epochs=epochs, device_name=device)
        values.append(predict_effect_model(model, fit, eval_x))
        metadata.append({key: value for key, value in fit.items() if key not in {"x_mean", "x_scale", "y_mean", "y_scale"}})
    return np.mean(np.stack(values), axis=0), metadata


def run(
    source_dataset: Path,
    source_arrays: Path,
    output: Path,
    *,
    epochs: int = 300,
    bootstrap_samples: int = 5000,
    device: str = "cuda",
    candidate_count: int = 16,
) -> dict[str, Any]:
    """Execute the predeclared unseen-action evaluation once, fail closed."""

    if device != "cuda":
        raise ValueError("formal unseen-action-bank transfer requires --device cuda")
    cuda = require_cuda_allocation(device)
    metadata, source_records = _load_records(source_dataset)
    if metadata.get("shared_world_action_bank") is not True:
        raise ValueError("source training bank must declare shared_world_action_bank=true")
    if metadata.get("candidate_bank_coordinate_frame") != "world_fixed":
        raise ValueError("source training bank must be world-fixed")
    source_arrays_payload = np.load(source_arrays, allow_pickle=False)
    train_records = [row for row in source_records if str(row.get("split")) in {"train", "calibration"}]
    if not train_records:
        raise ValueError("source bank has no train/calibration records")

    context_path, outcome_path, realized_path = materialize_unseen_bank(
        source_records, output, count=candidate_count,
    )
    _unseen_metadata, unseen_contexts = _load_records(context_path)
    if any("real_task_cost" in row for row in unseen_contexts):
        raise RuntimeError("unseen outcomes leaked into feature contexts")
    contract = audit_unseen_action_bank_contract(train_records, unseen_contexts)
    (output / "unseen_action_bank_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Train on historical action sequences and train roots only.  Calibration
    # roots stay completely outside fitting, and all unseen labels stay sealed.
    train_mask = np.asarray([str(row.get("split")) == "train" for row in source_records])
    full_x, full_y, _roots, _goals = _features(source_records, source_arrays_payload, "frozen_latent_action")
    action_width = int(full_x.shape[1] - len(source_records[0]["source_latent_model"]))
    if action_width <= 0:
        raise ValueError("source latent/action feature split is empty")
    unseen_full_x = np.asarray([
        np.concatenate((np.asarray(row["source_latent_model"], dtype=np.float32), np.asarray(row["action_sequence"], dtype=np.float32).reshape(-1)))
        for row in unseen_contexts
    ], dtype=np.float32)
    if unseen_full_x.shape[1] != full_x.shape[1]:
        raise ValueError("unseen context has incompatible frozen-latent/action width")
    full_effect, full_training = _ensemble_predict(
        full_x[train_mask], full_y[train_mask], unseen_full_x,
        epochs=epochs, device=device,
    )
    # The bank-only comparator learns an action response but gets no source
    # latent.  It cannot use candidate IDs and must extrapolate to new actions.
    bank_effect, bank_training = _ensemble_predict(
        full_x[train_mask, -action_width:], full_y[train_mask], unseen_full_x[:, -action_width:],
        epochs=epochs, device=device,
    )

    # No access to real_task_cost above this line.  This is the unique outcome
    # join, after both effect ensembles and all candidate scores are frozen.
    unseen_records = _append_outcome_labels(unseen_contexts, outcome_path)
    outcome_mask = np.asarray([str(row["split"]) == "outcome" for row in unseen_records])
    calibration_mask = np.asarray([str(row["split"]) == "calibration" for row in unseen_records])
    if not outcome_mask.any() or not calibration_mask.any():
        raise ValueError("unseen bank lacks calibration or outcome roots")
    source_xy = np.asarray([row["source_state_raw"][:2] for row in unseen_records], dtype=np.float64)
    goals = np.asarray([row["goal_raw"][:2] for row in unseen_records], dtype=np.float64)
    full_cost = np.linalg.norm(source_xy + full_effect - goals, axis=1)
    bank_cost = np.linalg.norm(source_xy + bank_effect - goals, axis=1)
    no_dynamics_cost = np.zeros(len(unseen_records), dtype=np.float64)
    outcome_records = [row for row, keep in zip(unseen_records, outcome_mask) if keep]
    full_selection = _selection(outcome_records, full_cost[outcome_mask])
    bank_selection = _selection(outcome_records, bank_cost[outcome_mask])
    no_dynamics_selection = _selection(outcome_records, no_dynamics_cost[outcome_mask])
    shuffled_selection = _selection(
        outcome_records,
        _cyclic_within_root(full_cost[outcome_mask], outcome_records),
    )
    controls = {
        "bank_only": _paired_gain(full_selection, bank_selection, seed=2026102101, samples=bootstrap_samples),
        "no_dynamics": _paired_gain(full_selection, no_dynamics_selection, seed=2026102102, samples=bootstrap_samples),
        "shuffled_successor": _paired_gain(full_selection, shuffled_selection, seed=2026102103, samples=bootstrap_samples),
    }
    calibration_selection = _selection(
        [row for row, keep in zip(unseen_records, calibration_mask) if keep],
        full_cost[calibration_mask],
    )
    diagnostics = {
        "full_vs_bank_only": _selection_disagreement(full_selection, bank_selection),
        "outcome_effect_rmse": {
            "full_adapter": _observed_effect_rmse(outcome_records, full_effect[outcome_mask], outcome_path),
            "bank_only": _observed_effect_rmse(outcome_records, bank_effect[outcome_mask], outcome_path),
        },
    }
    report = {
        "schema": SCHEMA,
        "status": (
            "passed_unseen_action_bank_kill_test"
            if all(float(value["lower"]) > 0.0 for value in controls.values())
            else "failed_unseen_action_bank_kill_test"
        ),
        "source": {
            "dataset": str(source_dataset.resolve()),
            "dataset_sha256": _sha256_file(source_dataset),
            "arrays": str(source_arrays.resolve()),
            "arrays_sha256": _sha256_file(source_arrays),
            "source_candidate_family": metadata.get("candidate_family"),
        },
        "unseen_bank": {
            "context": str(context_path.resolve()),
            "sealed_outcomes": str(outcome_path.resolve()),
            "realized_states": str(realized_path.resolve()),
            "candidate_family": UNSEEN_FAMILY,
            "candidate_count": int(candidate_count),
            "outcomes_opened_after_fit": True,
        },
        "contract": contract,
        "protocol": {
            "feature_mode": "frozen_latent_action",
            "full_adapter_input": "frozen source latent plus complete unseen action sequence",
            "bank_only_input": "complete unseen action sequence only",
            "no_dynamics_input": "zero predicted effect with deterministic candidate-ID tie break",
            "shuffled_successor": "cyclic within-root permutation of frozen full-adapter candidate scores",
            "training_roots": contract["train_root_counts"]["train"],
            "calibration_roots": contract["unseen_root_counts"]["calibration"],
            "outcome_roots": contract["unseen_root_counts"]["outcome"],
            "held_out_action_sequences": True,
            "outcome_used_for_fit_or_model_selection": False,
            "epochs": int(epochs),
            "seeds": list(SEEDS),
            "cuda_preflight": cuda,
        },
        "selection": {
            "full_adapter": full_selection,
            "bank_only": bank_selection,
            "no_dynamics": no_dynamics_selection,
            "shuffled_successor": shuffled_selection,
            "calibration_full_adapter_diagnostic": calibration_selection,
        },
        "paired_regret_gain": controls,
        "diagnostics": diagnostics,
        "training": {"full_adapter": full_training, "bank_only": bank_training},
        "primary_estimand": "paired held-root regret gain of full adapter over each preregistered control",
        "claim_boundary": (
            "An offline, root-disjoint PointMaze action-sequence generalization test for an "
            "intervention-supervised source-latent/action effect adapter. It is neither frozen "
            "world-model dynamics evidence nor closed-loop visual planning evidence."
        ),
    }
    (output / "unseen_action_bank_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, type=Path)
    parser.add_argument("--source-arrays", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    report = run(**vars(args))
    summary = {
        "status": report["status"],
        "outcome_roots": report["protocol"]["outcome_roots"],
        "paired_regret_gain": report["paired_regret_gain"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed_unseen_action_bank_kill_test" else 2


if __name__ == "__main__":
    raise SystemExit(main())
