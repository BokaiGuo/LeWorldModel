"""Development-frozen moderator selector and confirmatory PointMaze v4 test.

Subcommands deliberately separate (1) development-only selector fitting, (2)
fresh v4 panel construction, and (3) calibration/outcome evaluation.  The
selector sees no privileged simulator state and no v4 label.  V4 calibration
chooses only among a predeclared selector-score threshold grid before the
separately stored outcome file is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.decision_benchmark.audit_unseen_action_bank_contract import sequence_hash
from experiments.decision_benchmark.build_pointmaze_temporal_ood_bank import (
    FRAME_STRIDE, HISTORY_WINDOWS, causal_history, temporal_template_fingerprint,
)
from experiments.decision_benchmark.build_pointmaze_temporal_ood_v3_panel import (
    _context_identities, _encode_root_history, _renderer_calibration_roots, _sha256,
    _split_roots, v3_action_sequences,
)
from experiments.decision_benchmark.run_pointmaze_temporal_ood_test import (
    FEATURES, _costs, _fit_predict, _training_features,
)
from experiments.decision_benchmark.run_unseen_action_bank_pointmaze import (
    _bootstrap, _load_records, _selection, _simulate_sequence,
)
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_pointmaze_planning import (
    DatasetMatchedRenderer, _load_jepa_model, fit_visual_calibration, load_episode,
)


SELECTOR_SCHEMA = "pointmaze_temporal_ood_v4_development_selector.v1"
PANEL_SCHEMA = "pointmaze_temporal_ood_v4_renderer_control.v1"
V4_ACTION_FAMILY = "world_fixed_random_five_pulse_temporal_ood_v4"
V4_TRAIN_FAMILY = "world_fixed_random_train_five_pulse_temporal_ood_v4"
ROOT_SEED, ACTION_SEED, TRAIN_ACTION_SEED = 202610230, 202610231, 202610232
SPLIT_COUNTS = {"train": 64, "calibration": 48, "outcome": 32}
SOURCE_FRAME = 20
GOAL = np.asarray([1.2, 1.2], dtype=np.float64)
THRESHOLD_GRID = (-0.030, -0.015, 0.0, 0.015, 0.030)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _root_rows(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        result[str(row["instance_id"])].append(row)
    return dict(result)


def moderator_features(records: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], list[str]]:
    """Derive pre-outcome, non-privileged selector moderators per root.

    The value deliberately excludes ``source_state_raw``.  It uses exactly
    what a temporal representation/action-ranking deployment would have:
    frozen current/history latents, executed causal controls, and the offered
    candidate action bank.
    """

    features, roots, names = [], [], [
        "history_latent_innovation_mean", "history_latent_innovation_max", "current_latent_norm",
        "history_action_mean_amplitude", "history_action_amplitude_std", "history_action_direction_change",
        "bank_mean_amplitude", "bank_amplitude_std", "bank_temporal_variation", "bank_direction_diversity",
    ]
    for root, rows in sorted(_root_rows(records).items()):
        first = rows[0]
        latent = np.asarray(first["source_latent_model"], dtype=np.float64).reshape(-1)
        history = np.asarray(first["source_history_model"], dtype=np.float64).reshape(HISTORY_WINDOWS, -1)
        history_actions = np.asarray(first["history_actions_raw"], dtype=np.float64)
        candidates = np.asarray([row["action_sequence"] for row in rows], dtype=np.float64)
        h_diff = np.diff(history, axis=0)
        ha_norm = np.linalg.norm(history_actions, axis=1)
        unit_history = history_actions / np.maximum(ha_norm[:, None], 1.0e-8)
        bank_norm = np.linalg.norm(candidates, axis=2)
        unit = candidates / np.maximum(bank_norm[..., None], 1.0e-8)
        flat_direction = unit.reshape(-1, 2)
        # Mean pairwise cosine distance is an action-bank diversity statistic,
        # never a realized-effect or task-cost margin.
        gram = flat_direction @ flat_direction.T
        diversity = float(np.mean(1.0 - gram[np.triu_indices(len(flat_direction), k=1)])) if len(flat_direction) > 1 else 0.0
        features.append([
            float(np.mean(np.linalg.norm(h_diff, axis=1))), float(np.max(np.linalg.norm(h_diff, axis=1))), float(np.linalg.norm(latent)),
            float(ha_norm.mean()), float(ha_norm.std()), float(np.mean(1.0 - np.sum(unit_history[1:] * unit_history[:-1], axis=1))),
            float(bank_norm.mean()), float(bank_norm.std()), float(np.mean(np.abs(np.diff(candidates, axis=1)))), diversity,
        ])
        roots.append(root)
    return np.asarray(features, dtype=np.float64), roots, names


def _fit_ridge(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1.0e-6)
    normalized = (x - mean) / scale
    design = np.c_[normalized, np.ones(len(normalized))]
    penalty = np.eye(design.shape[1]) * 1.0e-2
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}


def _score(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return np.c_[(x - model["mean"]) / model["scale"], np.ones(len(x))] @ model["weights"]


def _development_rows(context: Path, report: Path, panel: str) -> list[dict[str, Any]]:
    _metadata, records = _load_records(context)
    payload = json.loads(report.read_text())
    by_root = _root_rows(records)
    x, roots, names = moderator_features(records)
    if not names:
        raise ValueError("empty development moderator design")
    rows = []
    for feature, root in zip(x, roots):
        root_records = by_root[root]
        split = str(root_records[0]["split"])
        if split not in {"calibration", "outcome"}:
            continue
        action = payload["selection"]["action_only"][split]["roots"].get(root)
        history = payload["selection"]["history_latent_action"][split]["roots"].get(root)
        if action is None or history is None:
            raise ValueError(f"development report misses {panel}/{root}")
        rows.append({"panel": panel, "root": root, "split": split, "x": feature.tolist(),
            "action_regret": float(action["regret"]), "history_regret": float(history["regret"]),
            "history_benefit": float(action["regret"] - history["regret"])})
    return rows


def fit_selector(dev_contexts: list[Path], dev_reports: list[Path], output: Path) -> dict[str, Any]:
    if len(dev_contexts) != 3 or len(dev_reports) != 3:
        raise ValueError("v4 selector requires exactly v1/v2/v3 development panels")
    names = ["v1", "v2", "v3"]
    rows = [item for context, report, panel in zip(dev_contexts, dev_reports, names) for item in _development_rows(context, report, panel)]
    x = np.asarray([row["x"] for row in rows], dtype=np.float64)
    y = np.asarray([row["history_benefit"] for row in rows], dtype=np.float64)
    panels = np.asarray([row["panel"] for row in rows])
    if len(rows) < 20 or set(panels) != set(names):
        raise ValueError("development selector needs rows from all v1/v2/v3 panels")
    cv = []
    for held_panel in names:
        train = panels != held_panel
        model = _fit_ridge(x[train], y[train])
        score = _score(model, x[~train])
        held_rows = [row for row, keep in zip(rows, ~train) if keep]
        for threshold in THRESHOLD_GRID:
            regret = np.asarray([row["history_regret"] if value > threshold else row["action_regret"] for row, value in zip(held_rows, score)])
            cv.append({"held_panel": held_panel, "threshold": threshold, "mean_regret": float(regret.mean()), "history_fraction": float(np.mean(score > threshold))})
    by_threshold = {threshold: float(np.mean([row["mean_regret"] for row in cv if row["threshold"] == threshold])) for threshold in THRESHOLD_GRID}
    # The production threshold remains a calibration choice in v4.  This
    # development argmin only defines a transparent default/tie order.
    development_default = min(THRESHOLD_GRID, key=lambda value: (by_threshold[value], -value))
    fitted = _fit_ridge(x, y)
    output.parent.mkdir(parents=True, exist_ok=True)
    package = {"schema": SELECTOR_SCHEMA, "feature_names": moderator_features([{"instance_id": "x", "source_latent_model": [0.0], "source_history_model": [0.0] * 4, "history_actions_raw": [[0.0, 0.0]] * 15, "action_sequence": [[0.0, 0.0]] * 20}])[2],
        "moderator_contract": "causal frozen latent/history/actions and offered action bank only; no source_state_raw, realized transition, or task outcome",
        "development_panels": names, "development_root_count": len(rows), "panel_root_aware_cv": "leave-one-panel-out; roots remain the analysis unit", "threshold_grid": list(THRESHOLD_GRID),
        "development_default_threshold": development_default, "development_cv": cv, "development_threshold_mean_regret": by_threshold,
        "model": {key: np.asarray(value).tolist() for key, value in fitted.items()}, "training_rows": rows,
        "v4_rule": "freeze coefficient model before v4 creation; select a threshold only from v4 calibration roots"}
    output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n")
    return package


def build_v4(dataset_root: Path, v1_context: Path, v2_context: Path, v3_context: Path, selector: Path, output: Path, *, encoder_checkpoint: Path, pca_checkpoint: Path, dinov2_source: Path, device: str = "cuda") -> dict[str, Any]:
    if device != "cuda":
        raise ValueError("formal v4 panel requires CUDA")
    cuda = require_cuda_allocation(device)
    selector_payload = json.loads(selector.read_text())
    if selector_payload.get("schema") != SELECTOR_SCHEMA:
        raise ValueError("v4 panel requires a frozen development selector")
    old_outcomes, old_hashes, old_templates = _context_identities(v1_context, v2_context, v3_context)
    split = _split_roots(seed=ROOT_SEED, forbidden=old_outcomes)
    if set(split) & old_outcomes:
        raise RuntimeError("v4 roots overlap v1-v3 outcome roots")
    calibration_roots = _renderer_calibration_roots(dataset_root, old_outcomes)
    calibration = fit_visual_calibration(dataset_root, calibration_roots, max_frames=1600)
    renderer_images, _ = load_episode(dataset_root, calibration_roots[0])
    renderer = DatasetMatchedRenderer.from_episode(renderer_images, calibration)
    root = Path(__file__).resolve().parents[3]
    model, transform, _stats, _mean, _std = _load_jepa_model(root, dataset_root, encoder_checkpoint, dinov2_source, torch.device(device))
    pca = torch.load(pca_checkpoint, map_location="cpu", weights_only=False)
    pca_mean, pca_basis = np.asarray(pca["pca_mean"], dtype=np.float64), np.asarray(pca["pca_basis"], dtype=np.float64)
    states = np.asarray(torch.load(dataset_root / "states.pth", map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    actions = np.asarray(torch.load(dataset_root / "actions.pth", map_location="cpu", weights_only=True).numpy(), dtype=np.float64)
    held, train = v3_action_sequences(seed=ACTION_SEED, train=False), v3_action_sequences(seed=TRAIN_ACTION_SEED, train=True)
    held_hashes, train_hashes = [sequence_hash(value) for value in held], [sequence_hash(value) for value in train]
    held_templates, train_templates = [temporal_template_fingerprint(value) for value in held], [temporal_template_fingerprint(value) for value in train]
    if set(held_hashes) & old_hashes or set(held_templates) & old_templates:
        raise RuntimeError("v4 held action bank overlaps a v1-v3 hash/template")
    if set(held_hashes) & set(train_hashes) or set(held_templates) & set(train_templates):
        raise RuntimeError("v4 held and fitting action banks overlap")
    histories = {episode: _encode_root_history(model, transform, pca_mean, pca_basis, renderer, calibration, states[episode]) for episode in sorted(split)}
    frames = causal_history(SOURCE_FRAME, actions[0])[0]
    source_records, source_arrays, contexts = [], {}, []
    split_outcomes: dict[str, dict[str, Any]] = {"calibration": {}, "outcome": {}}
    for episode in sorted(split):
        state = np.asarray(states[episode, SOURCE_FRAME], dtype=np.float64)
        history_actions = causal_history(SOURCE_FRAME, actions[episode])[1]
        history = histories[episode]
        common = {"instance_id": f"pointmaze_v4_episode_{episode}", "split": split[episode], "environment": "PointMaze", "start_episode_id": str(episode), "source_frame": SOURCE_FRAME,
            "history_frame_indices": frames, "history_action_interval": [frames[0], SOURCE_FRAME], "history_actions_raw": history_actions.tolist(),
            "source_latent_model": history[-len(pca_basis):].tolist(), "source_history_model": history.tolist(), "source_state_raw": state.tolist(), "goal_raw": GOAL.tolist()}
        if split[episode] == "train":
            for index, candidate in enumerate(train):
                key = f"terminal/v4/{episode}/{index}"
                source_arrays[key] = _simulate_sequence(state, candidate)
                source_records.append({**common, "candidate_id": index, "candidate_generator": V4_TRAIN_FAMILY, "candidate_bank_hash": _json_hash(train_hashes), "candidate_action_hash": train_hashes[index], "candidate_template_fingerprint": train_templates[index], "action_sequence": candidate.tolist(), "realized_terminal_state_ref": f"arrays.npz::{key}"})
        for index, candidate in enumerate(held):
            key = f"{common['instance_id']}/{index}"
            contexts.append({**common, "candidate_id": index, "candidate_generator": V4_ACTION_FAMILY, "candidate_bank_hash": _json_hash(held_hashes), "candidate_action_hash": held_hashes[index], "candidate_template_fingerprint": held_templates[index], "action_sequence": candidate.tolist(), "total_action_steps": 20, "outcome_label_status": "split_sealed_until_postfit"})
            if split[episode] in split_outcomes:
                terminal = _simulate_sequence(state, candidate)
                split_outcomes[split[episode]][key] = {"real_task_cost": float(np.linalg.norm(terminal[:2] - GOAL)), "realized_terminal_state": terminal.tolist()}
    output.mkdir(parents=True, exist_ok=True)
    source_path, arrays_path, context_path = output / "temporal_ood_v4_source_dataset.json", output / "temporal_ood_v4_source_arrays.npz", output / "temporal_ood_v4_context.json"
    cal_path, outcome_path = output / "temporal_ood_v4_calibration_outcomes.json", output / "temporal_ood_v4_outcomes.json"
    source_path.write_text(json.dumps({"metadata": {"history_windows": 4, "schema": "pointmaze_temporal_ood_v4_source.v1", "renderer_backed_frozen_observation_control": True}, "records": source_records}, indent=2, sort_keys=True) + "\n")
    np.savez(arrays_path, **source_arrays)
    context_path.write_text(json.dumps({"metadata": {"schema": PANEL_SCHEMA, "candidate_family": V4_ACTION_FAMILY, "root_split_seed": ROOT_SEED, "action_bank_seed": ACTION_SEED, "train_action_seed": TRAIN_ACTION_SEED, "history_windows": 4, "history_frame_stride": FRAME_STRIDE, "history_is_causal": True, "history_actions_are_strictly_pre_candidate": True, "outcomes_sealed_from_feature_construction": True, "renderer_backed_frozen_observation_control": True, "selector_sha256": _sha256(selector), "v1_v3_outcome_roots_excluded": sorted(old_outcomes), "v1_v3_action_hash_overlap": [], "v1_v3_template_overlap": []}, "records": contexts}, indent=2, sort_keys=True) + "\n")
    cal_path.write_text(json.dumps(split_outcomes["calibration"], indent=2, sort_keys=True) + "\n")
    outcome_path.write_text(json.dumps(split_outcomes["outcome"], indent=2, sort_keys=True) + "\n")
    report = {"schema": PANEL_SCHEMA, "cuda_preflight": cuda, "selector": str(selector.resolve()), "context": str(context_path.resolve()), "source_dataset": str(source_path.resolve()), "source_arrays": str(arrays_path.resolve()), "calibration_outcomes": str(cal_path.resolve()), "outcome_outcomes": str(outcome_path.resolve()), "root_counts": {name: sum(value == name for value in split.values()) for name in SPLIT_COUNTS}, "v1_v3_outcome_root_overlap": sorted(set(split) & old_outcomes), "v1_v3_action_hash_overlap": sorted(set(held_hashes) & old_hashes), "v1_v3_template_overlap": sorted(set(held_templates) & old_templates), "claim_boundary": "Renderer-backed offline control; not live RGB deployment evidence."}
    (output / "build_v4_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _labeled(records: list[dict[str, Any]], outcomes: Path) -> list[dict[str, Any]]:
    values = json.loads(outcomes.read_text())
    result = []
    for row in records:
        key = f"{row['instance_id']}/{int(row['candidate_id'])}"
        if key not in values:
            raise ValueError(f"sealed split outcome missing {key}")
        result.append({**row, "real_task_cost": float(values[key]["real_task_cost"])})
    return result


def _selection_split(records: list[dict[str, Any]], prediction: np.ndarray, split: str, outcomes: Path) -> dict[str, Any]:
    indices = np.asarray([str(row["split"]) == split for row in records])
    selected_records = [row for row, keep in zip(records, indices) if keep]
    return _selection(_labeled(selected_records, outcomes), _costs(selected_records, prediction[indices]))


def _paired(method: dict[str, Any], control: dict[str, Any], *, seed: int, samples: int) -> dict[str, float]:
    roots = sorted(method["roots"])
    if roots != sorted(control["roots"]):
        raise ValueError("paired selector comparisons require identical roots")
    values = np.asarray([control["roots"][root]["regret"] - method["roots"][root]["regret"] for root in roots])
    return _bootstrap(values, roots, seed=seed, samples=samples)


def run_v4(source_dataset: Path, source_arrays: Path, context: Path, calibration_outcomes: Path, outcome_outcomes: Path, selector: Path, output: Path, *, epochs: int = 300, bootstrap_samples: int = 5000, device: str = "cuda") -> dict[str, Any]:
    if device != "cuda":
        raise ValueError("formal v4 test requires CUDA")
    cuda = require_cuda_allocation(device)
    metadata, records = _load_records(context)
    frozen = json.loads(selector.read_text())
    if metadata.get("schema") != PANEL_SCHEMA or frozen.get("schema") != SELECTOR_SCHEMA:
        raise ValueError("v4 panel/selector schema mismatch")
    counts = {name: len({str(row["instance_id"]) for row in records if str(row["split"]) == name}) for name in SPLIT_COUNTS}
    if counts != SPLIT_COUNTS:
        raise ValueError(f"unexpected v4 root counts: {counts}")
    source_meta, source_records = _load_records(source_dataset)
    if int(source_meta.get("history_windows", 0)) != 4:
        raise ValueError("v4 source lacks four-frame history")
    arrays = np.load(source_arrays, allow_pickle=False)
    train_records = [row for row in source_records if str(row["split"]) == "train"]
    predictions, fits = {}, {}
    for family in FEATURES:
        x_train, y_train = _training_features(train_records, arrays, family)
        x_eval = np.asarray([__import__("experiments.decision_benchmark.run_pointmaze_temporal_ood_test", fromlist=["_feature"])._feature(row, family) for row in records], dtype=np.float32)
        predictions[family], fits[family] = _fit_predict(x_train, y_train, x_eval, epochs=epochs, device=device)
    # First and only calibration-outcome opening.  Threshold is frozen and
    # written before the held outcome file is read.
    cal = {family: _selection_split(records, prediction, "calibration", calibration_outcomes) for family, prediction in predictions.items()}
    x, roots, feature_names = moderator_features(records)
    model = {key: np.asarray(value, dtype=np.float64) for key, value in frozen["model"].items()}
    scores = dict(zip(roots, _score(model, x)))
    calibration_rows = sorted(cal["action_only"]["roots"])
    threshold_scores = []
    for threshold in frozen["threshold_grid"]:
        choice = {root: cal["history_latent_action"]["roots"][root] if scores[root] > threshold else cal["action_only"]["roots"][root] for root in calibration_rows}
        threshold_scores.append({"threshold": float(threshold), "mean_regret": float(np.mean([item["regret"] for item in choice.values()])), "history_fraction": float(np.mean([scores[root] > threshold for root in calibration_rows]))})
    chosen = min(threshold_scores, key=lambda item: (item["mean_regret"], -item["threshold"]))
    output.mkdir(parents=True, exist_ok=True)
    freeze = {"schema": "pointmaze_temporal_ood_v4_calibration_threshold.v1", "selected_threshold": chosen["threshold"], "threshold_grid": frozen["threshold_grid"], "selection_source": "v4 calibration roots only", "calibration_scores": threshold_scores, "outcome_file_not_opened_when_written": True}
    (output / "temporal_ood_v4_frozen_calibration_threshold.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    # Only after threshold persistence may the untouched outcome consequence
    # file be opened.  All four predictors and moderator scores are already
    # frozen at this point.
    outcome = {family: _selection_split(records, prediction, "outcome", outcome_outcomes) for family, prediction in predictions.items()}
    outcome_roots = sorted(outcome["action_only"]["roots"])
    selector_roots = {root: outcome["history_latent_action"]["roots"][root] if scores[root] > chosen["threshold"] else outcome["action_only"]["roots"][root] for root in outcome_roots}
    selected = {"root_count": len(selector_roots), "mean_regret": float(np.mean([item["regret"] for item in selector_roots.values()])), "zero_regret_fraction": float(np.mean([item["ranking_correct"] for item in selector_roots.values()])), "roots": selector_roots}
    versus_action, versus_history = _paired(selected, outcome["action_only"], seed=2026102330, samples=bootstrap_samples), _paired(selected, outcome["history_latent_action"], seed=2026102331, samples=bootstrap_samples)
    passes = float(versus_action["lower"]) > 0.0 and float(versus_history["lower"]) > 0.0
    status = "selector_permitted_offline_only" if passes else "action_only_fallback"
    report = {"schema": "pointmaze_temporal_ood_v4_repairability.v1", "cuda_preflight": cuda, "counts": counts, "four_arms": list(FEATURES), "selector_feature_names": feature_names, "training": fits,
        "calibration_only_threshold": freeze, "selection": {"calibration": cal, "outcome": outcome, "frozen_selector": selected},
        "primary_paired_gain": {"selector_over_action_only": versus_action, "selector_over_always_history": versus_history, "criterion": "both paired lower CIs > 0", "satisfied": passes},
        "policy_status": status, "claim_boundary": "Offline renderer-backed action-OOD repairability control, not a visual live-deployment result."}
    (output / "temporal_ood_v4_repairability_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-selector")
    fit.add_argument("--contexts", type=Path, nargs=3, required=True)
    fit.add_argument("--reports", type=Path, nargs=3, required=True)
    fit.add_argument("--output", type=Path, required=True)
    build = sub.add_parser("build-panel")
    build.add_argument("--dataset-root", type=Path, required=True); build.add_argument("--v1-context", type=Path, required=True); build.add_argument("--v2-context", type=Path, required=True); build.add_argument("--v3-context", type=Path, required=True); build.add_argument("--selector", type=Path, required=True); build.add_argument("--output", type=Path, required=True); build.add_argument("--encoder-checkpoint", type=Path, required=True); build.add_argument("--pca-checkpoint", type=Path, required=True); build.add_argument("--dinov2-source", type=Path, required=True); build.add_argument("--device", choices=("cuda",), default="cuda")
    run = sub.add_parser("run")
    run.add_argument("--source-dataset", type=Path, required=True); run.add_argument("--source-arrays", type=Path, required=True); run.add_argument("--context", type=Path, required=True); run.add_argument("--calibration-outcomes", type=Path, required=True); run.add_argument("--outcome-outcomes", type=Path, required=True); run.add_argument("--selector", type=Path, required=True); run.add_argument("--output", type=Path, required=True); run.add_argument("--epochs", type=int, default=300); run.add_argument("--bootstrap-samples", type=int, default=5000); run.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    values = vars(args); command = values.pop("command")
    if command == "fit-selector": result = fit_selector(values.pop("contexts"), values.pop("reports"), **values)
    elif command == "build-panel": result = build_v4(**values)
    else: result = run_v4(**values)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
