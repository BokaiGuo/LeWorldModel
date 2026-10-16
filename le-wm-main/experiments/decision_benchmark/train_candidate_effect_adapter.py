"""Train and evaluate an intervention-supervised candidate-effect adapter.

This is an explicit causal-data intervention, not a frozen-geometry result.
The adapter is trained only on train-root simulator outcomes from one shared
equal-duration candidate bank.  It receives the frozen RGB-derived source
latent and the complete action sequence, then predicts terminal XY effect.
Outcome roots are never used for fitting, normalization, or model selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from experiments.decision_benchmark.task_contrast import (
    calibrate_contrast_bound,
    calibrate_root_score_bound,
    certify_candidate_bank,
    evaluate_contrast_bound,
)


SEEDS = (8411, 8412, 8413, 8414, 8415)


class EffectMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _array_key(reference: str) -> str:
    parts = str(reference).split("::", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid array reference: {reference}")
    return parts[1]


def _features(
    records: list[dict[str, Any]],
    arrays: Any,
    feature_mode: str = "frozen_latent_action",
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if feature_mode not in {
        "frozen_latent_action",
        "history_latent_action",
        "privileged_state_action",
        "latent_plus_state_action",
    }:
        raise ValueError(f"unsupported feature_mode: {feature_mode}")
    x_values, y_values, roots, goals = [], [], [], []
    for row in records:
        if row.get("source_latent_model") is None:
            raise ValueError("candidate record lacks source_latent_model; regenerate the bank")
        source = np.asarray(row["source_latent_model"], dtype=np.float64)
        action = np.asarray(row["action_sequence"], dtype=np.float64).reshape(-1)
        realized = np.asarray(arrays[_array_key(row["realized_terminal_state_ref"])], dtype=np.float64)
        source_state = np.asarray(row["source_state_raw"], dtype=np.float64)
        if feature_mode == "frozen_latent_action":
            context = source
        elif feature_mode == "history_latent_action":
            if row.get("source_history_model") is None:
                raise ValueError("candidate record lacks source_history_model; regenerate the bank")
            context = np.asarray(row["source_history_model"], dtype=np.float64)
        elif feature_mode == "privileged_state_action":
            context = source_state
        else:
            context = np.concatenate([source, source_state])
        x_values.append(np.concatenate([context, action]))
        y_values.append(realized[:2] - source_state[:2])
        roots.append(str(row["instance_id"]))
        goals.append(np.asarray(row["goal_raw"], dtype=np.float64))
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32), roots, np.asarray(goals, dtype=np.float64)


def _resolve_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device: {requested}")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    seed: int,
    epochs: int,
    device_name: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    _seed(seed)
    device = _resolve_device(device_name)
    x_mean = train_x.mean(axis=0)
    x_scale = np.maximum(train_x.std(axis=0), 1e-3)
    y_mean = train_y.mean(axis=0)
    y_scale = np.maximum(train_y.std(axis=0), 1e-3)
    model = EffectMLP(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    x = torch.as_tensor((train_x - x_mean) / x_scale, dtype=torch.float32, device=device)
    y = torch.as_tensor((train_y - y_mean) / y_scale, dtype=torch.float32, device=device)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        pred = model(torch.as_tensor(
            (eval_x - x_mean) / x_scale, dtype=torch.float32, device=device,
        )).cpu().numpy()
    pred = pred * y_scale + y_mean
    return pred, {
        "x_mean": x_mean.tolist(), "x_scale": x_scale.tolist(),
        "y_mean": y_mean.tolist(), "y_scale": y_scale.tolist(),
        "epochs": int(epochs),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "device": str(device),
    }


def _bootstrap(values: np.ndarray, roots: list[str], seed: int, samples: int = 1000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    unique = np.asarray(sorted(set(roots)))
    if len(unique) < 2:
        mean = float(np.mean(values))
        return {"estimate": mean, "lower": mean, "upper": mean, "groups": int(len(unique))}
    rng = np.random.default_rng(seed)
    grouped = [values[np.asarray(roots) == root] for root in unique]
    draws = []
    for _ in range(samples):
        indices = rng.integers(0, len(unique), len(unique))
        draws.append(float(np.mean(np.concatenate([grouped[index] for index in indices]))))
    return {"estimate": float(values.mean()), "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975)), "groups": int(len(unique))}


def _root_selection(
    records: list[dict[str, Any]],
    predicted_cost: np.ndarray,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        grouped.setdefault(str(row["instance_id"]), []).append(index)
    roots = {}
    for root, indices_list in sorted(grouped.items()):
        indices = np.asarray(indices_list, dtype=np.int64)
        selected = int(indices[int(np.argmin(predicted_cost[indices]))])
        actual = np.asarray([float(records[index]["real_task_cost"]) for index in indices])
        oracle_local = int(np.argmin(actual))
        oracle = int(indices[oracle_local])
        roots[root] = {
            "selected_candidate": int(records[selected]["candidate_id"]),
            "oracle_candidate": int(records[oracle]["candidate_id"]),
            "regret": float(records[selected]["real_task_cost"] - records[oracle]["real_task_cost"]),
            "ranking_correct": bool(selected == oracle),
        }
    return {
        "root_count": len(roots),
        "mean_regret": float(np.mean([row["regret"] for row in roots.values()])),
        "zero_regret_fraction": float(np.mean([row["ranking_correct"] for row in roots.values()])),
        "roots": roots,
    }


def _cyclic_within_root(values: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    shuffled = np.asarray(values, dtype=np.float64).copy()
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        grouped.setdefault(str(row["instance_id"]), []).append(index)
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: int(records[index]["candidate_id"]))
        shuffled[ordered] = np.roll(values[ordered], 1)
    return shuffled


def _paired_regret_gain(
    method: dict[str, Any],
    control: dict[str, Any],
    *,
    seed: int,
    samples: int,
) -> dict[str, float]:
    roots = sorted(set(method["roots"]) & set(control["roots"]))
    values = np.asarray([
        float(control["roots"][root]["regret"] - method["roots"][root]["regret"])
        for root in roots
    ])
    return _bootstrap(values, roots, seed, samples)


def _matched_order_pair_audit(
    all_records: list[dict[str, Any]],
    predicted_cost: np.ndarray,
    *,
    train_mask: np.ndarray,
    outcome_mask: np.ndarray,
    samples: int,
) -> dict[str, Any]:
    """Audit same-action-multiset pairs whose temporal order is reversed."""

    pairs = ((0, 1), (2, 3), (4, 5), (6, 7))
    train_mean = {
        candidate: float(np.mean([
            float(row["real_task_cost"])
            for row, keep in zip(all_records, train_mask)
            if keep and int(row["candidate_id"]) == candidate
        ]))
        for pair in pairs for candidate in pair
    }
    outcome_rows = [row for row, keep in zip(all_records, outcome_mask) if keep]
    outcome_scores = predicted_cost[outcome_mask]
    by_root: dict[str, dict[int, tuple[dict[str, Any], float]]] = {}
    for row, score in zip(outcome_rows, outcome_scores):
        by_root.setdefault(str(row["instance_id"]), {})[int(row["candidate_id"])] = (row, float(score))
    root_method, root_bank, root_swap = {}, {}, {}
    left_better = {f"{left}-{right}": [] for left, right in pairs}
    for root, candidates in sorted(by_root.items()):
        method_regrets, bank_regrets, swap_regrets = [], [], []
        for left, right in pairs:
            left_row, left_score = candidates[left]
            right_row, right_score = candidates[right]
            left_cost = float(left_row["real_task_cost"])
            right_cost = float(right_row["real_task_cost"])
            oracle = min(left_cost, right_cost)
            method_choice = left_cost if left_score <= right_score else right_cost
            bank_choice = left_cost if train_mean[left] <= train_mean[right] else right_cost
            swap_choice = right_cost if left_score <= right_score else left_cost
            method_regrets.append(method_choice - oracle)
            bank_regrets.append(bank_choice - oracle)
            swap_regrets.append(swap_choice - oracle)
            left_better[f"{left}-{right}"].append(bool(left_cost < right_cost))
        root_method[root] = float(np.mean(method_regrets))
        root_bank[root] = float(np.mean(bank_regrets))
        root_swap[root] = float(np.mean(swap_regrets))
    roots = sorted(root_method)
    def report(values: dict[str, float]) -> dict[str, Any]:
        return {"mean_regret": float(np.mean(list(values.values()))), "root_regret": values}
    method_report = report(root_method)
    bank_report = report(root_bank)
    swap_report = report(root_swap)
    gain_bank = _bootstrap(
        np.asarray([root_bank[root] - root_method[root] for root in roots]),
        roots, 20261020, samples,
    )
    gain_swap = _bootstrap(
        np.asarray([root_swap[root] - root_method[root] for root in roots]),
        roots, 20261021, samples,
    )
    return {
        "pairs": [list(pair) for pair in pairs],
        "pair_definition": "same cumulative action and action multiset; temporal order reversed",
        "outcome_root_count": len(roots),
        "method": method_report,
        "bank_only_train_mean_order": bank_report,
        "swapped_prediction_control": swap_report,
        "paired_regret_gain_over_bank_only": gain_bank,
        "paired_regret_gain_over_swapped_prediction": gain_swap,
        "outcome_left_better_fraction": {
            pair: float(np.mean(values)) for pair, values in left_better.items()
        },
        "claim_boundary": "matched temporal-order mechanism audit inside the fixed bank; not full-bank or adaptive-CEM planning",
    }


def _train_only_root_support(
    records: list[dict[str, Any]],
    train_mask: np.ndarray,
    *,
    quantile: float = 0.9,
) -> dict[str, Any]:
    """Fit a root-level latent support gate without calibration/outcome data."""

    by_root: dict[str, np.ndarray] = {}
    split_by_root: dict[str, str] = {}
    for row in records:
        root = str(row["instance_id"])
        value = np.asarray(row["source_latent_model"], dtype=np.float64)
        if root in by_root and not np.allclose(by_root[root], value):
            raise ValueError(f"source latent varies within root {root}")
        by_root[root] = value
        split_by_root[root] = str(row["split"])
    train_roots = sorted({
        str(row["instance_id"]) for row, keep in zip(records, train_mask) if keep
    })
    train = np.asarray([by_root[root] for root in train_roots], dtype=np.float64)
    if len(train) < 3:
        raise ValueError("at least three train roots are required for support calibration")
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1.0e-6)
    standardized = (train - mean) / scale
    distances = np.sqrt(np.mean(
        (standardized[:, None, :] - standardized[None, :, :]) ** 2,
        axis=-1,
    ))
    np.fill_diagonal(distances, np.inf)
    leave_one_root_out = np.min(distances, axis=1)
    threshold = float(np.quantile(leave_one_root_out, quantile, method="higher"))
    roots = {}
    for root, value in by_root.items():
        query = (value - mean) / scale
        distance = float(np.min(np.sqrt(np.mean((standardized - query) ** 2, axis=1))))
        roots[root] = {
            "split": split_by_root[root],
            "nearest_train_root_distance": distance,
            "passed": bool(distance <= threshold),
        }
    return {
        "feature": "frozen source latent projection",
        "fit_split": "train roots only",
        "quantile": float(quantile),
        "threshold": threshold,
        "train_root_count": len(train_roots),
        "leave_one_train_root_out_distances": leave_one_root_out.tolist(),
        "roots": roots,
        "claim_boundary": "train-support gate; not a density guarantee or adaptive-candidate certificate",
    }


def run(
    dataset: Path,
    arrays: Path,
    output: Path,
    *,
    epochs: int = 300,
    bootstrap_samples: int = 1000,
    feature_mode: str = "frozen_latent_action",
    device: str = "auto",
) -> dict[str, Any]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    all_records = payload["records"]
    arrays_payload = np.load(arrays, allow_pickle=False)
    all_x, all_y, all_roots, all_goals = _features(all_records, arrays_payload, feature_mode)
    train_mask = np.asarray([row.get("split") == "train" for row in all_records])
    calibration_mask = np.asarray([row.get("split") == "calibration" for row in all_records])
    outcome_mask = np.asarray([row.get("split") == "outcome" for row in all_records])
    if not bool(train_mask.any()) or not bool(calibration_mask.any()) or not bool(outcome_mask.any()):
        raise ValueError("train, calibration, and outcome records are required")
    train_x, train_y = all_x[train_mask], all_y[train_mask]
    outcome_records = [row for row, keep in zip(all_records, outcome_mask) if keep]
    outcome_y = all_y[outcome_mask]
    outcome_roots = [root for root, keep in zip(all_roots, outcome_mask) if keep]
    outcome_goals = all_goals[outcome_mask]
    all_sources = np.asarray([row["source_state_raw"] for row in all_records], dtype=np.float64)[:, :2]
    outcome_sources = all_sources[outcome_mask]
    grouped_outcome: dict[str, list[int]] = {}
    for index, root in enumerate(outcome_roots):
        grouped_outcome.setdefault(root, []).append(index)
    reports = []
    all_predictions = []
    for seed in SEEDS:
        prediction_all, training = _fit_predict(train_x, train_y, all_x, seed, epochs, device)
        all_predictions.append(prediction_all)
        prediction = prediction_all[outcome_mask]
        train_prediction = prediction_all[train_mask]
        predicted_terminal = outcome_sources + prediction
        predicted_cost = np.linalg.norm(predicted_terminal - outcome_goals, axis=1)
        actual_cost = np.linalg.norm(outcome_sources + outcome_y - outcome_goals, axis=1)
        selected_errors, random_errors, oracle_errors = [], [], []
        selected_success, random_success = [], []
        for root, indices in sorted(grouped_outcome.items()):
            indices = np.asarray(indices, dtype=np.int64)
            selected = indices[int(np.argmin(predicted_cost[indices]))]
            random_index = indices[int(np.random.default_rng(seed + len(root)).integers(0, len(indices)))]
            oracle = float(np.min(actual_cost[indices]))
            selected_errors.append(float(actual_cost[selected] - oracle))
            random_errors.append(float(actual_cost[random_index] - oracle))
            oracle_errors.append(0.0)
            selected_success.append(float(actual_cost[selected] <= .25))
            random_success.append(float(actual_cost[random_index] <= .25))
        paired_regret_gain = np.asarray(random_errors, dtype=np.float64) - np.asarray(selected_errors, dtype=np.float64)
        paired_success_gain = np.asarray(selected_success, dtype=np.float64) - np.asarray(random_success, dtype=np.float64)
        reports.append({
            "seed": int(seed),
            "effect_rmse": float(np.sqrt(np.mean((prediction - outcome_y) ** 2))),
            "train_effect_rmse": float(np.sqrt(np.mean((train_prediction - train_y) ** 2))),
            "selected_regret": _bootstrap(np.asarray(selected_errors), sorted(grouped_outcome), seed + 100, bootstrap_samples),
            "random_regret": _bootstrap(np.asarray(random_errors), sorted(grouped_outcome), seed + 200, bootstrap_samples),
            "selected_success": _bootstrap(np.asarray(selected_success), sorted(grouped_outcome), seed + 300, bootstrap_samples),
            "random_success": _bootstrap(np.asarray(random_success), sorted(grouped_outcome), seed + 400, bootstrap_samples),
            "paired_regret_gain": _bootstrap(paired_regret_gain, sorted(grouped_outcome), seed + 500, bootstrap_samples),
            "paired_success_gain": _bootstrap(paired_success_gain, sorted(grouped_outcome), seed + 600, bootstrap_samples),
            "training": training,
        })
    ensemble_effect = np.mean(np.stack(all_predictions), axis=0)
    ensemble_terminal = all_sources + ensemble_effect
    ensemble_cost = np.linalg.norm(ensemble_terminal - all_goals, axis=1)
    real_cost = np.asarray([float(row["real_task_cost"]) for row in all_records], dtype=np.float64)
    groups = np.asarray(all_roots)
    calibration_groups = np.unique(groups[calibration_mask])
    calibration = calibrate_contrast_bound(
        real_cost[calibration_mask], ensemble_cost[calibration_mask], groups[calibration_mask],
        calibration_groups, alpha=0.1,
    )
    outcome_coverage = evaluate_contrast_bound(
        real_cost[outcome_mask], ensemble_cost[outcome_mask], groups[outcome_mask], calibration["bound"],
    )
    outcome_selection = _root_selection(outcome_records, ensemble_cost[outcome_mask])
    calibration_records = [row for row, keep in zip(all_records, calibration_mask) if keep]
    calibration_selection = _root_selection(
        calibration_records, ensemble_cost[calibration_mask],
    )
    root_support = _train_only_root_support(all_records, train_mask, quantile=0.9)
    supported_calibration_roots = [
        root for root in sorted(calibration_selection["roots"])
        if root_support["roots"][root]["passed"]
    ]
    regret_calibration = calibrate_root_score_bound(
        np.asarray([
            calibration_selection["roots"][root]["regret"]
            for root in supported_calibration_roots
        ]),
        alpha=0.1,
    )
    supported_outcome_roots = [
        root for root in sorted(outcome_selection["roots"])
        if root_support["roots"][root]["passed"]
    ]
    outcome_regrets_supported = np.asarray([
        outcome_selection["roots"][root]["regret"] for root in supported_outcome_roots
    ])
    regret_bound = regret_calibration["bound"]
    bounded_regret_coverage = (
        None if regret_bound is None or outcome_regrets_supported.size == 0
        else float(np.mean(outcome_regrets_supported <= float(regret_bound)))
    )
    tolerance_reports = {}
    for tolerance in (0.05, 0.1, 0.25):
        permitted = (
            supported_outcome_roots
            if regret_bound is not None and float(regret_bound) <= tolerance
            else []
        )
        tolerance_reports[str(tolerance)] = {
            "tolerance": tolerance,
            "permitted_root_count": len(permitted),
            "release_rate": float(len(permitted) / len(outcome_selection["roots"])),
            "empirical_violation_rate": None if not permitted else float(np.mean([
                outcome_selection["roots"][root]["regret"] > tolerance for root in permitted
            ])),
            "mean_released_regret": None if not permitted else float(np.mean([
                outcome_selection["roots"][root]["regret"] for root in permitted
            ])),
        }
    shuffled_selection = _root_selection(
        outcome_records,
        _cyclic_within_root(ensemble_cost[outcome_mask], outcome_records),
    )
    train_candidate_cost = {}
    for candidate in sorted({int(row["candidate_id"]) for row in all_records}):
        candidate_mask = train_mask & np.asarray([
            int(row["candidate_id"]) == candidate for row in all_records
        ])
        train_candidate_cost[candidate] = float(np.mean(real_cost[candidate_mask]))
    bank_only_cost = np.asarray([
        train_candidate_cost[int(row["candidate_id"])] for row in outcome_records
    ])
    bank_only_selection = _root_selection(outcome_records, bank_only_cost)
    no_dynamics_selection = _root_selection(outcome_records, np.zeros(len(outcome_records)))
    paired_controls = {
        "shuffled_prediction": _paired_regret_gain(
            outcome_selection, shuffled_selection, seed=20261016, samples=bootstrap_samples,
        ),
        "bank_only": _paired_regret_gain(
            outcome_selection, bank_only_selection, seed=20261017, samples=bootstrap_samples,
        ),
        "no_dynamics": _paired_regret_gain(
            outcome_selection, no_dynamics_selection, seed=20261018, samples=bootstrap_samples,
        ),
    }
    matched_order_pairs = _matched_order_pair_audit(
        all_records, ensemble_cost, train_mask=train_mask, outcome_mask=outcome_mask,
        samples=bootstrap_samples,
    )
    matched_order_calibration = _matched_order_pair_audit(
        all_records, ensemble_cost, train_mask=train_mask, outcome_mask=calibration_mask,
        samples=bootstrap_samples,
    )
    matched_supported_calibration_roots = [
        root for root in sorted(matched_order_calibration["method"]["root_regret"])
        if root_support["roots"][root]["passed"]
    ]
    matched_regret_calibration = calibrate_root_score_bound(
        np.asarray([
            matched_order_calibration["method"]["root_regret"][root]
            for root in matched_supported_calibration_roots
        ]),
        alpha=0.1,
    )
    matched_supported_outcome_roots = [
        root for root in sorted(matched_order_pairs["method"]["root_regret"])
        if root_support["roots"][root]["passed"]
    ]
    matched_bound = matched_regret_calibration["bound"]
    matched_outcome_values = np.asarray([
        matched_order_pairs["method"]["root_regret"][root]
        for root in matched_supported_outcome_roots
    ])
    matched_order_pairs["bounded_regret_certificate"] = {
        "supported_calibration_root_count": len(matched_supported_calibration_roots),
        "supported_outcome_root_count": len(matched_supported_outcome_roots),
        "calibration": matched_regret_calibration,
        "empirical_supported_outcome_coverage": (
            None if matched_bound is None or matched_outcome_values.size == 0
            else float(np.mean(matched_outcome_values <= float(matched_bound)))
        ),
        "tolerance_reports": {
            str(tolerance): {
                "tolerance": tolerance,
                "permitted_root_count": (
                    len(matched_supported_outcome_roots)
                    if matched_bound is not None and float(matched_bound) <= tolerance else 0
                ),
                "release_rate": (
                    len(matched_supported_outcome_roots) / len(matched_order_pairs["method"]["root_regret"])
                    if matched_bound is not None and float(matched_bound) <= tolerance else 0.0
                ),
                "empirical_violation_rate": (
                    float(np.mean(matched_outcome_values > tolerance))
                    if matched_bound is not None and float(matched_bound) <= tolerance and matched_outcome_values.size
                    else None
                ),
            }
            for tolerance in (0.02, 0.05, 0.1)
        },
        "certificate_type": "marginal epsilon-optimal temporal-order regret within four matched pairs",
    }

    candidate_ids = sorted({int(row["candidate_id"]) for row in all_records})
    heldout_candidate_ids = [candidate for index, candidate in enumerate(candidate_ids) if index % 4 == 3]
    new_candidate_train_mask = train_mask & np.asarray([
        int(row["candidate_id"]) not in set(heldout_candidate_ids) for row in all_records
    ])
    new_candidate_predictions = []
    for seed in SEEDS:
        prediction_all, _training = _fit_predict(
            all_x[new_candidate_train_mask], all_y[new_candidate_train_mask], all_x,
            seed, epochs, device,
        )
        new_candidate_predictions.append(prediction_all)
    new_candidate_effect = np.mean(np.stack(new_candidate_predictions), axis=0)
    new_candidate_cost = np.linalg.norm(all_sources + new_candidate_effect - all_goals, axis=1)
    new_candidate_outcome_mask = outcome_mask & np.asarray([
        int(row["candidate_id"]) in set(heldout_candidate_ids) for row in all_records
    ])
    new_candidate_records = [
        row for row, keep in zip(all_records, new_candidate_outcome_mask) if keep
    ]
    new_candidate_selection = _root_selection(
        new_candidate_records, new_candidate_cost[new_candidate_outcome_mask],
    )
    new_candidate_shuffled = _root_selection(
        new_candidate_records,
        _cyclic_within_root(new_candidate_cost[new_candidate_outcome_mask], new_candidate_records),
    )
    new_candidate_control_gain = _paired_regret_gain(
        new_candidate_selection, new_candidate_shuffled,
        seed=20261019, samples=bootstrap_samples,
    )
    certificate_roots = []
    for root in sorted(set(outcome_roots)):
        indices = np.flatnonzero(outcome_mask & (groups == root))
        certificate = certify_candidate_bank(
            ensemble_cost[indices], calibration["bound"], support_passed=True,
            candidate_ids=[int(all_records[index]["candidate_id"]) for index in indices],
        )
        selected = int(indices[int(np.argmin(ensemble_cost[indices]))])
        oracle = int(indices[int(np.argmin(real_cost[indices]))])
        certificate_roots.append({
            "instance_id": root,
            "status": certificate["status"],
            "minimum_predicted_margin": certificate["minimum_predicted_margin"],
            "ranking_correct": bool(selected == oracle),
            "regret": float(real_cost[selected] - real_cost[oracle]),
        })
    potential_releases = [row for row in certificate_roots if row["status"] == "permit"]
    prediction_rows = [
        {
            "instance_id": str(row["instance_id"]),
            "candidate_id": int(row["candidate_id"]),
            "split": str(row["split"]),
            "predicted_terminal_effect_xy": ensemble_effect[index].tolist(),
            "predicted_terminal_xy": ensemble_terminal[index].tolist(),
            "predicted_task_cost": float(ensemble_cost[index]),
            "real_task_cost": float(real_cost[index]),
        }
        for index, row in enumerate(all_records)
    ]
    report = {
        "schema": "pointmaze_intervention_supervised_candidate_effect.v1",
        "dataset": str(dataset.resolve()),
        "arrays": str(arrays.resolve()),
        "reports": reports,
        "summary": {
            "selected_regret_mean": float(np.mean([item["selected_regret"]["estimate"] for item in reports])),
            "random_regret_mean": float(np.mean([item["random_regret"]["estimate"] for item in reports])),
            "effect_rmse_mean": float(np.mean([item["effect_rmse"] for item in reports])),
            "train_effect_rmse_mean": float(np.mean([item["train_effect_rmse"] for item in reports])),
            "paired_regret_gain_mean": float(np.mean([item["paired_regret_gain"]["estimate"] for item in reports])),
            "paired_success_gain_mean": float(np.mean([item["paired_success_gain"]["estimate"] for item in reports])),
            "ensemble_outcome_regret": outcome_selection["mean_regret"],
            "shuffled_prediction_outcome_regret": shuffled_selection["mean_regret"],
        },
        "prediction_rows": prediction_rows,
        "ensemble_task_contrast": {
            "calibration": calibration,
            "outcome_coverage": outcome_coverage,
            "outcome_selection": outcome_selection,
            "bounded_regret_certificate": {
                "support": root_support,
                "supported_calibration_root_count": len(supported_calibration_roots),
                "supported_outcome_root_count": len(supported_outcome_roots),
                "calibration": regret_calibration,
                "empirical_supported_outcome_coverage": bounded_regret_coverage,
                "tolerance_reports": tolerance_reports,
                "certificate_type": "marginal epsilon-optimal fixed-bank regret; not exact-winner certification",
            },
            "shuffled_prediction_control": shuffled_selection,
            "bank_only_control": bank_only_selection,
            "no_dynamics_control": no_dynamics_selection,
            "paired_regret_gain_over_controls": paired_controls,
            "matched_order_pair_audit": matched_order_pairs,
            "new_candidate_control": {
                "partition_rule": "sorted candidate index mod 4 equals 3 is absent from adapter fit",
                "heldout_candidate_ids": heldout_candidate_ids,
                "fit_uses_outcome": False,
                "selection": new_candidate_selection,
                "shuffled_prediction": new_candidate_shuffled,
                "paired_regret_gain_over_shuffled": new_candidate_control_gain,
                "claim_boundary": "interpolation to predeclared unseen candidate templates within the fixed bank; not adaptive CEM",
            },
            "certificate_preview": {
                "support_gate_status": "not_independently_calibrated",
                "operational_release_count": 0,
                "potential_release_count_if_support_passed": len(potential_releases),
                "potential_false_release_rate": None if not potential_releases else float(np.mean([
                    not row["ranking_correct"] for row in potential_releases
                ])),
                "potential_useful_coverage": float(sum(row["ranking_correct"] for row in potential_releases) / len(certificate_roots)),
                "roots": certificate_roots,
            },
        },
        "protocol": {
            "training_split": "train roots only",
            "outcome_split": "root-disjoint outcome roots",
            "labels": "offline simulator terminal XY effect",
            "runtime_input": {
                "frozen_latent_action": "frozen source latent plus complete action sequence",
                "history_latent_action": "causal frozen latent history up to the decision point plus complete action sequence",
                "privileged_state_action": "privileged source simulator state plus complete action sequence (diagnostic upper bound only)",
                "latent_plus_state_action": "frozen source latent plus privileged source simulator state plus complete action sequence (diagnostic upper bound only)",
            }[feature_mode],
            "privileged_state_at_runtime": feature_mode in {"privileged_state_action", "latent_plus_state_action"},
            "outcome_used_for_training_or_selection": False,
            "calibration_used_only_for_frozen_conformal_bound": True,
            "adapter_role": "intervention_supervised",
            "feature_mode": feature_mode,
            "device": str(device),
            "feature_mode_boundary": "privileged_state modes are diagnostic upper bounds and not RGB-runtime evidence",
        },
        "claim_boundary": "Intervention-supervised candidate-effect adapter; certificate preview lacks an independently calibrated deployment-support gate and therefore releases no operational actions; not frozen geometry evidence, online RGB planning, or universal transfer.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--feature-mode",
        choices=("frozen_latent_action", "history_latent_action", "privileged_state_action", "latent_plus_state_action"),
        default="frozen_latent_action",
    )
    args = parser.parse_args()
    report = run(**vars(args))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
