"""Evaluate held-out candidate-action ranking for the RGB spatial adapter.

This is an independent consequence of the saved root-disjoint action-effect
panel.  It does not use simulator state, goal distance, or planning success.
For each declared candidate-bank group, the score of a candidate is the
predicted displacement along the calibration-only RGB goal direction. The
report compares the learned adapter against the train-only constant target
baseline and a local probe-fitted action map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.train_cube_multiview_spatial_adapter import SpatialAdapter


DEFAULT_MIN_PROBE_EFFECT = 5.0e-4


def _pairwise_agreement(true_scores: np.ndarray, predicted_scores: np.ndarray) -> float:
    pairs = []
    for i in range(len(true_scores)):
        for j in range(i + 1, len(true_scores)):
            if true_scores[i] == true_scores[j]:
                continue
            pairs.append(float((true_scores[i] - true_scores[j]) * (predicted_scores[i] - predicted_scores[j]) > 0))
    return float(np.mean(pairs)) if pairs else 0.0


def _score_rows(actions: np.ndarray, effects: np.ndarray) -> np.ndarray:
    # First two channels are front-view object XY effect.  The side-view
    # channels are retained by the adapter but are not used to define this
    # ranking, keeping the consequence tied to the candidate's planar action.
    return np.sum(actions * effects[:, :2], axis=1)


def _fit_local_probe_map(
    actions: np.ndarray, effects: np.ndarray, min_effect_norm: float = 1.0e-4,
) -> tuple[np.ndarray | None, dict[str, float | int | str]]:
    """Fit an action-to-front-effect map from probe rows only."""
    x = np.asarray(actions, dtype=np.float64)
    y = np.asarray(effects[:, :2], dtype=np.float64)
    rank = int(np.linalg.matrix_rank(x)) if x.ndim == 2 else 0
    if x.ndim != 2 or x.shape[1] != 2 or y.shape != x.shape or rank < 2:
        return None, {"status": "insufficient_probe_support", "rank": rank}
    mapping, _, _, singular = np.linalg.lstsq(x, y, rcond=1.0e-6)
    condition = float(singular[0] / max(singular[-1], 1.0e-8))
    norm = float(np.linalg.norm(mapping, ord="fro"))
    effect_norm = float(np.mean(np.linalg.norm(y, axis=1)))
    if not np.isfinite(mapping).all() or condition > 100.0 or norm < 1.0e-5 or effect_norm < float(min_effect_norm):
        return None, {
            "status": "unstable_or_weak_probe_map", "rank": rank,
            "condition": condition, "mapping_norm": norm,
            "probe_effect_norm": effect_norm, "min_effect_norm": float(min_effect_norm),
        }
    return mapping, {
        "status": "ok", "rank": rank, "condition": condition,
        "mapping_norm": norm, "probe_effect_norm": effect_norm,
    }


def _probe_and_held_indices(action_bank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed probe and held positions for the declared action bank."""
    bank = np.asarray(action_bank, dtype=np.float64)
    if bank.ndim != 2 or bank.shape[1] != 2:
        raise ValueError(f"candidate action bank must have shape (n, 2), got {bank.shape}")
    if len(bank) == 4:
        probe = np.asarray([0, 2], dtype=np.int64)
    elif len(bank) == 8:
        # +x, +y, -x: rank-2 probes with five genuinely held candidates.
        probe = np.asarray([0, 2, 4], dtype=np.int64)
    else:
        raise ValueError(f"unsupported candidate bank size: {len(bank)}")
    probe_set = set(probe.tolist())
    held = np.asarray([i for i in range(len(bank)) if i not in probe_set], dtype=np.int64)
    return probe, held


def evaluate(
    npz_path: Path,
    checkpoint_path: Path,
    device: str = "cuda",
    min_probe_effect: float = DEFAULT_MIN_PROBE_EFFECT,
) -> dict[str, object]:
    if not np.isfinite(float(min_probe_effect)) or float(min_probe_effect) < 0.0:
        raise ValueError("min_probe_effect must be finite and non-negative")
    data = np.load(npz_path)
    test_features = np.asarray(data["test_features"], dtype=np.float32)
    test_targets = np.asarray(data["test_targets"], dtype=np.float32)
    train_targets = np.asarray(data["train_targets"], dtype=np.float32)
    test_desired = np.asarray(data["test_desired"], dtype=np.float32)
    actions = test_features[:, -2:]
    candidate_actions = np.asarray(
        data.get("candidate_actions", np.asarray(
            [[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]],
            dtype=np.float32,
        )),
        dtype=np.float32,
    )
    probe_positions, held_positions = _probe_and_held_indices(candidate_actions)
    group_size = len(candidate_actions)
    if len(test_targets) % group_size != 0:
        raise ValueError("test panel row count is not divisible by candidate bank size")
    torch_device = torch.device(device)
    model = SpatialAdapter(input_dim=test_features.shape[1] - 2).to(torch_device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    protocol = checkpoint["protocol"]
    target_mean = np.asarray(protocol["target_mean"], dtype=np.float32)
    target_scale = np.asarray(protocol["target_scale"], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        _reconstruction, predicted = model(
            torch.from_numpy(test_features[:, :-2]).to(torch_device),
            torch.from_numpy(actions).to(torch_device),
        )
    predicted = predicted.detach().cpu().numpy() * target_scale + target_mean
    constant = np.repeat(np.mean(train_targets, axis=0, keepdims=True), len(test_targets), axis=0)
    train_actions = np.asarray(data["train_features"][:, -2:], dtype=np.float64)
    chart_map, _, _, _ = np.linalg.lstsq(train_actions, train_targets[:, :2], rcond=1.0e-6)
    chart_predicted = actions @ chart_map
    reports = {}
    for name, effects in (("adapter", predicted), ("zero_baseline", constant)):
        top1 = []
        pairwise = []
        for start in range(0, len(test_targets), group_size):
            stop = start + group_size
            true_scores = _score_rows(actions[start:stop], test_targets[start:stop])
            predicted_scores = _score_rows(actions[start:stop], effects[start:stop])
            top1.append(int(np.argmax(true_scores) == np.argmax(predicted_scores)))
            pairwise.append(_pairwise_agreement(true_scores, predicted_scores))
        reports[name] = {
            "groups": len(top1),
            "top1_agreement": float(np.mean(top1)),
            "pairwise_agreement": float(np.mean(pairwise)),
        }
    # Use fixed short probes; only held candidates are ranked, so evaluated
    # candidate outcomes are not used to fit the local map.
    local_top1, chart_top1, global_top1 = [], [], []
    local_pairwise, chart_pairwise, global_pairwise = [], [], []
    residuals, gate_failures, gate_reasons = [], 0, {}
    support_top1, support_pairwise = [], []
    for start in range(0, len(test_targets), group_size):
        probe = start + probe_positions
        held = start + held_positions
        mapping, diagnostics = _fit_local_probe_map(
            actions[probe], test_targets[probe], min_effect_norm=float(min_probe_effect),
        )
        if mapping is None:
            gate_failures += 1
            reason = str(diagnostics.get("status", "unknown"))
            gate_reasons[reason] = gate_reasons.get(reason, 0) + 1
            local_pred = chart_predicted[held]
        else:
            local_pred = actions[held] @ mapping
            residuals.extend(np.sqrt(np.mean((actions[probe] @ mapping - test_targets[probe, :2]) ** 2, axis=1)).tolist())
        desired = test_desired[start, :2]
        true_scores = _score_rows(actions[held], test_targets[held]) * 0.0
        true_scores = np.sum(test_targets[held, :2] * desired[None, :], axis=1)
        local_scores = np.sum(local_pred * desired[None, :], axis=1)
        chart_scores = np.sum(chart_predicted[held] * desired[None, :], axis=1)
        global_scores = np.sum(predicted[held, :2] * desired[None, :], axis=1)
        local_top1.append(int(np.argmax(local_scores) == np.argmax(true_scores)))
        chart_top1.append(int(np.argmax(chart_scores) == np.argmax(true_scores)))
        global_top1.append(int(np.argmax(global_scores) == np.argmax(true_scores)))
        local_pairwise.append(_pairwise_agreement(true_scores, local_scores))
        chart_pairwise.append(_pairwise_agreement(true_scores, chart_scores))
        global_pairwise.append(_pairwise_agreement(true_scores, global_scores))
        if mapping is not None:
            support_top1.append(int(np.argmax(local_scores) == np.argmax(true_scores)))
            support_pairwise.append(_pairwise_agreement(true_scores, local_scores))
    reports["verified_local_unprobed_candidates"] = {
        "groups": len(local_top1),
        "probe_actions": candidate_actions[probe_positions].tolist(),
        "ranked_actions": candidate_actions[held_positions].tolist(),
        "top1_agreement": float(np.mean(local_top1)),
        "pairwise_agreement": float(np.mean(local_pairwise)),
        "probe_effect_rmse": float(np.mean(residuals)) if residuals else None,
        "gate_failures": int(gate_failures),
        "gate_reasons": gate_reasons,
        "reprobe_or_fallback_rate": float(gate_failures / max(len(local_top1), 1)),
        "minimum_probe_effect_norm": float(min_probe_effect),
        "support_eligible_groups": int(len(support_top1)),
        "support_coverage": float(len(support_top1) / max(len(local_top1), 1)),
        "support_conditioned_top1_agreement": (
            float(np.mean(support_top1)) if support_top1 else None
        ),
        "support_conditioned_pairwise_agreement": (
            float(np.mean(support_pairwise)) if support_pairwise else None
        ),
    }
    reports["fixed_action_chart_unprobed_candidates"] = {
        "top1_agreement": float(np.mean(chart_top1)),
        "pairwise_agreement": float(np.mean(chart_pairwise)),
    }
    reports["global_adapter_unprobed_candidates"] = {
        "top1_agreement": float(np.mean(global_top1)),
        "pairwise_agreement": float(np.mean(global_pairwise)),
    }
    return {
        "schema": "cube_spatial_adapter_transfer.v2",
        "protocol": {
            "root_disjoint": True,
            "candidate_bank": candidate_actions.tolist(),
            "probe_positions": probe_positions.tolist(),
            "held_positions": held_positions.tolist(),
            "ranking_score": "front RGB object delta projected onto calibration-only goal displacement",
            "held_candidate_outcomes_used_for_selection": False,
            "probe_outcomes_used_for_local_update": True,
            "probe_candidates_excluded_from_ranking": True,
            "minimum_probe_effect_norm": float(min_probe_effect),
        },
        "adapter": reports["adapter"],
        "zero_baseline": reports["zero_baseline"],
        "verified_local_unprobed_candidates": reports["verified_local_unprobed_candidates"],
        "fixed_action_chart_unprobed_candidates": reports["fixed_action_chart_unprobed_candidates"],
        "global_adapter_unprobed_candidates": reports["global_adapter_unprobed_candidates"],
        "delta_adapter_minus_baseline": {
            key: reports["adapter"][key] - reports["zero_baseline"][key]
            for key in ("top1_agreement", "pairwise_agreement")
        },
        "unprobed_candidate_delta_verified_local_minus_global": {
            "top1_agreement": float(np.mean(local_top1) - np.mean(global_top1)),
            "pairwise_agreement": float(np.mean(local_pairwise) - np.mean(global_pairwise)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--min-probe-effect", type=float, default=DEFAULT_MIN_PROBE_EFFECT,
        help="fail-closed threshold for mean observed probe effect norm",
    )
    args = parser.parse_args()
    if args.device == "cuda":
        from experiments.gpu_preflight import require_cuda_allocation
        require_cuda_allocation("cuda")
    report = evaluate(
        args.data, args.checkpoint, args.device,
        min_probe_effect=float(args.min_probe_effect),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
