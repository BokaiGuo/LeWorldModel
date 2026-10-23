"""Root-disjoint TwoRoom action-effect repairability experiment.

This is a deliberately narrow external-environment check of the PointMaze
intervention-supervised action-effect finding.  Every policy receives only a
current RGB frame, a rendered goal RGB frame, a frozen LeWM rollout, and the
same world-fixed macro-action bank.  Simulator state and realised outcomes
are used only to create intervention labels on train roots and evaluate
already-frozen policies on held roots.

The experiment contains both a one-pulse ranking panel and a short RGB
receding-horizon consequence.  It does *not* turn the learned adapter into a
privileged controller: at deployment the adapter predicts an RGB marker
displacement and scores it against the current/goal RGB markers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
import torch
from stable_worldmodel.policy import BasePolicy

from experiments.audit_tworoom_shared_candidate_ranking import (
    _candidate_bank,
    _model_rollout,
)
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_tworoom_planning import (
    TwoRoomGoalAliasWrapper,
    _image_transform,
    _load_action_stats,
    _rgb_red_centroid,
)
from experiments.gdd_wm.tworoom_support_gated_repair import (
    FrozenSupportGate,
    calibrate as calibrate_support_gate,
    choose_scores,
)


SCHEMA = "tworoom_action_effect_repairability.v1"
BASE_ARM_NAMES = ("frozen_visual", "effect_adapter", "action_only", "no_dynamics")


class _Preprocessor(BasePolicy):
    def __init__(self, action_normalizer: ActionNormalizer):
        super().__init__()
        self.process = {"action": action_normalizer}
        self.transform = {"pixels": _image_transform(), "goal": _image_transform()}


class _EffectMLP(torch.nn.Module):
    def __init__(self, dimensions: int, width: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dimensions, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


@dataclass(frozen=True)
class _Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "_Standardizer":
        return cls(values.mean(axis=0), np.maximum(values.std(axis=0), 1.0e-6))

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_fresh_output(output: Path) -> None:
    """Forbid accidental replacement of a root-disjoint evidence panel."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite TwoRoom repairability output: {output}")


def _root_split(root_seeds: Iterable[int], seed: int) -> dict[int, str]:
    """Return a deterministic 60/20/20 root-disjoint split."""
    roots = np.asarray(sorted({int(root) for root in root_seeds}), dtype=np.int64)
    if len(roots) < 10:
        raise ValueError("need at least ten roots for train/calibration/outcome isolation")
    order = roots.copy()
    np.random.default_rng(seed).shuffle(order)
    train_end = max(1, int(math.floor(0.60 * len(order))))
    calibration_end = train_end + max(1, int(math.floor(0.20 * len(order))))
    if calibration_end >= len(order):
        calibration_end = len(order) - 1
    split = {int(root): "train" for root in order[:train_end]}
    split.update({int(root): "calibration" for root in order[train_end:calibration_end]})
    split.update({int(root): "outcome" for root in order[calibration_end:]})
    if set(split.values()) != {"train", "calibration", "outcome"}:
        raise RuntimeError("root partition is empty")
    return split


def _feature_views(
    source: np.ndarray,
    predicted: np.ndarray,
    action: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct predeclared adapter features without state or outcome input."""
    source = np.asarray(source, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if source.ndim != 2 or predicted.shape != source.shape or action.shape != (len(source), 2):
        raise ValueError("invalid frozen latent/action feature shapes")
    delta = predicted - source
    return {
        "effect_adapter": np.concatenate((source, delta, action), axis=1),
        "no_dynamics": np.concatenate((source, action), axis=1),
        "action_only": action.copy(),
    }


def _effect_scores(
    start_marker: np.ndarray,
    goal_marker: np.ndarray,
    effects: np.ndarray,
) -> np.ndarray:
    """Predeclared RGB utility: marker distance after the predicted pulse."""
    return np.linalg.norm(
        np.asarray(start_marker, dtype=np.float64)[None, :]
        + np.asarray(effects, dtype=np.float64)
        - np.asarray(goal_marker, dtype=np.float64)[None, :], axis=1,
    )


def _bootstrap_delta(values: np.ndarray, seed: int, samples: int) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "samples": int(samples),
    }


def _replay_candidate(
    env: gym.Env,
    root_seed: int,
    action: np.ndarray,
    steps: int,
) -> tuple[float, np.ndarray, bool]:
    """Evaluator-only counterfactual replay from a root; never policy input."""
    env.reset(seed=int(root_seed))
    final_info: dict[str, Any] = {}
    terminated = truncated = False
    for _ in range(int(steps)):
        _, _, terminated, truncated, final_info = env.step(np.asarray(action, dtype=np.float32))
        if terminated or truncated:
            break
    frame = np.asarray(env.render(), dtype=np.uint8).copy()
    distance = float(np.asarray(final_info.get("distance_to_target", np.inf)).reshape(-1)[0])
    return distance, frame, bool(terminated)


def _collect_panel(
    *,
    model: torch.nn.Module,
    preprocessor: _Preprocessor,
    action_normalizer: ActionNormalizer,
    raw_bank: list[tuple[float, float]],
    normalized_bank: torch.Tensor,
    root_seeds: list[int],
    pulse_steps: int,
    device: str,
) -> dict[str, Any]:
    """Collect frozen RGB features and evaluator-only intervention effects."""
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    source_latent: list[np.ndarray] = []
    predicted_latent: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    observed_effect: list[np.ndarray] = []
    real_cost: list[float] = []
    root_ids: list[int] = []
    frozen_cost: list[float] = []
    initial_marker: dict[int, np.ndarray] = {}
    goal_marker: dict[int, np.ndarray] = {}
    true_cost_by_root: dict[int, list[float]] = {}
    try:
        for number, root_seed in enumerate(root_seeds, start=1):
            _, info = env.reset(seed=int(root_seed))
            current = np.asarray(env.render(), dtype=np.uint8).copy()
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            start = _rgb_red_centroid(current)
            target = _rgb_red_centroid(goal)
            model_cost, source, predicted, _ = _model_rollout(
                model, preprocessor, current, goal, action_normalizer, normalized_bank,
                device, history_len=1, return_goal=True,
            )
            initial_marker[int(root_seed)] = start.astype(np.float32)
            goal_marker[int(root_seed)] = target.astype(np.float32)
            root_costs: list[float] = []
            for candidate, action in enumerate(raw_bank):
                distance, final_frame, _ = _replay_candidate(env, root_seed, np.asarray(action), pulse_steps)
                source_latent.append(np.asarray(source[candidate], dtype=np.float32))
                predicted_latent.append(np.asarray(predicted[candidate], dtype=np.float32))
                action_rows.append(np.asarray(action, dtype=np.float32))
                observed_effect.append((_rgb_red_centroid(final_frame) - start).astype(np.float32))
                real_cost.append(distance)
                root_ids.append(int(root_seed))
                frozen_cost.append(float(model_cost[candidate]))
                root_costs.append(distance)
            true_cost_by_root[int(root_seed)] = root_costs
            print(f"[tworoom-effect] collect root={number}/{len(root_seeds)} seed={root_seed}", flush=True)
    finally:
        env.close()
    return {
        "source": np.stack(source_latent),
        "predicted": np.stack(predicted_latent),
        "action": np.stack(action_rows),
        "effect": np.stack(observed_effect),
        "real_cost": np.asarray(real_cost, dtype=np.float64),
        "root": np.asarray(root_ids, dtype=np.int64),
        "frozen_cost": np.asarray(frozen_cost, dtype=np.float64),
        "initial_marker": initial_marker,
        "goal_marker": goal_marker,
        "true_cost_by_root": true_cost_by_root,
    }


def _merge_panels(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Join pre-freeze development and post-freeze outcome panels."""
    array_keys = ("source", "predicted", "action", "effect", "real_cost", "root", "frozen_cost")
    merged = {key: np.concatenate((left[key], right[key]), axis=0) for key in array_keys}
    merged["initial_marker"] = {**left["initial_marker"], **right["initial_marker"]}
    merged["goal_marker"] = {**left["goal_marker"], **right["goal_marker"]}
    merged["true_cost_by_root"] = {**left["true_cost_by_root"], **right["true_cost_by_root"]}
    return merged


def _fit_effect_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> tuple[_EffectMLP, _Standardizer, dict[str, Any]]:
    torch.manual_seed(int(seed))
    standardizer = _Standardizer.fit(x)
    model = _EffectMLP(x.shape[1], width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    features = torch.as_tensor(standardizer.apply(x), device=device)
    targets = torch.as_tensor(y, device=device)
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = torch.nn.functional.smooth_l1_loss(prediction, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        train_rmse = float(torch.sqrt(torch.mean((model(features) - targets).square())).item())
    return model, standardizer, {"device": str(device), "train_effect_rmse": train_rmse, "epochs": int(epochs)}


def _predict_effect(
    model: _EffectMLP,
    standardizer: _Standardizer,
    values: np.ndarray,
    device: str,
) -> np.ndarray:
    with torch.inference_mode():
        result = model(torch.as_tensor(standardizer.apply(values), device=device)).cpu().numpy()
    return np.asarray(result, dtype=np.float64)


def _root_regrets(
    root_seeds: list[int],
    costs: np.ndarray,
    panel: dict[str, Any],
    count: int,
) -> dict[str, Any]:
    values: dict[str, float] = {}
    selections: dict[str, int] = {}
    for root in root_seeds:
        mask = panel["root"] == int(root)
        candidate = int(np.argmin(costs[mask]))
        actual = panel["real_cost"][mask]
        values[str(root)] = float(actual[candidate] - actual.min())
        selections[str(root)] = candidate
    return {
        "root_regret": values,
        "selected_candidate": selections,
        "mean_regret": float(np.mean(list(values.values()))),
    }


def _scores_from_effect_prediction(panel: dict[str, Any], prediction: np.ndarray) -> np.ndarray:
    """Score candidates using only a frozen predicted RGB effect."""
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != np.asarray(panel["effect"]).shape:
        raise ValueError("effect prediction shape does not match panel")
    scores = np.empty(len(values), dtype=np.float64)
    for root in sorted({int(value) for value in panel["root"]}):
        mask = panel["root"] == root
        scores[mask] = _effect_scores(
            panel["initial_marker"][root], panel["goal_marker"][root], values[mask],
        )
    return scores


def _closed_loop(
    *,
    arm: str,
    roots: list[int],
    model: torch.nn.Module,
    preprocessor: _Preprocessor,
    action_normalizer: ActionNormalizer,
    raw_bank: list[tuple[float, float]],
    normalized_bank: torch.Tensor,
    fitted: dict[str, tuple[_EffectMLP, _Standardizer]],
    device: str,
    pulse_steps: int,
    decision_blocks: int,
    support_gate: FrozenSupportGate | None = None,
) -> dict[str, Any]:
    """RGB-only receding horizon evaluation after all adapters are frozen."""
    env = TwoRoomGoalAliasWrapper(gym.make("swm/TwoRoom-v1", render_mode="rgb_array"))
    final_distance: list[float] = []
    successes: list[int] = []
    chosen: dict[str, list[int]] = {}
    try:
        for root in roots:
            _, info = env.reset(seed=int(root))
            goal = np.asarray(info["goal"], dtype=np.uint8).copy()
            arm_actions: list[int] = []
            distance = float("inf")
            done = False
            for _ in range(int(decision_blocks)):
                current = np.asarray(env.render(), dtype=np.uint8).copy()
                model_cost, source, predicted, _ = _model_rollout(
                    model, preprocessor, current, goal, action_normalizer, normalized_bank,
                    device, history_len=1, return_goal=True,
                )
                if arm == "frozen_visual":
                    candidate = int(np.argmin(model_cost))
                elif arm == "support_gated":
                    if support_gate is None:
                        raise ValueError("support_gated arm requires a frozen support gate")
                    views = _feature_views(source, predicted, np.asarray(raw_bank, dtype=np.float32))
                    adapter_effect = _predict_effect(*fitted["effect_adapter"], views["effect_adapter"], device)
                    fallback_effect = _predict_effect(*fitted["action_only"], views["action_only"], device)
                    adapter_scores = _effect_scores(
                        _rgb_red_centroid(current), _rgb_red_centroid(goal), adapter_effect,
                    )
                    fallback_scores = _effect_scores(
                        _rgb_red_centroid(current), _rgb_red_centroid(goal), fallback_effect,
                    )
                    selected_scores, _ = choose_scores(
                        support_gate, views["effect_adapter"], adapter_scores, fallback_scores,
                    )
                    candidate = int(np.argmin(selected_scores))
                else:
                    view = _feature_views(source, predicted, np.asarray(raw_bank, dtype=np.float32))[arm]
                    predicted_effect = _predict_effect(*fitted[arm], view, device)
                    candidate = int(np.argmin(_effect_scores(
                        _rgb_red_centroid(current), _rgb_red_centroid(goal), predicted_effect,
                    )))
                arm_actions.append(candidate)
                final_info: dict[str, Any] = {}
                for _ in range(int(pulse_steps)):
                    _, _, terminated, truncated, final_info = env.step(np.asarray(raw_bank[candidate], dtype=np.float32))
                    if terminated or truncated:
                        done = bool(terminated)
                        break
                distance = float(np.asarray(final_info.get("distance_to_target", np.inf)).reshape(-1)[0])
                if done or truncated:
                    break
            final_distance.append(distance)
            successes.append(int(done))
            chosen[str(root)] = arm_actions
    finally:
        env.close()
    return {
        "roots": [int(root) for root in roots],
        "final_distance": final_distance,
        "success": successes,
        "mean_final_distance": float(np.mean(final_distance)),
        "success_rate": float(np.mean(successes)),
        "chosen_candidates": chosen,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _require_fresh_output(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for TwoRoom repairability experiment but unavailable")
    if args.roots < 20 or args.pulse_steps <= 0 or args.decision_blocks <= 0:
        raise ValueError("require roots >= 20 and positive pulse/decision lengths")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    action_mean, action_std = _load_action_stats(args.adapter_artifact, "swm/TwoRoom-v1")
    raw_bank, normalized_bank = _candidate_bank(action_mean, action_std, horizon=5, action_block=5)
    raw_bank_array = np.asarray(raw_bank, dtype=np.float32)
    roots = [int(args.seed + index) for index in range(args.roots)]
    split = _root_split(roots, args.seed + 17)
    development_roots = sorted(root for root, label in split.items() if label in {"train", "calibration"})
    outcome_roots = sorted(root for root, label in split.items() if label == "outcome")
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(ActionNormalizer(action_mean, action_std))
    # Outcome roots are intentionally not even materialised until after all
    # train-root fitting has completed.  This is stronger than simply masking
    # their labels out of a preloaded array and makes the execution order auditable.
    development_panel = _collect_panel(
        model=model, preprocessor=preprocessor, action_normalizer=ActionNormalizer(action_mean, action_std),
        raw_bank=raw_bank, normalized_bank=normalized_bank, root_seeds=development_roots,
        pulse_steps=args.pulse_steps, device=args.device,
    )
    development_features = _feature_views(
        development_panel["source"], development_panel["predicted"], development_panel["action"],
    )
    development_masks = {
        name: np.asarray([split[int(root)] == name for root in development_panel["root"]])
        for name in ("train", "calibration")
    }
    arm_names = BASE_ARM_NAMES + (("support_gated",) if args.support_gated_repair else ())
    training: dict[str, Any] = {}
    fitted: dict[str, tuple[_EffectMLP, _Standardizer]] = {}
    for offset, arm in enumerate(("effect_adapter", "action_only", "no_dynamics")):
        fitted_model, fitted_standardizer, fit_metadata = _fit_effect_model(
            development_features[arm][development_masks["train"]],
            development_panel["effect"][development_masks["train"]], width=args.width,
            epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed + offset,
            device=args.device,
        )
        fitted[arm] = (fitted_model, fitted_standardizer)
        prediction = _predict_effect(*fitted[arm], development_features[arm], args.device)
        training[arm] = {
            **fit_metadata, "width": int(args.width),
            "feature_dimension": int(development_features[arm].shape[1]),
            "train_effect_rmse": float(np.sqrt(np.mean((
                prediction[development_masks["train"]] - development_panel["effect"][development_masks["train"]]
            ) ** 2))),
            "calibration_effect_rmse": float(np.sqrt(np.mean((
                prediction[development_masks["calibration"]] - development_panel["effect"][development_masks["calibration"]]
            ) ** 2))),
        }
    support_gate: FrozenSupportGate | None = None
    support_gate_receipt: dict[str, Any] | None = None
    if args.support_gated_repair:
        development_predictions = {
            arm: _predict_effect(*fitted[arm], development_features[arm], args.device)
            for arm in ("effect_adapter", "action_only")
        }
        development_scores = {
            arm: _scores_from_effect_prediction(development_panel, prediction)
            for arm, prediction in development_predictions.items()
        }
        calibration_roots = sorted(root for root, label in split.items() if label == "calibration")
        adapter_regret = _root_regrets(
            calibration_roots, development_scores["effect_adapter"], development_panel, len(raw_bank),
        )["root_regret"]
        fallback_regret = _root_regrets(
            calibration_roots, development_scores["action_only"], development_panel, len(raw_bank),
        )["root_regret"]
        support_gate, support_gate_receipt = calibrate_support_gate(
            train_features=development_features["effect_adapter"][development_masks["train"]],
            calibration_features=development_features["effect_adapter"][development_masks["calibration"]],
            calibration_roots=development_panel["root"][development_masks["calibration"]],
            adapter_regret={int(root): float(value) for root, value in adapter_regret.items()},
            fallback_regret={int(root): float(value) for root, value in fallback_regret.items()},
            minimum_release_coverage=args.support_gate_minimum_coverage,
        )
    # Only now is the held-out outcome panel executed and read.
    outcome_panel = _collect_panel(
        model=model, preprocessor=preprocessor, action_normalizer=ActionNormalizer(action_mean, action_std),
        raw_bank=raw_bank, normalized_bank=normalized_bank, root_seeds=outcome_roots,
        pulse_steps=args.pulse_steps, device=args.device,
    )
    panel = _merge_panels(development_panel, outcome_panel)
    features = _feature_views(panel["source"], panel["predicted"], panel["action"])
    masks = {name: np.asarray([split[int(root)] == name for root in panel["root"]]) for name in ("train", "calibration", "outcome")}
    scores: dict[str, np.ndarray] = {"frozen_visual": panel["frozen_cost"]}
    for arm in ("effect_adapter", "action_only", "no_dynamics"):
        prediction = _predict_effect(*fitted[arm], features[arm], args.device)
        training[arm]["outcome_effect_rmse"] = float(np.sqrt(np.mean((
            prediction[masks["outcome"]] - panel["effect"][masks["outcome"]]
        ) ** 2)))
        scores[arm] = _scores_from_effect_prediction(panel, prediction)
    if support_gate is not None:
        gated_scores = np.empty(len(panel["root"]), dtype=np.float64)
        for root in roots:
            mask = panel["root"] == root
            selected, _ = choose_scores(
                support_gate, features["effect_adapter"][mask],
                scores["effect_adapter"][mask], scores["action_only"][mask],
            )
            gated_scores[mask] = selected
        scores["support_gated"] = gated_scores
    ranking = {
        arm: {
            split_name: _root_regrets(
                sorted(root for root, label in split.items() if label == split_name), scores[arm], panel, len(raw_bank),
            )
            for split_name in ("calibration", "outcome")
        }
        for arm in arm_names
    }
    closed_loop = {
        arm: _closed_loop(
            arm=arm, roots=outcome_roots, model=model, preprocessor=preprocessor,
            action_normalizer=ActionNormalizer(action_mean, action_std), raw_bank=raw_bank,
            normalized_bank=normalized_bank, fitted=fitted, device=args.device,
            pulse_steps=args.pulse_steps, decision_blocks=args.decision_blocks,
            support_gate=support_gate,
        )
        for arm in arm_names
    }
    comparisons: dict[str, Any] = {}
    baseline_regret = ranking["frozen_visual"]["outcome"]["root_regret"]
    baseline_distance = np.asarray(closed_loop["frozen_visual"]["final_distance"], dtype=np.float64)
    for arm in ("effect_adapter", "action_only", "no_dynamics"):
        regret_delta = np.asarray([
            baseline_regret[str(root)] - ranking[arm]["outcome"]["root_regret"][str(root)]
            for root in outcome_roots
        ])
        distance_delta = baseline_distance - np.asarray(closed_loop[arm]["final_distance"], dtype=np.float64)
        comparisons[arm] = {
            "offline_regret_gain_over_frozen": _bootstrap_delta(regret_delta, args.seed + 101, args.bootstrap_samples),
            "closed_loop_distance_gain_over_frozen": _bootstrap_delta(distance_delta, args.seed + 211, args.bootstrap_samples),
        }
    # The mechanistic question is not merely whether any supervised arm beats
    # the frozen visual baseline.  The full latent-plus-predicted-dynamics
    # adapter must also improve on the two deliberately simpler controls.
    for control, offset in (("action_only", 307), ("no_dynamics", 401)):
        adapter_regret = ranking["effect_adapter"]["outcome"]["root_regret"]
        control_regret = ranking[control]["outcome"]["root_regret"]
        adapter_distance = np.asarray(closed_loop["effect_adapter"]["final_distance"], dtype=np.float64)
        control_distance = np.asarray(closed_loop[control]["final_distance"], dtype=np.float64)
        comparisons[f"effect_adapter_over_{control}"] = {
            "offline_regret_gain": _bootstrap_delta(np.asarray([
                control_regret[str(root)] - adapter_regret[str(root)] for root in outcome_roots
            ]), args.seed + offset, args.bootstrap_samples),
            "closed_loop_distance_gain": _bootstrap_delta(
                control_distance - adapter_distance, args.seed + offset + 1, args.bootstrap_samples,
            ),
        }
    if support_gate is not None:
        for control, offset in (("action_only", 509), ("effect_adapter", 613)):
            gated_regret = ranking["support_gated"]["outcome"]["root_regret"]
            control_regret = ranking[control]["outcome"]["root_regret"]
            gated_distance = np.asarray(closed_loop["support_gated"]["final_distance"], dtype=np.float64)
            control_distance = np.asarray(closed_loop[control]["final_distance"], dtype=np.float64)
            comparisons[f"support_gated_over_{control}"] = {
                "offline_regret_gain": _bootstrap_delta(np.asarray([
                    control_regret[str(root)] - gated_regret[str(root)] for root in outcome_roots
                ]), args.seed + offset, args.bootstrap_samples),
                "closed_loop_distance_gain": _bootstrap_delta(
                    control_distance - gated_distance, args.seed + offset + 1, args.bootstrap_samples,
                ),
            }
    action_hashes = {_json_hash(np.asarray(action, dtype=np.float64).round(8).tolist()) for action in raw_bank}
    report = {
        "schema": SCHEMA,
        "protocol": {
            "environment": "swm/TwoRoom-v1", "checkpoint": str(args.policy), "checkpoint_frozen": True,
            "device": args.device, "cuda_available": bool(torch.cuda.is_available()),
            "root_disjoint": True, "root_counts": {name: int(sum(value == name for value in split.values())) for name in ("train", "calibration", "outcome")},
            "root_ids": {name: [root for root, value in split.items() if value == name] for name in ("train", "calibration", "outcome")},
            "candidate_bank": {"count": len(raw_bank), "raw_action_hashes": sorted(action_hashes), "same_world_fixed_bank_all_roots": True, "pulse_steps": int(args.pulse_steps)},
            "policy_input_contract": "current RGB, rendered goal RGB, frozen LeWM source/predicted latents, and declared fixed action bank only",
            "runtime_state_access": False,
            "training_outcome_roots_read": False,
            "outcome_used_for_selection": False,
            "outcome_execution_order": "frozen train/calibration adapter fit precedes first outcome counterfactual replay",
            "adapter_target": "observed RGB red-marker displacement after one fixed action pulse",
            "closed_loop": {"enabled": True, "decision_blocks": int(args.decision_blocks), "all_arms_same_policy_input_and_bank": True},
            "support_gated_repair": bool(args.support_gated_repair),
        },
        "training": training,
        "support_gate": support_gate_receipt,
        "ranking": ranking,
        "closed_loop": closed_loop,
        "comparisons": comparisons,
        "claim_boundary": (
            "Root-disjoint TwoRoom reproduction check only. Positive or negative results do not establish "
            "cross-environment generalization, a causal world model, or a universal transfer certificate."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "panel_arrays.npz", source=panel["source"].astype(np.float32),
        predicted=panel["predicted"].astype(np.float32), action=panel["action"].astype(np.float32),
        observed_rgb_effect=panel["effect"].astype(np.float32), real_task_cost=panel["real_cost"].astype(np.float32),
        root_id=panel["root"].astype(np.int64), frozen_visual_cost=panel["frozen_cost"].astype(np.float32),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        arm: {"offline_regret": ranking[arm]["outcome"]["mean_regret"], "closed_loop_distance": closed_loop[arm]["mean_final_distance"]}
        for arm in arm_names
    }, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm")
    parser.add_argument("--adapter-artifact", type=Path, default=Path("results/live_cross_environment_prediction_20261015/tworoom_cuda/cross_environment_prediction.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--roots", type=int, default=80)
    parser.add_argument("--pulse-steps", type=int, default=5)
    parser.add_argument("--decision-blocks", type=int, default=6)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--support-gated-repair", action="store_true")
    parser.add_argument("--support-gate-minimum-coverage", type=float, default=0.10)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
