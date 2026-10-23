"""Independent balanced confirmation of a narrow PushT RGB-history mechanism.

This is intentionally an *offline representation* experiment.  Every root is
reset to the same contact state for a zero, ``+y`` and ``-y`` pulse.  The
offline label is which of the two non-zero, two-step pulses has the larger
zero-drift-corrected angular response.  A decoder sees only rendered RGB
history and the declared history actions; it never sees simulator state,
successor RGB, a goal, or the privileged label.

The panel was predeclared after the torque-contrast screen identified
``magnitude=0.55, duration=2`` as a potentially measurable regime.  It
collects all train/calibration/test labels before *any* decoder is fitted and
stratifies every split by those offline labels.  In particular, the held test
roots are balanced before training, so balanced accuracy is identifiable.

This runner does not select or execute decoder-chosen actions.  It therefore
cannot establish a planner, pose closure, world-model causality, or a
deployment action-ranking benefit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm/PushT-v1
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gpu_preflight import require_cuda_allocation  # noqa: E402
from experiments.run_pusht_contact_history_velocity_diagnostic import (  # noqa: E402
    collect_trajectory_data,
    mode_for_state,
)

gym.logger.min_level = 40

SCHEMA = "pusht.balanced_temporal_torque_confirmation.v1"
TORQUE_MAGNITUDE = 0.55
TORQUE_DURATION = 2
# These bands are deliberately far outside every 2026-10-21 PushT smoke band
# recorded in this project (20M / 202M ranges).  They are not inferred from
# outcomes and are frozen in the result manifest.
DEFAULT_SEED_BANDS = {
    "train": 1_700_210_000,
    "calibration": 1_700_220_000,
    "test": 1_700_230_000,
}


@dataclass(frozen=True)
class RootExample:
    """One root-level RGB/action history and its offline-only label."""

    root_id: int
    root_digest: str
    rgb_frames: np.ndarray
    history_actions: np.ndarray
    preference: int
    preference_margin: float
    zero_corrected_plus_response: float
    zero_corrected_minus_response: float


class BalancedCohortUnavailable(RuntimeError):
    """A declared root band cannot support its pre-outcome balanced cohort.

    The exception deliberately carries aggregate collection diagnostics only.
    A caller can persist a fail-closed rejection artifact without fabricating a
    decoder score or exposing held-out root-level labels.
    """

    def __init__(self, message: str, *, collection: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.collection = dict(collection)


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    """Map angles into [-pi, pi)."""

    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def state_digest(state: np.ndarray) -> str:
    """Stable digest used to detect root reuse across split bands."""

    return hashlib.sha256(np.asarray(state, dtype=np.float64).tobytes()).hexdigest()


def validate_seed_bands(bands: dict[str, int]) -> None:
    """Reject overlapping or implausible seed bands before environment work."""

    if set(bands) != {"train", "calibration", "test"}:
        raise ValueError("exactly train, calibration, and test seed bands are required")
    values = [int(bands[key]) for key in ("train", "calibration", "test")]
    if len(set(values)) != 3:
        raise ValueError("train/calibration/test seed bands must be distinct")
    if any(value < 1_000_000_000 or value >= 2**32 - 100_000 for value in values):
        raise ValueError("confirmation seed bands must be distinct, high, and valid Gym seeds")


def _goal_for_root(state: np.ndarray, root_id: int) -> np.ndarray:
    """Supply a deterministic rendered goal independent of offline labels."""

    rng = np.random.default_rng(1_701_000_000 + int(root_id))
    goal = np.asarray(state, dtype=np.float64).copy()
    direction = rng.normal(size=2)
    direction *= 70.0 / max(float(np.linalg.norm(direction)), 1.0e-8)
    goal[2:4] = np.clip(goal[2:4] + direction, 70.0, 442.0)
    goal[4] = float(rng.uniform(-np.pi, np.pi))
    goal[5:] = 0.0
    return goal


def fixed_history_actions(root_id: int, steps: int, magnitude: float) -> np.ndarray:
    """Predeclared cardinal schedule; it has no state/label-dependent branch."""

    axes = np.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)), dtype=np.float32)
    indices = (np.arange(int(steps), dtype=np.int64) + int(root_id)) % len(axes)
    return (axes[indices] * float(magnitude)).astype(np.float32)


def advance(env: gym.Env, action: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply a declared pulse and return its final real state/RGB frame."""

    if int(steps) < 1:
        raise ValueError("all declared pulse durations must be positive")
    state: np.ndarray | None = None
    frame: np.ndarray | None = None
    for _ in range(int(steps)):
        observation, _reward, _terminated, _truncated, _info = env.step(np.asarray(action, dtype=np.float32))
        state = np.asarray(observation["state"], dtype=np.float64).copy()
        frame = np.asarray(env.render(), dtype=np.uint8).copy()
    assert state is not None and frame is not None
    return state, frame


def offline_preference(
    *, before_angle: float, zero_angle: float, plus_angle: float, minus_angle: float, minimum_margin: float
) -> tuple[int | None, float, float, float]:
    """Build a two-step privileged preference label, never used as a feature."""

    zero_delta = float(wrap_angle(float(zero_angle) - float(before_angle)))
    plus_response = abs(float(wrap_angle(float(plus_angle) - float(before_angle))) - zero_delta)
    minus_response = abs(float(wrap_angle(float(minus_angle) - float(before_angle))) - zero_delta)
    margin = float(plus_response - minus_response)
    if abs(margin) < float(minimum_margin):
        return None, margin, plus_response, minus_response
    return int(margin > 0.0), margin, plus_response, minus_response


def signed_offline_preference(
    *, before_angle: float, zero_angle: float, plus_angle: float, minus_angle: float, minimum_margin: float
) -> tuple[int | None, float, float, float]:
    """Label which mirrored pulse yields the more positive angular response.

    The historical ``offline_preference`` target compares *absolute*
    zero-corrected response and is therefore a sensitivity target, not a
    torque-direction target.  This independent label retains the response
    sign.  A downstream controller with a declared desired turn direction can
    use it to select one of the two declared mirrored pulses; no goal or
    successor state is exposed to the visual readout here.
    """

    zero_delta = float(wrap_angle(float(zero_angle) - float(before_angle)))
    plus_response = float(wrap_angle(float(wrap_angle(float(plus_angle) - float(before_angle))) - zero_delta))
    minus_response = float(wrap_angle(float(wrap_angle(float(minus_angle) - float(before_angle))) - zero_delta))
    margin = float(plus_response - minus_response)
    if abs(margin) < float(minimum_margin):
        return None, margin, plus_response, minus_response
    return int(margin > 0.0), margin, plus_response, minus_response


def matched_probe_reset_seed(band_seed: int, source_index: int) -> int:
    """Return the one reset seed shared by all counterfactual probe arms.

    A valid action-effect contrast changes only the declared action.  Using
    separate reset seeds for zero/positive/negative pulses could couple the
    label to simulator reset randomness in future environment versions.
    """

    return int(band_seed) + 50_000 + int(source_index)


ProbeActionFn = Callable[[gym.Env, np.ndarray], tuple[Mapping[str, np.ndarray], Mapping[str, Any]]]


def global_y_probe_actions(_env: gym.Env, _state: np.ndarray) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
    """Historical world-frame probe family, retained for previous protocols.

    New contact-frame protocols must pass their own constructor explicitly.
    Keeping this function as the default preserves the interpretation of the
    completed absolute-response panel rather than silently changing it.
    """

    return {
        "zero": np.zeros(2, dtype=np.float32),
        "plus": np.asarray([0.0, TORQUE_MAGNITUDE], dtype=np.float32),
        "minus": np.asarray([0.0, -TORQUE_MAGNITUDE], dtype=np.float32),
    }, {"family": "world_frame_y", "pair_valid": True}


def validate_probe_actions(actions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Enforce the fixed zero/plus/minus probing contract before simulation."""

    if set(actions) != {"zero", "plus", "minus"}:
        raise ValueError("probe action constructor must return exactly zero, plus, and minus actions")
    validated = {name: np.asarray(actions[name], dtype=np.float32) for name in ("zero", "plus", "minus")}
    if any(value.shape != (2,) or not np.isfinite(value).all() for value in validated.values()):
        raise ValueError("probe actions must be finite two-dimensional vectors")
    if not np.allclose(validated["zero"], 0.0, atol=1.0e-7):
        raise ValueError("the zero-drift reference action must be exactly zero")
    return validated


def root_feature_two_frame(frames: np.ndarray, actions: np.ndarray, image_size: int) -> np.ndarray:
    """Raw reference: final two RGB frames/difference and final declared action."""

    frames, actions = validate_history(frames, actions)
    resized = resize_rgb(frames[-2:], image_size)
    rgb = torch.cat((resized[0], resized[1], resized[1] - resized[0]), dim=0).flatten().numpy()
    return np.concatenate((rgb.astype(np.float32), actions[-1].astype(np.float32)))


def root_feature_temporal_history(frames: np.ndarray, actions: np.ndarray, image_size: int) -> np.ndarray:
    """Predeclared multi-frame RGB-difference/action-history representation."""

    frames, actions = validate_history(frames, actions)
    resized = resize_rgb(frames, image_size)
    differences = resized[1:] - resized[:-1]
    rgb = torch.cat((resized[-1:], differences), dim=0).flatten().numpy()
    return np.concatenate((rgb.astype(np.float32), actions.reshape(-1).astype(np.float32)))


def validate_history(frames: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(frames, dtype=np.uint8)
    actions = np.asarray(actions, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 3:
        raise ValueError("confirmation requires at least three HWC RGB frames")
    if actions.ndim != 2 or actions.shape != (frames.shape[0] - 1, 2):
        raise ValueError("history needs one declared two-dimensional action per RGB transition")
    return frames, actions


def resize_rgb(frames: np.ndarray, image_size: int) -> torch.Tensor:
    if int(image_size) < 4:
        raise ValueError("image_size must be at least four")
    tensor = torch.as_tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return F.interpolate(tensor, size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False)


FEATURES = {"two_frame_raw": root_feature_two_frame, "temporal_history": root_feature_temporal_history}


def select_balanced(rows: Iterable[RootExample], *, per_class: int, split: str, seed: int) -> list[RootExample]:
    """Freeze a root-level balanced cohort before any fitting begins."""

    if int(per_class) < 2:
        raise ValueError("balanced confirmation needs at least two roots per label class")
    candidates = list(rows)
    rng = np.random.default_rng(int(seed))
    by_label = {label: [row for row in candidates if int(row.preference) == label] for label in (0, 1)}
    shortages = [(len(by_label[label]), label) for label in (0, 1) if len(by_label[label]) < int(per_class)]
    if shortages:
        # Report the rarer class.  On an empty/all-one-class panel the positive
        # class wins the tie so the failure explicitly states the missing
        # counterfactual rather than whichever class happened to be iterated.
        count, label = min(shortages, key=lambda item: (item[0], -item[1]))
        raise RuntimeError(
            f"{split} seed band has {count} eligible roots for label {label}; needs {per_class}. "
            "Do not train a decoder or reveal a test score from this incomplete balanced panel."
        )
    selected: list[RootExample] = []
    for label in (0, 1):
        class_rows = by_label[label]
        order = rng.permutation(len(class_rows))[: int(per_class)]
        selected.extend(class_rows[int(index)] for index in order)
    selected.sort(key=lambda row: row.root_id)
    return selected


def assert_root_separation(*splits: Iterable[RootExample]) -> None:
    """Fail closed on root identity or source-state reuse across any split."""

    seen_ids: set[int] = set()
    seen_digests: set[str] = set()
    for split in splits:
        current = list(split)
        ids = {int(row.root_id) for row in current}
        digests = {str(row.root_digest) for row in current}
        if len(ids) != len(current) or len(digests) != len(current):
            raise ValueError("a split contains duplicated root identities or source states")
        if seen_ids.intersection(ids) or seen_digests.intersection(digests):
            raise ValueError("root-disjoint train/calibration/test split violated")
        seen_ids.update(ids)
        seen_digests.update(digests)


def collect_band(
    env: gym.Env,
    *,
    split: str,
    band_seed: int,
    per_class: int,
    pool_transitions: int,
    history_steps: int,
    history_magnitude: float,
    max_contact_gap: float,
    minimum_margin: float,
    preference_fn: Callable[..., tuple[int | None, float, float, float]] = offline_preference,
    probe_action_fn: ProbeActionFn = global_y_probe_actions,
) -> tuple[list[RootExample], dict[str, Any]]:
    """Collect then label all candidate roots before stratified cohort selection."""

    pool = collect_trajectory_data(env, int(pool_transitions), int(band_seed))
    candidates = [
        np.asarray(state, dtype=np.float64).copy()
        for state in np.asarray(pool["states"], dtype=np.float64)
        if mode_for_state(state) == "contact"
        and float(np.linalg.norm(np.asarray(state)[2:4] - np.asarray(state)[:2])) <= float(max_contact_gap)
    ]
    rows: list[RootExample] = []
    left_contact = 0
    ineligible_margin = 0
    invalid_probe_geometry = 0
    probe_families: set[str] = set()
    valid_probe_pairs = 0
    for source_index, source_state in enumerate(candidates):
        root_id = int(band_seed) + int(source_index)
        goal = _goal_for_root(source_state, root_id)
        history = fixed_history_actions(root_id, int(history_steps), float(history_magnitude))
        env.reset(seed=root_id, options={"state": source_state.copy(), "goal_state": goal.copy()})
        frames = [np.asarray(env.render(), dtype=np.uint8).copy()]
        current_state = source_state.copy()
        for action in history:
            current_state, frame = advance(env, action, 1)
            frames.append(frame)
        if mode_for_state(current_state) != "contact":
            left_contact += 1
            continue
        actions, probe_audit = probe_action_fn(env, current_state)
        family = str(probe_audit.get("family", "unspecified"))
        probe_families.add(family)
        if not bool(probe_audit.get("pair_valid", False)):
            invalid_probe_geometry += 1
            continue
        actions = validate_probe_actions(actions)
        valid_probe_pairs += 1
        before_angle = float(current_state[4])
        probe_seed = matched_probe_reset_seed(int(band_seed), int(source_index))
        probe_results: dict[str, np.ndarray] = {}
        for name in ("zero", "plus", "minus"):
            # Each arm starts from the same state, goal, and reset RNG state.
            # The action is the sole counterfactual difference.
            env.reset(seed=probe_seed, options={"state": current_state.copy(), "goal_state": goal.copy()})
            probe_results[name], _ = advance(env, actions[name], TORQUE_DURATION)
        preference, margin, plus_response, minus_response = preference_fn(
            before_angle=before_angle,
            zero_angle=float(probe_results["zero"][4]),
            plus_angle=float(probe_results["plus"][4]),
            minus_angle=float(probe_results["minus"][4]),
            minimum_margin=float(minimum_margin),
        )
        if preference is None:
            ineligible_margin += 1
            continue
        rows.append(RootExample(
            root_id=root_id,
            root_digest=state_digest(source_state),
            rgb_frames=np.stack(frames).astype(np.uint8),
            history_actions=history,
            preference=int(preference),
            preference_margin=float(margin),
            zero_corrected_plus_response=float(plus_response),
            zero_corrected_minus_response=float(minus_response),
        ))
    raw_counts = {str(label): int(sum(row.preference == label for row in rows)) for label in (0, 1)}
    collection = {
        "split": split,
        "seed_band": int(band_seed),
        "pool_transitions": int(pool_transitions),
        "contact_candidates": int(len(candidates)),
        "history_left_contact": int(left_contact),
        "probe_action_families": sorted(probe_families),
        "valid_probe_pairs": int(valid_probe_pairs),
        "invalid_probe_geometry": int(invalid_probe_geometry),
        "ineligible_margin": int(ineligible_margin),
        "eligible_by_label_before_sampling": raw_counts,
        "selected_per_class": int(per_class),
        "counterfactual_reset_contract": "zero/plus/minus share one reset seed per source root",
    }
    try:
        selected = select_balanced(rows, per_class=int(per_class), split=split, seed=int(band_seed + 700_000))
    except RuntimeError as error:
        raise BalancedCohortUnavailable(str(error), collection=collection) from error
    collection["selected_roots"] = int(len(selected))
    return selected, collection


def feature_matrix(rows: Iterable[RootExample], representation: str, image_size: int) -> np.ndarray:
    if representation not in FEATURES:
        raise ValueError(f"unknown representation {representation}")
    matrix = np.stack([
        FEATURES[representation](row.rgb_frames, row.history_actions, int(image_size)) for row in rows
    ]).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("RGB/action feature matrix is not finite")
    return matrix


def labels(rows: Iterable[RootExample]) -> np.ndarray:
    return np.asarray([int(row.preference) for row in rows], dtype=np.int64)


def fit_linear_readout(
    train_x: np.ndarray, train_y: np.ndarray, evaluation_x: np.ndarray, *, device: str, epochs: int,
    learning_rate: float, weight_decay: float, seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train-only standardized linear readout shared by both representations."""

    if len(np.unique(train_y)) != 2:
        raise ValueError("train cohort must include both predeclared labels")
    torch.manual_seed(int(seed))
    mean = train_x.mean(axis=0, keepdims=True)
    scale = np.maximum(train_x.std(axis=0, keepdims=True), 1.0e-5)
    x_train = torch.as_tensor((train_x - mean) / scale, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    x_evaluation = torch.as_tensor((evaluation_x - mean) / scale, dtype=torch.float32, device=device)
    model = torch.nn.Linear(x_train.shape[1], 1, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        logits = model(x_train).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        train_loss = float(F.binary_cross_entropy_with_logits(model(x_train).reshape(-1), y_train).item())
        probability = torch.sigmoid(model(x_evaluation).reshape(-1)).detach().cpu().numpy()
    return probability.astype(np.float64), {"feature_dimension": int(x_train.shape[1]), "train_loss": train_loss}


def balanced_accuracy(probability: np.ndarray, target: np.ndarray) -> float:
    prediction = (np.asarray(probability, dtype=np.float64) >= 0.5).astype(np.int64)
    target = np.asarray(target, dtype=np.int64)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("balanced accuracy requires both held-out preference classes")
    return float((np.mean(prediction[target == 0] == 0) + np.mean(prediction[target == 1] == 1)) / 2.0)


def bootstrap_balanced_accuracy(
    probability: np.ndarray, target: np.ndarray, *, draws: int, seed: int, device: str
) -> dict[str, Any]:
    """Root-level stratified bootstrap CI for a frozen threshold of 0.5."""

    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    positive = np.flatnonzero(target == 1)
    negative = np.flatnonzero(target == 0)
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("bootstrap confirmation requires at least two roots per test class")
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(int(seed))
    correct = torch.as_tensor(((probability >= 0.5).astype(np.int64) == target).astype(np.float64), device=torch_device)
    pos = torch.as_tensor(positive, dtype=torch.long, device=torch_device)
    neg = torch.as_tensor(negative, dtype=torch.long, device=torch_device)
    pos_draws = pos[torch.randint(len(pos), (int(draws), len(pos)), generator=generator, device=torch_device)]
    neg_draws = neg[torch.randint(len(neg), (int(draws), len(neg)), generator=generator, device=torch_device)]
    samples = (correct[pos_draws].mean(dim=1) + correct[neg_draws].mean(dim=1)) / 2.0
    result = {
        "estimate": balanced_accuracy(probability, target),
        "ci95_low": float(torch.quantile(samples, 0.025).item()),
        "ci95_high": float(torch.quantile(samples, 0.975).item()),
        "draws": int(draws),
        "unit": "root-level, stratified by fixed offline preference class",
    }
    del correct, pos, neg, pos_draws, neg_draws, samples
    torch.cuda.synchronize(torch_device)
    return result


def paired_bootstrap_delta(
    first_probability: np.ndarray, second_probability: np.ndarray, target: np.ndarray, *, draws: int, seed: int,
    device: str,
) -> dict[str, Any]:
    """Root-paired temporal-minus-raw balanced-accuracy CI."""

    first_probability = np.asarray(first_probability, dtype=np.float64)
    second_probability = np.asarray(second_probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    if first_probability.shape != second_probability.shape or first_probability.shape != target.shape:
        raise ValueError("paired comparison needs aligned root-level predictions")
    positive = np.flatnonzero(target == 1)
    negative = np.flatnonzero(target == 0)
    torch_device = torch.device(device)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(int(seed))
    first_correct = torch.as_tensor(((first_probability >= 0.5).astype(np.int64) == target).astype(np.float64), device=torch_device)
    second_correct = torch.as_tensor(((second_probability >= 0.5).astype(np.int64) == target).astype(np.float64), device=torch_device)
    pos = torch.as_tensor(positive, dtype=torch.long, device=torch_device)
    neg = torch.as_tensor(negative, dtype=torch.long, device=torch_device)
    pos_draws = pos[torch.randint(len(pos), (int(draws), len(pos)), generator=generator, device=torch_device)]
    neg_draws = neg[torch.randint(len(neg), (int(draws), len(neg)), generator=generator, device=torch_device)]
    temporal = (second_correct[pos_draws].mean(1) + second_correct[neg_draws].mean(1)) / 2.0
    raw = (first_correct[pos_draws].mean(1) + first_correct[neg_draws].mean(1)) / 2.0
    delta = temporal - raw
    result = {
        "estimate": balanced_accuracy(second_probability, target) - balanced_accuracy(first_probability, target),
        "ci95_low": float(torch.quantile(delta, 0.025).item()),
        "ci95_high": float(torch.quantile(delta, 0.975).item()),
        "direction": "temporal_history minus two_frame_raw",
        "draws": int(draws),
    }
    del first_correct, second_correct, pos, neg, pos_draws, neg_draws, temporal, raw, delta
    torch.cuda.synchronize(torch_device)
    return result


def evaluate(
    train: list[RootExample], calibration: list[RootExample], test: list[RootExample], *, representation: str,
    image_size: int, device: str, epochs: int, learning_rate: float, weight_decay: float, bootstrap: int, seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit once on train roots only and evaluate fixed calibration/test cohorts."""

    train_x = feature_matrix(train, representation, image_size)
    train_y = labels(train)
    calibration_x = feature_matrix(calibration, representation, image_size)
    test_x = feature_matrix(test, representation, image_size)
    calibration_probability, fit = fit_linear_readout(
        train_x, train_y, calibration_x, device=device, epochs=epochs, learning_rate=learning_rate,
        weight_decay=weight_decay, seed=seed,
    )
    # Refit is deliberately forbidden: this second deterministic fit uses the
    # same train roots and seed to obtain held predictions, never calibration.
    test_probability, _ = fit_linear_readout(
        train_x, train_y, test_x, device=device, epochs=epochs, learning_rate=learning_rate,
        weight_decay=weight_decay, seed=seed,
    )
    calibration_y = labels(calibration)
    test_y = labels(test)
    return {
        "representation": representation,
        "fit": fit,
        "calibration_balanced_accuracy": balanced_accuracy(calibration_probability, calibration_y),
        "test_balanced_accuracy": bootstrap_balanced_accuracy(
            test_probability, test_y, draws=bootstrap, seed=seed + 11, device=device
        ),
    }, test_probability


def confirmation_rule(temporal: dict[str, Any], raw: dict[str, Any], delta: dict[str, Any], *, minimum_ba: float,
                      minimum_ci_lower: float) -> dict[str, Any]:
    """Apply the predeclared confirmation rule without inspecting results first."""

    metric = temporal["test_balanced_accuracy"]
    passed = bool(
        float(metric["estimate"]) >= float(minimum_ba)
        and float(metric["ci95_low"]) >= float(minimum_ci_lower)
        and float(delta["ci95_low"]) > 0.0
    )
    return {
        "status": "confirmed" if passed else "not_confirmed",
        "predeclared_rule": {
            "temporal_test_balanced_accuracy_at_least": float(minimum_ba),
            "temporal_test_ba_ci95_lower_at_least": float(minimum_ci_lower),
            "paired_temporal_minus_raw_ba_ci95_lower_strictly_greater_than": 0.0,
        },
        "observed": {
            "temporal_test_balanced_accuracy": float(metric["estimate"]),
            "temporal_ci95_low": float(metric["ci95_low"]),
            "two_frame_test_balanced_accuracy": float(raw["test_balanced_accuracy"]["estimate"]),
            "paired_temporal_minus_raw": float(delta["estimate"]),
            "paired_delta_ci95_low": float(delta["ci95_low"]),
        },
        "interpretation": (
            "The fixed multi-frame RGB/action representation passed this narrow offline confirmation rule. "
            "This is still not an action-selection, planning, world-model, or pose-closure result."
            if passed else
            "The narrow offline multi-frame mechanism was not confirmed under the frozen rule; do not tune it on this test panel."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--train-seed-band", type=int, default=DEFAULT_SEED_BANDS["train"])
    parser.add_argument("--calibration-seed-band", type=int, default=DEFAULT_SEED_BANDS["calibration"])
    parser.add_argument("--test-seed-band", type=int, default=DEFAULT_SEED_BANDS["test"])
    parser.add_argument("--train-per-class", type=int, default=24)
    parser.add_argument("--calibration-per-class", type=int, default=12)
    parser.add_argument("--test-per-class", type=int, default=24)
    parser.add_argument("--pool-transitions", type=int, default=8192)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--history-magnitude", type=float, default=0.30)
    parser.add_argument("--max-contact-gap", type=float, default=65.0)
    parser.add_argument("--minimum-preference-margin", type=float, default=2.0e-4)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-ci-lower", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.history_steps < 2 or args.pool_transitions < 100:
        raise ValueError("confirmation requires multi-frame history and a substantial fixed root pool")
    if not (0.0 < args.history_magnitude <= 1.0 and 0.0 < args.minimum_preference_margin):
        raise ValueError("history magnitude and label margin must be positive")
    if not (0.5 < args.minimum_balanced_accuracy <= 1.0 and 0.5 <= args.minimum_ci_lower <= 1.0):
        raise ValueError("confirmation thresholds must lie above chance and no more than one")
    bands = {"train": args.train_seed_band, "calibration": args.calibration_seed_band, "test": args.test_seed_band}
    validate_seed_bands(bands)
    runtime = require_cuda_allocation(args.device)
    started = time.perf_counter()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=128)
    try:
        train, train_info = collect_band(
            env, split="train", band_seed=int(args.train_seed_band), per_class=int(args.train_per_class),
            pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps),
            history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap),
            minimum_margin=float(args.minimum_preference_margin),
        )
        calibration, calibration_info = collect_band(
            env, split="calibration", band_seed=int(args.calibration_seed_band), per_class=int(args.calibration_per_class),
            pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps),
            history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap),
            minimum_margin=float(args.minimum_preference_margin),
        )
        test, test_info = collect_band(
            env, split="test", band_seed=int(args.test_seed_band), per_class=int(args.test_per_class),
            pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps),
            history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap),
            minimum_margin=float(args.minimum_preference_margin),
        )
    finally:
        env.close()
    assert_root_separation(train, calibration, test)
    # All label cohorts, including the balanced held set, now exist.  The first
    # decoder is only fitted after this point.
    two_frame, two_frame_test_probability = evaluate(
        train, calibration, test, representation="two_frame_raw", image_size=int(args.image_size), device=args.device,
        epochs=int(args.epochs), learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay),
        bootstrap=int(args.bootstrap), seed=int(args.train_seed_band + 1),
    )
    temporal, temporal_test_probability = evaluate(
        train, calibration, test, representation="temporal_history", image_size=int(args.image_size), device=args.device,
        epochs=int(args.epochs), learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay),
        bootstrap=int(args.bootstrap), seed=int(args.train_seed_band + 2),
    )
    delta = paired_bootstrap_delta(
        two_frame_test_probability, temporal_test_probability, labels(test), draws=int(args.bootstrap),
        seed=int(args.train_seed_band + 999), device=args.device,
    )
    decision = confirmation_rule(
        temporal, two_frame, delta, minimum_ba=float(args.minimum_balanced_accuracy),
        minimum_ci_lower=float(args.minimum_ci_lower),
    )
    report = {
        "schema": SCHEMA,
        "runtime": runtime,
        "protocol": {
            "environment": "swm/PushT-v1",
            "mechanism_predeclared_from": "privileged torque-contrast sensitivity screen only",
            "mirrored_pulse": {"plus_y": [0.0, TORQUE_MAGNITUDE], "minus_y": [0.0, -TORQUE_MAGNITUDE], "duration_steps": TORQUE_DURATION},
            "runtime_inputs": "rendered RGB history and declared history actions only",
            "offline_only": "source state reset, zero/+y/-y successors, and angle-based preference label",
            "split_protocol": "three disjoint high root-seed bands; all labels sampled/stratified before any readout fit",
            "test_balance": {"class_0": int(args.test_per_class), "class_1": int(args.test_per_class)},
            "feature_comparison": ["two_frame_raw", "temporal_history"],
            "decoder": "same train-root-only standardized linear binary readout, fixed threshold 0.5",
        },
        "seed_bands": bands,
        "collection": {"train": train_info, "calibration": calibration_info, "test": test_info},
        "sanity_checks": {
            "root_disjoint": True,
            "source_state_digest_disjoint": True,
            "all_labels_collected_before_fitting": True,
            "held_test_is_balanced_before_training": True,
            "same_examples_same_labels_for_representations": True,
            "same_fixed_torque_mechanism": True,
            "no_runtime_action_selection": True,
        },
        "results": {"two_frame_raw": two_frame, "temporal_history": temporal, "paired_test_delta": delta},
        "confirmation": decision,
        "claim_boundary": (
            "This is an offline representation confirmation on a fixed privileged torque-preference label. It does not "
            "execute a readout-selected action and provides no claim about planning, pose closure, a learned world model, "
            "causal dynamics, or deployment action ranking."
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "balanced_temporal_torque_confirmation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# PushT independent balanced temporal torque confirmation", "",
        "The held cohort was root-disjoint and sampled to contain equal offline `+y` and `-y` preference classes before any decoder fit.", "",
        "| representation | calibration BA | test BA | test 95% CI |", "|---|---:|---:|---:|",
    ]
    for name, value in (("two_frame_raw", two_frame), ("temporal_history", temporal)):
        metric = value["test_balanced_accuracy"]
        lines.append(f"| {name} | {value['calibration_balanced_accuracy']:.3f} | {metric['estimate']:.3f} | [{metric['ci95_low']:.3f}, {metric['ci95_high']:.3f}] |")
    lines += [
        "", f"Paired temporal - two-frame test BA: `{delta['estimate']:.3f}`; 95% CI `[{delta['ci95_low']:.3f}, {delta['ci95_high']:.3f}]`.",
        f"", f"Confirmation status: **{decision['status']}**.", "", f"Claim boundary: {report['claim_boundary']}", "",
    ]
    (args.output / "balanced_temporal_torque_confirmation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "confirmation": decision, "wall_seconds": report["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
