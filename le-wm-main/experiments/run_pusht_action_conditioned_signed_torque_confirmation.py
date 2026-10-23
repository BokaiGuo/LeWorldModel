"""Fresh action-conditioned RGB confirmation for PushT signed torque response.

The preceding contact-frame confirmation supplied RGB history and historical
actions to its readout, but it omitted the *candidate pair* whose signed
effect formed the offline label.  That is not the deployment interface: a
visual controller must rank ``p(effect | RGB history, candidate action)``.

This protocol repairs that interface on three fresh, root-disjoint seed bands.
The runtime descriptors are all RGB/action derived:

* ``geometry_history``: RGB mask moments, an RGB contact-interface chart, and
  the declared action history;
* ``geometry_history_plus_candidate_pair``: the same descriptor plus the
  currently compared positive- and negative-moment action vectors.

The two candidates are still constructed only to create a fixed, legal local
pair; their real successors are used only offline for the signed label.  A
positive outcome establishes selective offline two-action identifiability, not
online planning, pose closure, causal world modelling, or general transfer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 - registers swm/PushT-v1
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.gpu_preflight import require_cuda_allocation  # noqa: E402
from experiments.run_causal_visual_pusht_world_model import MaskMomentGeometryEncoder  # noqa: E402
from experiments.run_pusht_balanced_temporal_torque_confirmation import (  # noqa: E402
    BalancedCohortUnavailable,
    TORQUE_DURATION,
    _goal_for_root,
    advance,
    bootstrap_balanced_accuracy,
    collect_trajectory_data,
    fixed_history_actions,
    mode_for_state,
    select_balanced,
    signed_offline_preference,
    state_digest,
    validate_probe_actions,
    validate_seed_bands,
)
from experiments.run_pusht_signed_temporal_torque_confirmation import contact_frame_probe_actions  # noqa: E402


SCHEMA = "pusht.action_conditioned_signed_torque_confirmation.v1"
SEED_BANDS = {"train": 1_704_210_000, "calibration": 1_704_220_000, "test": 1_704_230_000}


@dataclass(frozen=True)
class ActionConditionedExample:
    """A root-level RGB/action context and one offline signed-pair label."""

    root_id: int
    root_digest: str
    rgb_frames: np.ndarray
    history_actions: np.ndarray
    plus_action: np.ndarray
    minus_action: np.ndarray
    preference: int
    preference_margin: float
    zero_corrected_plus_response: float
    zero_corrected_minus_response: float


def assert_root_separation(*splits: Iterable[ActionConditionedExample]) -> None:
    """Reject source-root reuse across fit, calibration, and outcome cohorts."""

    ids: set[int] = set()
    digests: set[str] = set()
    for split in splits:
        rows = list(split)
        split_ids = {int(row.root_id) for row in rows}
        split_digests = {str(row.root_digest) for row in rows}
        if len(split_ids) != len(rows) or len(split_digests) != len(rows):
            raise ValueError("a split contains duplicate source roots")
        if ids.intersection(split_ids) or digests.intersection(split_digests):
            raise ValueError("root-disjoint split contract violated")
        ids.update(split_ids)
        digests.update(split_digests)


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
    inward_magnitude: float,
    tangent_magnitude: float,
) -> tuple[list[ActionConditionedExample], dict[str, Any]]:
    """Collect all labels first, then draw a balanced root cohort once."""

    pool = collect_trajectory_data(env, int(pool_transitions), int(band_seed))
    candidates = [
        np.asarray(state, dtype=np.float64).copy()
        for state in np.asarray(pool["states"], dtype=np.float64)
        if mode_for_state(state) == "contact"
        and float(np.linalg.norm(np.asarray(state)[2:4] - np.asarray(state)[:2])) <= float(max_contact_gap)
    ]
    rows: list[ActionConditionedExample] = []
    left_contact = 0
    invalid_probe_geometry = 0
    ineligible_margin = 0
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
        actions, audit = contact_frame_probe_actions(
            env,
            current_state,
            inward_magnitude=float(inward_magnitude),
            tangent_magnitude=float(tangent_magnitude),
            max_contact_gap=float(max_contact_gap),
        )
        if not bool(audit.get("pair_valid", False)):
            invalid_probe_geometry += 1
            continue
        actions = validate_probe_actions(actions)
        probe_seed = int(band_seed) + 50_000 + int(source_index)
        successors: dict[str, np.ndarray] = {}
        for name in ("zero", "plus", "minus"):
            env.reset(seed=probe_seed, options={"state": current_state.copy(), "goal_state": goal.copy()})
            successors[name], _ = advance(env, actions[name], TORQUE_DURATION)
        preference, margin, plus_response, minus_response = signed_offline_preference(
            before_angle=float(current_state[4]),
            zero_angle=float(successors["zero"][4]),
            plus_angle=float(successors["plus"][4]),
            minus_angle=float(successors["minus"][4]),
            minimum_margin=float(minimum_margin),
        )
        if preference is None:
            ineligible_margin += 1
            continue
        rows.append(
            ActionConditionedExample(
                root_id=root_id,
                root_digest=state_digest(source_state),
                rgb_frames=np.stack(frames).astype(np.uint8),
                history_actions=history.astype(np.float32),
                plus_action=np.asarray(actions["plus"], dtype=np.float32),
                minus_action=np.asarray(actions["minus"], dtype=np.float32),
                preference=int(preference),
                preference_margin=float(margin),
                zero_corrected_plus_response=float(plus_response),
                zero_corrected_minus_response=float(minus_response),
            )
        )
    collection = {
        "split": str(split),
        "seed_band": int(band_seed),
        "pool_transitions": int(pool_transitions),
        "contact_candidates": int(len(candidates)),
        "history_left_contact": int(left_contact),
        "invalid_probe_geometry": int(invalid_probe_geometry),
        "ineligible_margin": int(ineligible_margin),
        "eligible_by_label_before_sampling": {
            str(label): int(sum(int(row.preference) == label for row in rows)) for label in (0, 1)
        },
        "selected_per_class": int(per_class),
        "counterfactual_reset_contract": "zero/plus/minus share one reset seed per source root",
        "candidate_interface": "positive- and negative-moment vectors are runtime readout inputs",
    }
    try:
        selected = select_balanced(rows, per_class=int(per_class), split=str(split), seed=int(band_seed + 700_000))
    except RuntimeError as error:
        raise BalancedCohortUnavailable(str(error), collection=collection) from error
    collection["selected_roots"] = int(len(selected))
    return selected, collection


@torch.inference_mode()
def geometry_history_features(rows: Iterable[ActionConditionedExample], *, device: str) -> np.ndarray:
    """Extract a fixed RGB-only pose/interface history chart on CUDA."""

    items = list(rows)
    if not items:
        raise ValueError("geometry chart needs at least one root")
    frames = np.stack([row.rgb_frames for row in items])
    if frames.ndim != 5 or frames.shape[-1] != 3:
        raise ValueError("RGB histories must be [root, time, height, width, channel]")
    root_count, time_count = frames.shape[:2]
    image = torch.as_tensor(frames, dtype=torch.float32, device=device).reshape(-1, *frames.shape[2:])
    image = image.permute(0, 3, 1, 2) / 255.0
    masks = MaskMomentGeometryEncoder._masks(image)
    moment = torch.cat([MaskMomentGeometryEncoder._moments(mask) for mask in masks], dim=-1)
    interface = MaskMomentGeometryEncoder._interface_features(masks[0], masks[1])
    chart = torch.cat((moment, interface), dim=-1).reshape(root_count, time_count, -1)
    values = torch.cat((chart[:, -1], (chart[:, 1:] - chart[:, :-1]).flatten(1)), dim=1)
    history = torch.as_tensor(np.stack([row.history_actions for row in items]), dtype=torch.float32, device=device).flatten(1)
    values = torch.cat((values, history), dim=1)
    result = values.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("RGB geometry history produced non-finite values")
    return result


def candidate_pair_features(rows: Iterable[ActionConditionedExample]) -> np.ndarray:
    """Return declared, executable plus/minus candidate vectors only."""

    values = np.asarray(
        [np.concatenate((row.plus_action, row.minus_action)).astype(np.float32) for row in rows], dtype=np.float32
    )
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("candidate-pair feature contract requires finite [plus_x, plus_y, minus_x, minus_y]")
    return values


def contact_action_mechanics_features(
    geometry: np.ndarray, rows: Iterable[ActionConditionedExample]
) -> np.ndarray:
    """Build fixed RGB-contact-vector/action cross terms.

    A linear readout over concatenated geometry and action cannot represent a
    torque: the minimal contact mechanism is ``r_x a_y - r_y a_x``.  The two
    lever arms below are both derived from the final RGB interface chart: the
    pusher-to-block-centroid vector and the pusher-to-nearest-block-pixel
    contact vector.  No simulator coordinate, contact label, or successor is
    used at runtime.
    """

    geometry = np.asarray(geometry, dtype=np.float32)
    pairs = candidate_pair_features(rows)
    if geometry.ndim != 2 or len(geometry) != len(pairs) or geometry.shape[1] < 41:
        raise ValueError("mechanics features require aligned geometry histories with a final 41-D RGB chart")
    # geometry_history_features starts with the final [30 moment | 11
    # interface] chart.  Interface layout is [relative(2), distance,
    # contact_vector(2), ...].
    relative = geometry[:, 30:32]
    contact = geometry[:, 33:35]
    plus, minus = pairs[:, :2], pairs[:, 2:]

    def cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0])[:, None]

    def dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.sum(left * right, axis=1, keepdims=True)

    values = np.concatenate(
        (
            cross(relative, plus), dot(relative, plus),
            cross(relative, minus), dot(relative, minus),
            cross(contact, plus), dot(contact, plus),
            cross(contact, minus), dot(contact, minus),
        ),
        axis=1,
    ).astype(np.float32)
    if values.shape != (len(geometry), 8) or not np.isfinite(values).all():
        raise ValueError("RGB contact/action mechanics chart is malformed")
    return values


def body_frame_mechanics_features(
    geometry: np.ndarray, rows: Iterable[ActionConditionedExample]
) -> np.ndarray:
    """Express RGB contact geometry and candidate actions in the T's body frame.

    The moment chart exposes the block's second- and third-order central
    moments.  They define a directed body axis from RGB alone.  Projecting the
    lever arms and candidate forces into that chart supplies the missing
    contact-location-by-orientation interaction to a linear readout.
    """

    geometry = np.asarray(geometry, dtype=np.float32)
    pairs = candidate_pair_features(rows)
    if geometry.ndim != 2 or len(geometry) != len(pairs) or geometry.shape[1] < 41:
        raise ValueError("body-frame features require aligned geometry histories with a final 41-D RGB chart")
    # First 30 entries are agent/block/goal 10-D moment blocks.  In the block
    # block: [mass, mean_x, mean_y, mu20, mu11, mu02, mu30, mu21, mu12, mu03].
    block = geometry[:, 10:20]
    axis_angle = 0.5 * np.arctan2(2.0 * block[:, 4], block[:, 3] - block[:, 5])
    axis_x, axis_y = np.cos(axis_angle), np.sin(axis_angle)
    projected_third = (
        block[:, 6] * axis_x ** 3
        + 3.0 * block[:, 7] * axis_x ** 2 * axis_y
        + 3.0 * block[:, 8] * axis_x * axis_y ** 2
        + block[:, 9] * axis_y ** 3
    )
    axis_angle = axis_angle + np.where(projected_third < 0.0, np.pi, 0.0)
    axis = np.stack((np.cos(axis_angle), np.sin(axis_angle)), axis=1).astype(np.float32)
    perpendicular = np.stack((-axis[:, 1], axis[:, 0]), axis=1).astype(np.float32)
    relative = geometry[:, 30:32]
    contact = geometry[:, 33:35]
    plus, minus = pairs[:, :2], pairs[:, 2:]

    def components(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.sum(vector * axis, axis=1, keepdims=True), np.sum(vector * perpendicular, axis=1, keepdims=True)

    pieces = []
    for vector in (relative, contact, plus, minus):
        pieces.extend(components(vector))
    values = np.concatenate(pieces, axis=1).astype(np.float32)
    if values.shape != (len(geometry), 8) or not np.isfinite(values).all():
        raise ValueError("RGB body-frame mechanics chart is malformed")
    return values


def fit_linear_readout(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluation_x: np.ndarray,
    *,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one train-only standardized, regularized action-effect readout."""

    if len(np.unique(train_y)) != 2:
        raise ValueError("both signed preference classes are required in the fit split")
    torch.manual_seed(int(seed))
    mean = train_x.mean(axis=0, keepdims=True)
    scale = np.maximum(train_x.std(axis=0, keepdims=True), 1.0e-5)
    x_train = torch.as_tensor((train_x - mean) / scale, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    x_evaluation = torch.as_tensor((evaluation_x - mean) / scale, dtype=torch.float32, device=device)
    model = torch.nn.Linear(int(x_train.shape[1]), 1, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        loss = F.binary_cross_entropy_with_logits(model(x_train).reshape(-1), y_train)
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
    return float((np.mean(prediction[target == 0] == 0) + np.mean(prediction[target == 1] == 1)) / 2.0)


def paired_delta(
    baseline_probability: np.ndarray,
    conditioned_probability: np.ndarray,
    target: np.ndarray,
    *, draws: int, seed: int, device: str, direction: str,
) -> dict[str, Any]:
    """Root-paired bootstrap for action conditioning minus geometry-only."""

    target = np.asarray(target, dtype=np.int64)
    first = np.asarray(baseline_probability, dtype=np.float64)
    second = np.asarray(conditioned_probability, dtype=np.float64)
    if first.shape != second.shape or first.shape != target.shape:
        raise ValueError("paired delta requires aligned root predictions")
    indices_by_label = [np.flatnonzero(target == label) for label in (0, 1)]
    if min(len(indices) for indices in indices_by_label) < 2:
        raise ValueError("paired bootstrap needs two roots per label")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    baseline = torch.as_tensor(((first >= 0.5).astype(np.int64) == target).astype(np.float32), device=device)
    conditioned = torch.as_tensor(((second >= 0.5).astype(np.int64) == target).astype(np.float32), device=device)
    samples = []
    for indices in indices_by_label:
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        sampled = index[torch.randint(len(index), (int(draws), len(index)), generator=generator, device=device)]
        samples.append((conditioned[sampled] - baseline[sampled]).mean(dim=1))
    delta = (samples[0] + samples[1]) / 2.0
    return {
        "estimate": balanced_accuracy(second, target) - balanced_accuracy(first, target),
        "ci95_low": float(torch.quantile(delta, 0.025).item()),
        "ci95_high": float(torch.quantile(delta, 0.975).item()),
        "direction": str(direction),
        "draws": int(draws),
    }


def evaluate(
    train: list[ActionConditionedExample],
    calibration: list[ActionConditionedExample],
    test: list[ActionConditionedExample],
    *,
    representation: str,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    bootstrap: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit once, then score calibration and outcome roots with one frozen model."""

    train_geometry = geometry_history_features(train, device=device)
    calibration_geometry = geometry_history_features(calibration, device=device)
    test_geometry = geometry_history_features(test, device=device)
    if representation == "geometry_history":
        train_x, calibration_x, test_x = train_geometry, calibration_geometry, test_geometry
    elif representation == "geometry_plus_candidate_pair":
        train_x = np.concatenate((train_geometry, candidate_pair_features(train)), axis=1)
        calibration_x = np.concatenate((calibration_geometry, candidate_pair_features(calibration)), axis=1)
        test_x = np.concatenate((test_geometry, candidate_pair_features(test)), axis=1)
    elif representation == "geometry_plus_pair_mechanics":
        train_x = np.concatenate(
            (train_geometry, candidate_pair_features(train), contact_action_mechanics_features(train_geometry, train)), axis=1
        )
        calibration_x = np.concatenate(
            (calibration_geometry, candidate_pair_features(calibration), contact_action_mechanics_features(calibration_geometry, calibration)), axis=1
        )
        test_x = np.concatenate(
            (test_geometry, candidate_pair_features(test), contact_action_mechanics_features(test_geometry, test)), axis=1
        )
    elif representation == "geometry_plus_rigid_body_mechanics":
        train_x = np.concatenate(
            (train_geometry, candidate_pair_features(train), contact_action_mechanics_features(train_geometry, train), body_frame_mechanics_features(train_geometry, train)), axis=1
        )
        calibration_x = np.concatenate(
            (calibration_geometry, candidate_pair_features(calibration), contact_action_mechanics_features(calibration_geometry, calibration), body_frame_mechanics_features(calibration_geometry, calibration)), axis=1
        )
        test_x = np.concatenate(
            (test_geometry, candidate_pair_features(test), contact_action_mechanics_features(test_geometry, test), body_frame_mechanics_features(test_geometry, test)), axis=1
        )
    else:
        raise ValueError(f"unknown representation: {representation}")
    train_y = np.asarray([row.preference for row in train], dtype=np.int64)
    calibration_y = np.asarray([row.preference for row in calibration], dtype=np.int64)
    test_y = np.asarray([row.preference for row in test], dtype=np.int64)
    combined_probability, fit = fit_linear_readout(
        train_x, train_y, np.concatenate((calibration_x, test_x), axis=0), device=device,
        epochs=int(epochs), learning_rate=float(learning_rate), weight_decay=float(weight_decay), seed=int(seed),
    )
    calibration_probability = combined_probability[: len(calibration)]
    test_probability = combined_probability[len(calibration):]
    return {
        "representation": representation,
        "fit": fit,
        "calibration_balanced_accuracy": balanced_accuracy(calibration_probability, calibration_y),
        "test_balanced_accuracy": bootstrap_balanced_accuracy(
            test_probability, test_y, draws=int(bootstrap), seed=int(seed + 17), device=device,
        ),
    }, test_probability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--train-seed-band", type=int, default=SEED_BANDS["train"])
    parser.add_argument("--calibration-seed-band", type=int, default=SEED_BANDS["calibration"])
    parser.add_argument("--test-seed-band", type=int, default=SEED_BANDS["test"])
    parser.add_argument("--train-per-class", type=int, default=128)
    parser.add_argument("--calibration-per-class", type=int, default=64)
    parser.add_argument("--test-per-class", type=int, default=128)
    parser.add_argument("--pool-transitions", type=int, default=8192)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--history-magnitude", type=float, default=0.30)
    parser.add_argument("--max-contact-gap", type=float, default=65.0)
    parser.add_argument("--minimum-preference-margin", type=float, default=2.0e-4)
    parser.add_argument("--contact-inward-magnitude", type=float, default=0.18)
    parser.add_argument("--contact-tangent-magnitude", type=float, default=0.55)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--minimum-balanced-accuracy", type=float, default=0.70)
    parser.add_argument("--minimum-ci-lower", type=float, default=0.60)
    parser.add_argument(
        "--confirmation-representation",
        choices=("geometry_plus_candidate_pair", "geometry_plus_pair_mechanics", "geometry_plus_rigid_body_mechanics"),
        default="geometry_plus_rigid_body_mechanics",
        help="predeclared conditioned arm used for the confirmation rule",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {args.output}")
    if args.history_steps < 2 or args.pool_transitions < 100:
        raise ValueError("confirmation needs temporal history and a substantial pool")
    bands = {"train": args.train_seed_band, "calibration": args.calibration_seed_band, "test": args.test_seed_band}
    validate_seed_bands(bands)
    runtime = require_cuda_allocation(args.device)
    started = time.perf_counter()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=128)
    collection: dict[str, Any] = {}
    try:
        train, collection["train"] = collect_band(env, split="train", band_seed=int(bands["train"]), per_class=int(args.train_per_class), pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps), history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap), minimum_margin=float(args.minimum_preference_margin), inward_magnitude=float(args.contact_inward_magnitude), tangent_magnitude=float(args.contact_tangent_magnitude))
        calibration, collection["calibration"] = collect_band(env, split="calibration", band_seed=int(bands["calibration"]), per_class=int(args.calibration_per_class), pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps), history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap), minimum_margin=float(args.minimum_preference_margin), inward_magnitude=float(args.contact_inward_magnitude), tangent_magnitude=float(args.contact_tangent_magnitude))
        test, collection["test"] = collect_band(env, split="test", band_seed=int(bands["test"]), per_class=int(args.test_per_class), pool_transitions=int(args.pool_transitions), history_steps=int(args.history_steps), history_magnitude=float(args.history_magnitude), max_contact_gap=float(args.max_contact_gap), minimum_margin=float(args.minimum_preference_margin), inward_magnitude=float(args.contact_inward_magnitude), tangent_magnitude=float(args.contact_tangent_magnitude))
    except BalancedCohortUnavailable as error:
        collection[str(error.collection.get("split", "unknown"))] = error.collection
        args.output.mkdir(parents=True)
        report = {"schema": SCHEMA, "runtime": runtime, "seed_bands": bands, "collection": collection, "status": "rejected_pre_outcome_balanced_cohort", "rejection_reason": str(error), "claim_boundary": "A cohort-availability rejection, not a decoder or planning result.", "wall_seconds": time.perf_counter() - started}
        (args.output / "action_conditioned_signed_torque_confirmation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
        return
    finally:
        env.close()
    assert_root_separation(train, calibration, test)
    geometry, geometry_probability = evaluate(train, calibration, test, representation="geometry_history", device=args.device, epochs=int(args.epochs), learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay), bootstrap=int(args.bootstrap), seed=int(args.train_seed_band + 1))
    pair, pair_probability = evaluate(train, calibration, test, representation="geometry_plus_candidate_pair", device=args.device, epochs=int(args.epochs), learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay), bootstrap=int(args.bootstrap), seed=int(args.train_seed_band + 2))
    mechanics, mechanics_probability = evaluate(train, calibration, test, representation="geometry_plus_pair_mechanics", device=args.device, epochs=int(args.epochs), learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay), bootstrap=int(args.bootstrap), seed=int(args.train_seed_band + 3))
    rigid_body, rigid_body_probability = evaluate(train, calibration, test, representation="geometry_plus_rigid_body_mechanics", device=args.device, epochs=int(args.epochs), learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay), bootstrap=int(args.bootstrap), seed=int(args.train_seed_band + 4))
    test_y = np.asarray([row.preference for row in test], dtype=np.int64)
    selected, selected_probability = {
        "geometry_plus_candidate_pair": (pair, pair_probability),
        "geometry_plus_pair_mechanics": (mechanics, mechanics_probability),
        "geometry_plus_rigid_body_mechanics": (rigid_body, rigid_body_probability),
    }[str(args.confirmation_representation)]
    delta = paired_delta(geometry_probability, selected_probability, test_y, draws=int(args.bootstrap), seed=int(args.train_seed_band + 991), device=args.device, direction=f"{args.confirmation_representation} minus geometry_history")
    metric = selected["test_balanced_accuracy"]
    confirmed = bool(float(metric["estimate"]) >= float(args.minimum_balanced_accuracy) and float(metric["ci95_low"]) >= float(args.minimum_ci_lower) and float(delta["ci95_low"]) > 0.0)
    report = {
        "schema": SCHEMA,
        "runtime": runtime,
        "seed_bands": bands,
        "protocol": {
            "label": "signed zero-drift-corrected response of matched positive- versus negative-moment pulses",
            "runtime_inputs": "rendered RGB history, declared history actions, and the current declared plus/minus candidate pair",
            "geometry_chart": "RGB mask moments plus RGB agent-block contact-interface features",
            "comparison": "geometry history alone, raw candidate conditioning, RGB lever-arm/action mechanics, and T body-frame action mechanics",
            "confirmation_representation": str(args.confirmation_representation),
            "selection": "all labels collected and each root cohort balanced before any readout fit",
        },
        "collection": collection,
        "results": {"geometry_history": geometry, "geometry_plus_candidate_pair": pair, "geometry_plus_pair_mechanics": mechanics, "geometry_plus_rigid_body_mechanics": rigid_body, "paired_action_conditioning_delta": delta},
        "confirmation": {
            "status": "confirmed_selective_action_conditioned_identifiability" if confirmed else "not_confirmed",
            "predeclared_rule": {"candidate_conditioned_test_balanced_accuracy_at_least": float(args.minimum_balanced_accuracy), "candidate_conditioned_test_ci95_lower_at_least": float(args.minimum_ci_lower), "paired_action_conditioning_delta_ci95_lower_strictly_greater_than": 0.0},
            "observed": {"geometry_test_balanced_accuracy": float(geometry["test_balanced_accuracy"]["estimate"]), "conditioned_test_balanced_accuracy": float(metric["estimate"]), "conditioned_test_ci95_low": float(metric["ci95_low"]), "paired_delta": float(delta["estimate"]), "paired_delta_ci95_low": float(delta["ci95_low"])},
        },
        "sanity_checks": {"root_disjoint": True, "source_state_digest_disjoint": True, "counterfactual_reset_seed_shared_across_zero_plus_minus": True, "candidate_pair_is_runtime_input": True, "all_labels_collected_before_fitting": True, "no_runtime_successor_input": True, "immutable_output": True},
        "claim_boundary": "An offline, two-action signed-effect identifiability test. It does not execute a selected action and does not establish planning, pose closure, causal dynamics, or a universal world model.",
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.mkdir(parents=True)
    (args.output / "action_conditioned_signed_torque_confirmation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["confirmation"]["status"], "output": str(args.output), "observed": report["confirmation"]["observed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
