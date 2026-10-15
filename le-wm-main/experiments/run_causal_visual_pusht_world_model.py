"""Train and evaluate an intervention-supervised visual PushT world model.

Unlike the frozen released checkpoint, this experiment constructs its training
rows from matched simulator counterfactuals: the same rendered source state is
followed by ``+a``, ``-a``, and zero actions.  The model consumes images and
actions only at policy time.  Simulator state is retained exclusively as an
offline supervision/evaluation target.

The runnable is deliberately self-contained so a positive result can be
audited as a *new trained visual world model*, rather than being conflated with
the earlier frozen-LeWM checkpoint diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401: registers PushT
import torch
from torch import nn
from torch.nn import functional as F


# ``swm/PushT-v1`` renders a 224x224 RGB frame.  This is part of the
# environment contract, not a tunable planner weight.  The earlier 256 scale
# silently shortened the visual contact distance and is retained only in old
# result artifacts for auditability.
PUSHT_RENDER_SIZE = 224.0


def jsonable(value: Any) -> Any:
    """Convert NumPy containers in audit traces before artifact serialization."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument(
        "--data-seed", type=int, default=None,
        help="root/action split seed; defaults to --seed, but formal checkpoint pools keep it fixed across model seeds",
    )
    parser.add_argument("--train-roots", type=int, default=1600)
    parser.add_argument("--val-roots", type=int, default=240)
    parser.add_argument("--root-rollout-steps", type=int, default=8)
    parser.add_argument(
        "--counterfactual-fan-size", type=int, default=0,
        help="additional goal-relative action directions per visual source beyond +a/-a/zero",
    )
    parser.add_argument(
        "--action-repeat", type=int, default=1,
        help="physical simulator steps represented by one world-model action and one MPC decision",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--snapshot-epochs", nargs="*", type=int, default=[],
        help="immutable checkpoint snapshots to save during training; must be in [1, epochs]",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument(
        "--vision-encoder", choices=("mask_moments", "color_geometry", "spatial_mlp", "conv"), default="mask_moments",
        help="mask_moments retains RGB-derived object geometry; other choices are visual ablations",
    )
    parser.add_argument(
        "--temporal-context", action="store_true",
        help="condition state/dynamics on previous and current rendered frames (mask_moments only)",
    )
    parser.add_argument(
        "--current-only-readout", action="store_true",
        help="when temporal context is enabled, decode absolute pose from the current frame only",
    )
    parser.add_argument(
        "--angle-readout-current-only", action="store_true",
        help="when explicit angle readout is enabled, exclude previous-frame differences from that angle head",
    )
    parser.add_argument(
        "--temporal-dynamics", action="store_true",
        help="supply RGB frame-to-frame centroid motion to action dynamics while keeping absolute pose readout current-frame-only",
    )
    parser.add_argument(
        "--rich-temporal-dynamics", action="store_true",
        help="also supply frame-to-frame block silhouette moment changes to action dynamics",
    )
    parser.add_argument(
        "--previous-action-dynamics", action="store_true",
        help="condition local contact dynamics and utility on the last executed RGB-policy action",
    )
    parser.add_argument(
        "--explicit-angle-effect", action="store_true",
        help="use a dedicated wrapped-angle action-effect head for terminal pose prediction",
    )
    parser.add_argument("--angle-effect-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--explicit-angle-readout", action="store_true",
        help="use a dedicated RGB block-angle readout and unit-circle supervision for current pose estimation",
    )
    parser.add_argument(
        "--invariant-angle-readout", action="store_true",
        help="feed the current angle head block-only central moments, removing position/agent nuisance coordinates",
    )
    parser.add_argument(
        "--contact-interface-dynamics", action="store_true",
        help="append an RGB-only pusher-to-block-boundary interface chart to action-effect dynamics",
    )
    parser.add_argument(
        "--contact-angle-readout-weight", type=float, default=1.0,
        help="extra unit-circle weight for source roots collected on the RGB contact manifold",
    )
    parser.add_argument(
        "--angle-readout-warmup-steps", type=int, default=0,
        help="pretrain the explicit RGB angle head before joint dynamics training",
    )
    parser.add_argument(
        "--freeze-angle-readout", action="store_true",
        help="freeze the explicit RGB angle head after its optional warmup",
    )
    parser.add_argument(
        "--explicit-goal-angle-readout", action="store_true",
        help="use a separate RGB block-angle readout and unit-circle supervision for the goal image only",
    )
    parser.add_argument("--angle-readout-loss-weight", type=float, default=0.0)
    parser.add_argument("--goal-angle-loss-weight", type=float, default=0.0)
    parser.add_argument("--contact-root-fraction", type=float, default=0.5)
    parser.add_argument("--terminal-root-fraction", type=float, default=0.0,
        help="fraction of roots placed offline on a near-goal, angle-mismatched terminal manifold",
    )
    parser.add_argument("--contact-approach-steps", type=int, default=48)
    parser.add_argument(
        "--bootstrap-checkpoint", type=Path, default=None,
        help="immutable RGB-only controller used only to collect closed-loop source roots for DAgger-style offline labels",
    )
    parser.add_argument("--bootstrap-root-fraction", type=float, default=0.0)
    parser.add_argument("--bootstrap-rollout-episodes", type=int, default=80)
    parser.add_argument("--bootstrap-rollout-steps", type=int, default=100)
    parser.add_argument("--bootstrap-contact-distance", type=float, default=70.0)
    parser.add_argument("--planning-episodes", type=int, default=12)
    parser.add_argument("--skip-planning", action="store_true", help="train and audit held-out physical effects without online planning")
    parser.add_argument("--planning-steps", type=int, default=100)
    parser.add_argument("--planning-horizon", type=int, default=6)
    parser.add_argument("--cem-samples", type=int, default=192)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument(
        "--contact-servo", action="store_true",
        help="use visual block-relative approach control until contact; retain learned dynamics for closure",
    )
    parser.add_argument("--contact-distance", type=float, default=42.0)
    parser.add_argument("--contact-tolerance", type=float, default=24.0)
    parser.add_argument(
        "--visual-contact-gate", action="store_true",
        help="require RGB-observed block motion before entering learned contact/torque phase",
    )
    parser.add_argument(
        "--direct-utility", action="store_true",
        help="rank contact actions with the learned one-step goal-utility head instead of multi-step rollout cost",
    )
    parser.add_argument(
        "--task-space-objective", action="store_true",
        help="rank actions by the learned successor in the fixed raw PushT task-space metric, not by the utility head",
    )
    parser.add_argument(
        "--successor-orientation-phase", action="store_true",
        help="during the RGB terminal phase, rank candidates by the explicit learned successor orientation effect",
    )
    parser.add_argument(
        "--geometry-state-readout", action="store_true",
        help="replace current agent/block position coordinates with RGB mask centroids at runtime; keep learned orientation/effect heads",
    )
    parser.add_argument(
        "--utility-mode", choices=("scalar", "phase"), default="scalar",
        help=(
            "scalar is the legacy, single objective utility used by the formal checkpoint; "
            "phase predicts separate translation/orientation costs.  The mode is checkpoint architecture, "
            "not a controller-only switch."
        ),
    )
    parser.add_argument(
        "--adapter-architecture", choices=("legacy", "mode_uncertain"), default="legacy",
        help="mode_uncertain adds offline-supervised contact-mode and effect-variance heads; legacy preserves prior checkpoints",
    )
    parser.add_argument(
        "--structured-readout", action="store_true",
        help="use linear RGB mask-moment pose readouts; this is a declared architecture variant, not post-hoc test calibration",
    )
    parser.add_argument(
        "--contact-frame-dynamics", action="store_true",
        help="condition learned dynamics on RGB-derived agent/block relative geometry and block-frame action coordinates",
    )
    parser.add_argument(
        "--mode-conditioned-dynamics", action="store_true",
        help="use contact-mode experts for the learned residual dynamics; runtime uses predicted mode probabilities",
    )
    parser.add_argument(
        "--soft-mode-training", action="store_true",
        help="train mode-conditioned dynamics with the same predicted-mode soft mixture used at runtime; keep labels only for the auxiliary mode loss",
    )
    parser.add_argument(
        "--torque-features", action="store_true",
        help="append the RGB-derived signed lever-arm/action cross product to contact-frame dynamics",
    )
    parser.add_argument("--uncertainty-risk-weight", type=float, default=0.25)
    parser.add_argument("--separation-risk-weight", type=float, default=0.5)
    parser.add_argument("--contact-preservation-weight", type=float, default=1.0)
    parser.add_argument(
        "--orientation-phase-distance", type=float, default=35.0,
        help="block-goal distance in pixels below which the phase utility uses orientation ranking",
    )
    parser.add_argument("--orientation-translation-weight", type=float, default=0.03)
    parser.add_argument("--visual-terminal-xy", type=float, default=20.0)
    parser.add_argument("--visual-terminal-angle", type=float, default=math.pi / 9.0)
    parser.add_argument("--orientation-action-max", type=float, default=1.0)
    parser.add_argument(
        "--train-action-support", choices=("isotropic", "cardinal"), default="isotropic",
        help="cardinal restricts training interventions to axes for held-out diagonal action tests",
    )
    parser.add_argument(
        "--train-action-max-magnitude", type=float, default=0.85,
        help="maximum nonzero intervention magnitude used in training data",
    )
    parser.add_argument("--readout-loss-weight", type=float, default=1.0)
    parser.add_argument("--transition-loss-weight", type=float, default=1.0)
    parser.add_argument("--effect-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--block-effect-loss-weight", type=float, default=0.0,
        help="additional matched-intervention loss on block position/orientation effect; zero preserves legacy checkpoints",
    )
    parser.add_argument("--orientation-transition-loss-weight", type=float, default=0.0)
    parser.add_argument("--orientation-effect-loss-weight", type=float, default=0.0)
    parser.add_argument("--goal-loss-weight", type=float, default=1.0)
    parser.add_argument("--utility-loss-weight", type=float, default=1.0)
    parser.add_argument("--rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--successor-rank-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--contact-successor-rank-only", action="store_true",
        help="apply successor action-ranking loss only to source roots collected on the contact manifold",
    )
    parser.add_argument(
        "--angle-rank-loss-weight", type=float, default=0.0,
        help="directly rank contact-root actions by the explicit wrapped-angle effect head",
    )
    parser.add_argument(
        "--contact-angle-rank-only", action="store_true",
        help="restrict the direct angle-ranking loss to source roots collected on the contact manifold",
    )
    parser.add_argument("--uncertainty-loss-weight", type=float, default=1.0)
    parser.add_argument("--mode-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def angle_error_from_sincos(left: np.ndarray, right: np.ndarray) -> float:
    left_angle = math.atan2(float(left[4]), float(left[5]))
    right_angle = math.atan2(float(right[4]), float(right[5]))
    return abs((left_angle - right_angle + math.pi) % (2.0 * math.pi) - math.pi)


def state_vector(state: np.ndarray) -> np.ndarray:
    """Physical state target with a continuous orientation chart."""
    state = np.asarray(state, dtype=np.float32)
    return np.asarray(
        [
            state[0], state[1], state[2], state[3],
            np.sin(state[4]), np.cos(state[4]), state[5], state[6],
        ],
        dtype=np.float32,
    )


def observable_pose_vector(state: np.ndarray) -> np.ndarray:
    """Single-frame visually observable PushT chart used by the model.

    The simulator state includes two velocity components.  They cannot be
    identified from one rendered image, so supervising them in an image
    readout creates an avoidable partial-observability error.  Planning needs
    agent position, block position, and block orientation only.
    """
    return state_vector(state)[:6]


def goal_pose_vector(goal_pose: np.ndarray) -> np.ndarray:
    """Convert PushT's rendered target pose ``[x, y, angle]`` to the model chart.

    ``info["goal_state"]`` is the simulator reset/evaluation state and is not
    the pose rendered in ``info["goal"]``.  In particular, its block angle is
    generally the sampled source angle, while the goal image uses
    ``info["goal_pose"]``.  Mixing the two creates an unidentifiable target
    readout task and invalidates orientation-planning labels.
    """
    pose = np.asarray(goal_pose, dtype=np.float32)
    if pose.shape != (3,):
        raise ValueError(f"goal_pose must have shape [3], got {pose.shape}")
    return np.asarray([pose[0], pose[1], np.sin(pose[2]), np.cos(pose[2])], dtype=np.float32)


def task_goal_state(goal_state: np.ndarray, goal_pose: np.ndarray) -> np.ndarray:
    """Make the simulator goal-state chart agree with the rendered goal pose."""
    pose = np.asarray(goal_pose, dtype=np.float32)
    if pose.shape != (3,):
        raise ValueError(f"goal_pose must have shape [3], got {pose.shape}")
    corrected = np.asarray(goal_state, dtype=np.float32).copy()
    corrected[2:4] = pose[:2]
    corrected[4] = pose[2]
    corrected[5:] = 0.0
    return corrected


def physical_contact_mode(source: np.ndarray, successor: np.ndarray) -> int:
    """Offline PushT contact-mode label; this label is never runtime input.

    0=no contact, 1=contact without appreciable object motion, 2=sliding,
    3=separation after contact.  The thresholds belong to the protocol and are
    intentionally stored with every generated dataset.
    """
    start_distance = float(np.linalg.norm(source[:2] - source[2:4]))
    end_distance = float(np.linalg.norm(successor[:2] - successor[2:4]))
    block_motion = float(np.linalg.norm(successor[2:4] - source[2:4]))
    if start_distance > 65.0:
        return 0
    if end_distance > 80.0:
        return 3
    if block_motion > 1.0:
        return 2
    return 1


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    image = cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_AREA)
    return np.moveaxis(image, -1, 0).astype(np.uint8, copy=False)


def reset_with_case(env: gym.Env, state: np.ndarray, goal: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    return env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})


def diverse_source_cases(
    env: gym.Env, roots: int, root_rollout_steps: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]]:
    """Collect rendered source states from a reproducible mixed behavior policy."""
    rng = np.random.default_rng(seed)
    cases: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = []
    episode_count = int(math.ceil(roots / max(root_rollout_steps, 1)))
    for episode in range(episode_count):
        observation, info = env.reset(seed=seed + episode)
        goal = task_goal_state(info["goal_state"], info["goal_pose"])
        goal_image = np.asarray(info["goal"], dtype=np.uint8).copy()
        previous_image = env.render().copy()
        previous_action = np.zeros(2, dtype=np.float32)
        for step in range(root_rollout_steps):
            state = np.asarray(observation["state"], dtype=np.float32).copy()
            image = env.render().copy()
            cases.append((state, goal.copy(), image, previous_image.copy(), goal_image.copy(), previous_action.copy(), False))
            if len(cases) >= roots:
                return cases
            to_block = state[2:4] - state[:2]
            direction = to_block / max(float(np.linalg.norm(to_block)), 1e-6)
            if step % 3 == 0:
                action = np.clip(direction + 0.15 * rng.normal(size=2), -1.0, 1.0)
            elif step % 3 == 1:
                action = rng.uniform(-0.8, 0.8, size=2)
            else:
                action = 0.55 * direction + 0.25 * rng.normal(size=2)
            observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
            previous_image = image
            previous_action = np.asarray(action, dtype=np.float32).copy()
            if terminated or truncated:
                break
    return cases[:roots]


def contact_source_cases(
    env: gym.Env, roots: int, seed: int, approach_steps: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]]:
    """Collect source images on the genuine contact/near-contact manifold."""
    rng = np.random.default_rng(seed)
    cases: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = []
    for episode in range(max(roots * 3, 1)):
        observation, info = env.reset(seed=seed + episode)
        goal = task_goal_state(info["goal_state"], info["goal_pose"])
        goal_image = np.asarray(info["goal"], dtype=np.uint8).copy()
        previous_image = env.render().copy()
        previous_action = np.zeros(2, dtype=np.float32)
        for step in range(approach_steps):
            state = np.asarray(observation["state"], dtype=np.float32).copy()
            image = env.render().copy()
            relative = state[2:4] - state[:2]
            distance = float(np.linalg.norm(relative))
            direction = relative / max(distance, 1e-6)
            if distance <= 65.0:
                cases.append((state, goal.copy(), image, previous_image.copy(), goal_image.copy(), previous_action.copy(), True))
                if len(cases) >= roots:
                    return cases
                # Perturb within the local contact chart so the offline
                # counterfactual fan contains varied contact orientations.
                tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
                action = np.clip(0.55 * direction + rng.uniform(-0.45, 0.45) * tangent, -1.0, 1.0)
            else:
                action = np.clip(direction + 0.10 * rng.normal(size=2), -1.0, 1.0)
            observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
            previous_image = image
            previous_action = np.asarray(action, dtype=np.float32).copy()
            if terminated or truncated:
                break
    return cases


def balanced_source_cases(
    env: gym.Env, roots: int, root_rollout_steps: int, seed: int,
    contact_fraction: float, contact_approach_steps: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]]:
    if not 0.0 <= contact_fraction <= 1.0:
        raise ValueError("contact_root_fraction must lie in [0, 1]")
    contact_count = int(round(roots * contact_fraction))
    free_count = roots - contact_count
    contact = contact_source_cases(env, contact_count, seed + 700_000, contact_approach_steps)
    if len(contact) != contact_count:
        raise RuntimeError(f"could not collect requested contact roots: {len(contact)} < {contact_count}")
    free = diverse_source_cases(env, free_count, root_rollout_steps, seed)
    return [*free, *contact]


def terminal_pose_cases(
    env: gym.Env, roots: int, seed: int, position_jitter: float = 48.0,
    support_gap_max: float = 0.0, support_angle_samples: int = 32,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]]:
    """Create disjoint near-goal pose roots for orientation-closure coverage.

    The simulator state is used only to construct an offline training root.
    The resulting RGB frames, not the state, are the model's runtime input.
    Angles are deliberately sampled away from the goal so torque ranking is
    measured on the terminal manifold rather than inferred from far-away
    translation cases.
    """
    if roots < 0 or position_jitter <= 0:
        raise ValueError("roots must be nonnegative and position_jitter positive")
    if support_gap_max < 0.0 or support_angle_samples < 8:
        raise ValueError("support-gap search parameters are invalid")

    def visible_support_metrics(
        image: np.ndarray, goal_direction: np.ndarray,
    ) -> tuple[float, float]:
        """Return the runtime support gap and support-normal side alignment.

        The runtime controller does not use the radial block-to-agent vector
        to certify a terminal side on an asymmetric T.  It uses the visible
        boundary lever ``-support_lever``.  Root construction must therefore
        apply the same criterion; a radial-only filter admits roots whose
        rendered support normal is on the wrong task side.
        """
        raw = np.asarray(image)
        if raw.ndim != 3 or raw.shape[-1] != 3:
            return float("inf"), float("-inf")
        # Match the runtime chart exactly: the deployed observer first
        # resizes to 64x64 and then thresholds the masks. Searching on the
        # native renderer resolution would accept roots whose downsampled
        # support gap violates the actual runtime lease.
        rgb = cv2.resize(raw, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
        if float(np.max(rgb, initial=0.0)) > 1.5:
            rgb /= 255.0
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        agent_mask = (blue > 0.70) & (blue > green + 0.12) & (blue > red + 0.18)
        block_mask = (blue < 0.93) & (red < 0.82) & (blue > green + 0.025) & (green > red + 0.025)
        agent_y, agent_x = np.where(agent_mask)
        block_y, block_x = np.where(block_mask)
        if len(agent_x) == 0 or len(block_x) == 0:
            return float("inf"), float("-inf")
        height, width = rgb.shape[:2]
        x_grid = 2.0 * np.arange(width, dtype=np.float32) / max(width - 1, 1) - 1.0
        y_grid = 2.0 * np.arange(height, dtype=np.float32) / max(height - 1, 1) - 1.0
        agent = np.asarray([x_grid[agent_x].mean(), y_grid[agent_y].mean()])
        block = np.asarray([x_grid[block_x].mean(), y_grid[block_y].mean()])
        relative = agent - block
        normal = relative / max(float(np.linalg.norm(relative)), 1.0e-6)
        points = np.column_stack([x_grid[block_x], y_grid[block_y]])
        offsets = points - block[None]
        support = offsets[int(np.argmax(offsets @ normal))]
        support_normal = -support / max(float(np.linalg.norm(support)), 1.0e-6)
        side_alignment = float(np.dot(support_normal, goal_direction))
        return (
            float(np.linalg.norm(agent - (block + support)) * 256.0),
            side_alignment,
        )

    rng = np.random.default_rng(seed)
    cases: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = []
    # Support-aware roots can be sparse near workspace boundaries; use a
    # generous deterministic rejection budget while preserving the requested
    # root count and seed-disjoint protocol.
    attempt_budget = max(roots, 1) * (20 if support_gap_max > 0.0 else 2)
    for index in range(attempt_budget):
        if len(cases) >= roots:
            break
        _observation, info = env.reset(seed=seed + index)
        goal = task_goal_state(info["goal_state"], info["goal_pose"])
        goal_image = np.asarray(info["goal"], dtype=np.uint8).copy()
        state = np.asarray(info["goal_state"], dtype=np.float32).copy()
        offset = rng.uniform(-position_jitter, position_jitter, size=2).astype(np.float32)
        block = np.clip(goal[2:4] + offset, 48.0, 464.0)
        angle = float(goal[4] + rng.uniform(-math.pi, math.pi))
        direction = goal[2:4] - block
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.asarray([1.0, 0.0], dtype=np.float32)
            norm = 1.0
        direction = direction / norm
        state[2:4] = block
        state[4] = angle
        if support_gap_max > 0.0:
            # The historical root placed the pusher on the block-goal radial
            # line. For an asymmetric T this can be close in centroid
            # distance but far from the visible support point. Search only
            # the declared RGB root-construction chart; simulator state is
            # used offline to instantiate and label the resulting root.
            candidates: list[tuple[float, np.ndarray]] = []
            for candidate_angle in np.linspace(
                -np.pi, np.pi, int(support_angle_samples), endpoint=False,
            ):
                candidate_direction = np.asarray(
                    [np.cos(candidate_angle), np.sin(candidate_angle)],
                    dtype=np.float32,
                )
                # The runtime contact gate defines the valid side as the
                # pusher being opposite the block-to-goal direction. A
                # support point on the wrong side is a valid geometric
                # contact but an invalid terminal-control root.
                side_alignment = float(np.dot(-candidate_direction, direction))
                if side_alignment < 0.35:
                    continue
                # A centroid-radius construction is not a contact
                # construction for an asymmetric T.  The old ``block +
                # 42*direction`` root can look close in RGB while the
                # pusher circle is still separated from the actual polygon
                # by 10--25 px; the runtime then enters the terminal torque
                # lease with zero physical contact and no angular authority.
                # Build the root from the simulator shape support instead:
                # place the pusher centre one agent radius outside the
                # visible support point along the candidate radial direction.
                # This is used only to instantiate/label offline roots; the
                # deployed controller still receives RGB frames only.
                state[0:2] = np.clip(
                    block + 80.0 * candidate_direction,
                    24.0, 488.0,
                )
                state[5:] = 0.0
                env.reset(
                    seed=0,
                    options={"state": state.copy(), "goal_state": goal.copy()},
                )
                block_body = env.unwrapped.block
                support_points = np.asarray([
                    [float(world_point.x), float(world_point.y)]
                    for shape in block_body.shapes
                    for point in shape.get_vertices()
                    for world_point in [block_body.local_to_world(point)]
                ], dtype=np.float32)
                support_point = support_points[int(
                    np.argmax(support_points @ candidate_direction)
                )]
                agent_radius = float(next(iter(env.unwrapped.agent.shapes)).radius)
                candidate = np.clip(
                    support_point + (agent_radius + 0.25) * candidate_direction,
                    24.0, 488.0,
                ).astype(np.float32)
                state[0:2] = candidate
                state[5:] = 0.0
                env.reset(
                    seed=0,
                    options={"state": state.copy(), "goal_state": goal.copy()},
                )
                support_gap, support_side_alignment = visible_support_metrics(
                    env.render(), direction,
                )
                # Match the runtime terminal lease exactly: radial contact
                # direction alone is insufficient for an asymmetric T.
                if support_side_alignment < 0.35:
                    continue
                candidates.append((support_gap, candidate.copy()))
            # Some sampled block poses have no candidate direction satisfying
            # both the visible-support and task-side constraints.  That is a
            # root-construction rejection, not a reason to crash the entire
            # disjoint-root protocol; keep searching within the attempt
            # budget and fail only if the requested root count is unreachable.
            if not candidates:
                continue
            best_gap, best_agent = min(candidates, key=lambda item: item[0])
            if not np.isfinite(best_gap) or best_gap > float(support_gap_max):
                continue
            state[0:2] = best_agent
        else:
            state[0:2] = np.clip(block - 42.0 * direction, 24.0, 488.0)
        state[5:] = 0.0
        _observation, _info = env.reset(seed=0, options={"state": state.copy(), "goal_state": goal.copy()})
        image = env.render().copy()
        cases.append((state, goal, image, image.copy(), goal_image, np.zeros(2, dtype=np.float32), True))
    if len(cases) != roots:
        raise RuntimeError(f"could not collect requested terminal roots: {len(cases)} < {roots}")
    return cases


def bootstrap_closed_loop_source_cases(
    env: gym.Env, checkpoint: Path, roots: int, seed: int, episodes: int,
    rollout_steps: int, contact_distance: float, device: str,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]]:
    """Collect visual-controller visitation roots, then label them offline.

    The bootstrap controller is immutable.  During collection it consumes only
    the current/previous rendered frames and target render; physical state is
    retained afterward solely to reset matched counterfactual label rollouts.
    This closes the observed state-distribution gap without turning simulator
    successors into a runtime planning interface.
    """
    if roots < 0 or episodes < 1 or rollout_steps < 1 or contact_distance <= 0:
        raise ValueError("invalid bootstrap root collection parameters")
    if roots == 0:
        return []
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = dict(payload["args"])
    model = VisualCausalWorldModel(
        int(saved["latent_dim"]), str(saved["vision_encoder"]), bool(saved["temporal_context"]),
        str(saved.get("utility_mode", "scalar")), str(saved.get("adapter_architecture", "legacy")),
        bool(saved.get("structured_readout", False)), bool(saved.get("contact_frame_dynamics", False)),
        bool(saved.get("mode_conditioned_dynamics", False)), bool(saved.get("current_only_readout", False)),
        bool(saved.get("torque_features", False)), bool(saved.get("temporal_dynamics", False)),
        bool(saved.get("previous_action_dynamics", False)),
        bool(saved.get("explicit_angle_effect", False)),
        bool(saved.get("explicit_angle_readout", False)),
        bool(saved.get("explicit_goal_angle_readout", False)),
        bool(saved.get("rich_temporal_dynamics", False)),
        bool(saved.get("invariant_angle_readout", False)),
        bool(saved.get("contact_interface_dynamics", False)),
        bool(saved.get("angle_readout_current_only", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalizer = Normalization(**payload["normalization"])
    mean, std = normalizer.tensors(device)
    model.set_state_normalization(mean, std)
    runtime = SimpleNamespace(**saved)
    runtime.device = device
    runtime.contact_servo = True
    runtime.direct_utility = True
    runtime.task_space_objective = False
    runtime.cem_samples = int(getattr(runtime, "cem_samples", 192))
    runtime.cem_iterations = int(getattr(runtime, "cem_iterations", 4))
    runtime.orientation_phase_distance = float(getattr(runtime, "orientation_phase_distance", 35.0))
    runtime.orientation_translation_weight = float(getattr(runtime, "orientation_translation_weight", 0.03))
    runtime.orientation_action_max = float(getattr(runtime, "orientation_action_max", 1.0))
    runtime.visual_terminal_xy = float(getattr(runtime, "visual_terminal_xy", 20.0))
    runtime.visual_terminal_angle = float(getattr(runtime, "visual_terminal_angle", math.pi / 9.0))
    runtime.contact_distance = float(getattr(runtime, "contact_distance", 42.0))
    runtime.contact_tolerance = float(getattr(runtime, "contact_tolerance", 24.0))
    runtime.uncertainty_risk_weight = float(getattr(runtime, "uncertainty_risk_weight", 0.25))
    runtime.separation_risk_weight = float(getattr(runtime, "separation_risk_weight", 0.5))
    runtime.contact_preservation_weight = float(getattr(runtime, "contact_preservation_weight", 1.0))
    runtime.utility_mode = str(getattr(runtime, "utility_mode", "scalar"))
    runtime.adapter_architecture = str(getattr(runtime, "adapter_architecture", "legacy"))
    generator = torch.Generator(device=device).manual_seed(seed + 991)
    collected: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]] = []
    with torch.inference_mode():
        for episode in range(episodes):
            if len(collected) >= roots:
                break
            observation, info = env.reset(seed=seed + episode)
            goal = task_goal_state(info["goal_state"], info["goal_pose"])
            goal_image = np.asarray(info["goal"], dtype=np.uint8).copy()
            previous = env.render().copy()
            previous_action = np.zeros(2, dtype=np.float32)
            runtime._orientation_phase_latched = False
            for _step in range(rollout_steps):
                image = env.render().copy()
                state = np.asarray(observation["state"], dtype=np.float32).copy()
                if float(np.linalg.norm(state[:2] - state[2:4])) <= contact_distance:
                    collected.append((state, goal.copy(), image, previous.copy(), goal_image.copy(), previous_action.copy(), True))
                    if len(collected) >= roots:
                        break
                action = visual_plan_action(model, image, previous, goal_image, normalizer, runtime, generator)
                observation, _reward, terminated, truncated, _info = env.step(action)
                previous = image
                previous_action = np.asarray(action, dtype=np.float32).copy()
                if terminated or truncated:
                    break
    if len(collected) != roots:
        raise RuntimeError(f"bootstrap controller supplied {len(collected)} contact roots, expected {roots}")
    return collected


def make_counterfactual_dataset(
    env: gym.Env, roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]],
    image_size: int, seed: int, action_repeat: int, counterfactual_fan_size: int,
    train_action_support: str = "isotropic", train_action_max_magnitude: float = 0.85,
) -> dict[str, np.ndarray]:
    """Expand every visual source into matched +a/-a/zero action outcomes."""
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    previous_images: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    goal_poses: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    previous_actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    root_ids: list[int] = []
    contact_roots: list[bool] = []
    contact_modes: list[int] = []
    for root_id, (state, goal, image, previous_image, goal_image, previous_action, contact_root) in enumerate(roots):
        if train_action_support == "isotropic":
            direction = rng.normal(size=2).astype(np.float32)
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
        elif train_action_support == "cardinal":
            direction = np.eye(2, dtype=np.float32)[int(rng.integers(0, 2))]
            if bool(rng.integers(0, 2)):
                direction *= -1.0
        else:
            raise ValueError(f"unknown train_action_support: {train_action_support}")
        if not 0.18 <= train_action_max_magnitude <= 1.0:
            raise ValueError("train_action_max_magnitude must lie in [0.18, 1]")
        magnitude = float(rng.uniform(0.18, train_action_max_magnitude))
        actions_for_root = [magnitude * direction, -magnitude * direction, np.zeros(2, dtype=np.float32)]
        if counterfactual_fan_size:
            if counterfactual_fan_size < 0:
                raise ValueError("counterfactual_fan_size must be nonnegative")
            # Offline intervention construction may use state/goal labels; the
            # learned model and runtime controller receive only rendered
            # images.  This fan covers normal and tangential contact effects
            # that random +/- actions almost never resolve reliably.
            base = np.asarray(goal[2:4] - state[2:4], dtype=np.float32)
            base /= max(float(np.linalg.norm(base)), 1e-6)
            for offset in np.linspace(0.0, 2.0 * np.pi, int(counterfactual_fan_size), endpoint=False):
                c, s = float(np.cos(offset)), float(np.sin(offset))
                rotated = np.asarray([c * base[0] - s * base[1], s * base[0] + c * base[1]], dtype=np.float32)
                actions_for_root.append(0.85 * rotated)
        for action in actions_for_root:
            observation, _ = reset_with_case(env, state, goal)
            for _ in range(action_repeat):
                observation, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
                if terminated or truncated:
                    break
            images.append(resize_image(image, image_size))
            previous_images.append(resize_image(previous_image, image_size))
            goals.append(resize_image(goal_image, image_size))
            # The goal image depicts the target object alone.  Its target pose
            # must be learned through its own visual readout rather than passed
            # through the current-scene state head (which is trained on a
            # pusher + block + target observation distribution).
            goal_poses.append(observable_pose_vector(goal)[2:6])
            actions.append(np.asarray(action, dtype=np.float32))
            previous_actions.append(np.asarray(previous_action, dtype=np.float32))
            states.append(observable_pose_vector(state))
            next_states.append(observable_pose_vector(np.asarray(observation["state"], dtype=np.float32)))
            root_ids.append(root_id)
            contact_roots.append(bool(contact_root))
            contact_modes.append(physical_contact_mode(state, np.asarray(observation["state"], dtype=np.float32)))
    out = {
        "image": np.asarray(images, dtype=np.uint8),
        "previous_image": np.asarray(previous_images, dtype=np.uint8),
        "goal_image": np.asarray(goals, dtype=np.uint8),
        "goal_pose": np.asarray(goal_poses, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "previous_action": np.asarray(previous_actions, dtype=np.float32),
        "state": np.asarray(states, dtype=np.float32),
        "next_state": np.asarray(next_states, dtype=np.float32),
        "root_id": np.asarray(root_ids, dtype=np.int64),
        "contact_root": np.asarray(contact_roots, dtype=bool),
        "contact_mode": np.asarray(contact_modes, dtype=np.int64),
    }
    informative = np.zeros(len(out["root_id"]), dtype=bool)
    for root_id in np.unique(out["root_id"]):
        idx = np.flatnonzero(out["root_id"] == root_id)
        translation = np.linalg.norm(out["next_state"][idx, 2:4] - out["state"][idx, 2:4], axis=1)
        orientation = np.linalg.norm(out["next_state"][idx, 4:6] - out["state"][idx, 4:6], axis=1)
        informative[idx] = bool(np.max(translation) > 1.0 or np.max(orientation) > 0.02)
    out["utility_informative"] = informative
    return out


class VisualCausalWorldModel(nn.Module):
    """Image encoder, state readout, and action-conditioned local dynamics."""

    def __init__(
        self, latent_dim: int = 96, vision_encoder: str = "spatial_mlp", temporal_context: bool = False,
        utility_mode: str = "scalar", adapter_architecture: str = "legacy", structured_readout: bool = False,
        contact_frame_dynamics: bool = False, mode_conditioned_dynamics: bool = False,
        current_only_readout: bool = False, torque_features: bool = False, temporal_dynamics: bool = False,
        previous_action_dynamics: bool = False,
        explicit_angle_effect: bool = False,
        explicit_angle_readout: bool = False,
        explicit_goal_angle_readout: bool = False,
        rich_temporal_dynamics: bool = False,
        invariant_angle_readout: bool = False,
        contact_interface_dynamics: bool = False,
        angle_readout_current_only: bool = False,
    ):
        super().__init__()
        if utility_mode not in {"scalar", "phase"}:
            raise ValueError(f"unknown utility_mode: {utility_mode}")
        if adapter_architecture not in {"legacy", "mode_uncertain"}:
            raise ValueError(f"unknown adapter_architecture: {adapter_architecture}")
        self.uses_mask_moments = vision_encoder == "mask_moments"
        self.temporal_context = bool(temporal_context)
        self.utility_mode = utility_mode
        self.adapter_architecture = adapter_architecture
        self.structured_readout = bool(structured_readout)
        self.contact_frame_dynamics = bool(contact_frame_dynamics)
        self.mode_conditioned_dynamics = bool(mode_conditioned_dynamics)
        self.current_only_readout = bool(current_only_readout)
        self.torque_features = bool(torque_features)
        self.temporal_dynamics = bool(temporal_dynamics)
        self.rich_temporal_dynamics = bool(rich_temporal_dynamics)
        self.previous_action_dynamics = bool(previous_action_dynamics)
        self.explicit_angle_effect = bool(explicit_angle_effect)
        self.explicit_angle_readout = bool(explicit_angle_readout)
        self.explicit_goal_angle_readout = bool(explicit_goal_angle_readout)
        self.invariant_angle_readout = bool(invariant_angle_readout)
        self.contact_interface_dynamics = bool(contact_interface_dynamics)
        self.contact_interface_dim = MaskMomentGeometryEncoder.interface_feature_dim if self.contact_interface_dynamics else 0
        self.contact_interface_action_dim = 3 if self.contact_interface_dynamics else 0
        self.angle_readout_current_only = bool(angle_readout_current_only)
        self._state_mean: torch.Tensor | None = None
        self._state_std: torch.Tensor | None = None
        if self.structured_readout and vision_encoder != "mask_moments":
            raise ValueError("structured_readout requires the mask_moments RGB geometry encoder")
        if self.explicit_angle_readout and not self.uses_mask_moments:
            raise ValueError("explicit_angle_readout requires the mask_moments RGB geometry encoder")
        if self.invariant_angle_readout and not self.explicit_angle_readout:
            raise ValueError("invariant_angle_readout requires explicit_angle_readout")
        if self.temporal_context and not self.uses_mask_moments:
            raise ValueError("temporal_context is currently implemented for mask_moments only")
        if self.contact_interface_dynamics and not self.uses_mask_moments:
            raise ValueError("contact_interface_dynamics requires the mask_moments RGB encoder")
        if self.temporal_dynamics and not self.temporal_context:
            raise ValueError("temporal_dynamics requires temporal_context")
        if self.contact_interface_dynamics and not self.temporal_dynamics:
            raise ValueError("contact_interface_dynamics requires temporal_dynamics")
        if self.contact_interface_dynamics and not self.contact_frame_dynamics:
            raise ValueError("contact_interface_dynamics requires contact_frame_dynamics")
        if self.rich_temporal_dynamics and not self.temporal_dynamics:
            raise ValueError("rich_temporal_dynamics requires temporal_dynamics")
        if vision_encoder == "mask_moments":
            self.encoder = MaskMomentGeometryEncoder(latent_dim)
        elif vision_encoder == "color_geometry":
            self.encoder = ColorGeometryEncoder(latent_dim)
        elif vision_encoder == "spatial_mlp":
            # PushT is a low-dimensional geometric scene.  Preserve its
            # 16x16 colour/location chart directly so position is not
            # discarded before learning the latent state map.
            self.encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d((16, 16)), nn.Flatten(),
                nn.Linear(3 * 16 * 16, 512), nn.SiLU(),
                nn.Linear(512, latent_dim), nn.SiLU(),
            )
        elif vision_encoder == "conv":
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.SiLU(),
                nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                nn.Linear(128 * 4 * 4, latent_dim), nn.SiLU(),
            )
        else:
            raise ValueError(f"unknown vision encoder {vision_encoder}")
        # Keep the full RGB-derived moment chart available to the pose heads.
        # A two-stage latent bottleneck was unnecessarily lossy for this small
        # geometric benchmark; the latent is still used by the non-structured
        # vision ablations.
        feature_dim = MaskMomentGeometryEncoder.feature_dim if self.uses_mask_moments else latent_dim
        state_feature_dim = 2 * feature_dim if self.temporal_context and not self.current_only_readout else feature_dim
        if self.structured_readout:
            self.state_head = nn.Linear(state_feature_dim, 6)
        else:
            self.state_head = nn.Sequential(nn.Linear(state_feature_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, 6))
        if self.explicit_angle_readout:
            angle_feature_dim = MaskMomentGeometryEncoder.object_feature_dim if self.invariant_angle_readout else feature_dim
            self.angle_readout_head = nn.Sequential(
                nn.Linear(
                    (2 * angle_feature_dim if self.temporal_context and not self.angle_readout_current_only else angle_feature_dim)
                    if self.invariant_angle_readout else
                    (2 * feature_dim if self.temporal_context and not self.angle_readout_current_only else feature_dim),
                    128,
                ),
                nn.SiLU(), nn.Linear(128, 2),
            )
        if self.explicit_goal_angle_readout:
            self.goal_angle_head = nn.Sequential(
                nn.Linear(feature_dim, 128), nn.SiLU(), nn.Linear(128, 2),
            )
        # Goal rendering requires a nonlinear calibration: a T's visual
        # centroid changes with orientation relative to its simulator pose.
        # Keep this decoder separate from current-scene state readout.
        self.goal_head = (
            nn.Linear(feature_dim, 4)
            if self.structured_readout else
            nn.Sequential(
                nn.Linear(feature_dim, latent_dim), nn.SiLU(),
                nn.Linear(latent_dim, latent_dim), nn.SiLU(), nn.Linear(latent_dim, 4),
            )
        )
        dynamics_core_dim = (16 if self.torque_features else 15) if self.contact_frame_dynamics else 6 + 2
        temporal_base_dim = 11 if self.rich_temporal_dynamics else 4
        temporal_dim = temporal_base_dim + self.contact_interface_dim
        dynamics_temporal_dim = (4 if self.rich_temporal_dynamics else temporal_base_dim) + self.contact_interface_dim
        dynamics_input_dim = dynamics_core_dim + (dynamics_temporal_dim if self.temporal_dynamics else 0) + self.contact_interface_action_dim + (2 if self.previous_action_dynamics else 0)
        angle_dynamics_input_dim = dynamics_core_dim + (temporal_dim if self.temporal_dynamics else 0) + self.contact_interface_action_dim + (2 if self.previous_action_dynamics else 0)
        self.dynamics = nn.Sequential(
            nn.Linear(dynamics_input_dim, 192), nn.SiLU(), nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 6),
        )
        if self.explicit_angle_effect:
            self.angle_effect_head = nn.Sequential(
                nn.Linear(angle_dynamics_input_dim, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1),
            )
        if adapter_architecture == "mode_uncertain":
            self.effect_log_variance = nn.Sequential(
                nn.Linear(dynamics_input_dim, 128), nn.SiLU(), nn.Linear(128, 6),
            )
            self.contact_mode_head = nn.Sequential(
                nn.Linear(dynamics_input_dim, 128), nn.SiLU(), nn.Linear(128, 4),
            )
        if self.mode_conditioned_dynamics:
            if adapter_architecture != "mode_uncertain":
                raise ValueError("mode_conditioned_dynamics requires adapter_architecture=mode_uncertain")
            self.mode_dynamics = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dynamics_input_dim, 192), nn.SiLU(),
                    nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 6),
                ) for _ in range(4)
            ])
        self.utility_head = nn.Sequential(
            nn.Linear(6 + 4 + 2 + (temporal_dim if self.temporal_dynamics else 0) + (2 if self.previous_action_dynamics else 0), 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1 if utility_mode == "scalar" else 2),
        )

    def encode_state(self, image: torch.Tensor, previous_image: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder.features(image) if self.uses_mask_moments else self.encoder(image)
        if self.temporal_context and not self.current_only_readout:
            if previous_image is None:
                previous_image = image
            previous = self.encoder.features(previous_image)
            # Current geometry and its observed finite difference make velocity
            # and contact approach direction visually identifiable.
            features = torch.cat([features, features - previous], dim=-1)
        state = self.state_head(features)
        if self.explicit_angle_readout:
            if self._state_mean is None or self._state_std is None:
                raise RuntimeError("explicit_angle_readout requires set_state_normalization before encoding")
            # The angle head predicts a unit-circle chart directly from the
            # current RGB geometry.  Convert it into the normalized state
            # chart only at the interface consumed by dynamics.
            current_features = self._angle_readout_features(image)
            if self.temporal_context and not self.angle_readout_current_only:
                if previous_image is None:
                    previous_image = image
                previous_features = self._angle_readout_features(previous_image)
                angle_features = torch.cat([current_features, current_features - previous_features], dim=-1)
            else:
                angle_features = current_features
            raw_angle = F.normalize(self.angle_readout_head(angle_features), dim=-1)
            normalized_angle = (raw_angle - self._state_mean[4:6]) / self._state_std[4:6]
            state = state.clone()
            state[:, 4:6] = normalized_angle
        return state

    def _angle_readout_features(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder.features(image)
        if self.invariant_angle_readout:
            # MaskMomentGeometryEncoder concatenates agent, block, and target
            # ten-dimensional moment charts.  The block chart is translation
            # invariant in its central moments and avoids letting the angle
            # head memorize contact position or pusher location.
            features = features[:, MaskMomentGeometryEncoder.feature_dim // 3: 2 * MaskMomentGeometryEncoder.feature_dim // 3]
        return features

    def set_state_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Attach the raw pose chart used by the contact-frame feature map.

        The tensors are runtime calibration, not trainable parameters, and are
        deliberately kept out of the state dict so legacy checkpoints remain
        loadable.  A contact-frame model must be calibrated before prediction.
        """
        if mean.shape[-1] < 6 or std.shape[-1] < 6:
            raise ValueError("state normalization must cover the six pose coordinates")
        self._state_mean = mean.detach().clone()
        self._state_std = std.detach().clone().clamp_min(1e-6)

    def encode_dynamics_context(
        self, image: torch.Tensor, previous_image: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Return RGB-observed centroid velocities for local effect prediction.

        Absolute pose is deliberately decoded from the current frame under
        ``current_only_readout``.  This separate channel preserves the
        temporally observable agent/block motion required by contact dynamics
        without letting frame differencing corrupt the pose chart.
        """
        if not self.temporal_dynamics:
            return None
        if previous_image is None:
            previous_image = image
        current = self.encoder.features(image)
        previous = self.encoder.features(previous_image)
        delta = current - previous
        if self.rich_temporal_dynamics:
            # Pusher centroid motion plus block silhouette changes.  These
            # features are causal RGB observations and avoid injecting an
            # offline angle label into the runtime dynamics interface.
            context = torch.cat([delta[:, 1:3], delta[:, 11:20]], dim=-1)
        else:
            context = delta[:, (1, 2, 11, 12)]
        if self.contact_interface_dynamics:
            # This chart is computed from the current RGB masks, not from
            # simulator state.  It gives the effect head the observed local
            # boundary/interface geometry that moments alone do not retain.
            context = torch.cat([context, self.encoder.contact_interface_features(image)], dim=-1)
        return context

    def _dynamics_input(
        self, state: torch.Tensor, action: torch.Tensor, temporal_context: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None, rich_context: bool = False,
    ) -> torch.Tensor:
        base = torch.cat([state, action], dim=-1)
        if not self.contact_frame_dynamics:
            values = [base]
            if self.temporal_dynamics:
                if temporal_context is None:
                    temporal_context = torch.zeros(state.shape[0], (11 if self.rich_temporal_dynamics else 4) + self.contact_interface_dim, dtype=state.dtype, device=state.device)
                elif self.rich_temporal_dynamics and not rich_context:
                    temporal_context = temporal_context[:, :4 + self.contact_interface_dim]
                values.append(temporal_context)
            if self.previous_action_dynamics:
                if previous_action is None:
                    previous_action = torch.zeros_like(action)
                values.append(previous_action)
            return torch.cat(values, dim=-1)
        if self._state_mean is None or self._state_std is None:
            raise RuntimeError("contact-frame dynamics requires set_state_normalization before prediction")
        raw = state * self._state_std + self._state_mean
        # state[4:6] is the continuous sin/cos block-orientation chart.
        orientation = F.normalize(raw[:, 4:6], dim=-1)
        block_normal = torch.stack([orientation[:, 1], orientation[:, 0]], dim=-1)
        block_tangent = torch.stack([-orientation[:, 0], orientation[:, 1]], dim=-1)
        agent_relative = (raw[:, :2] - raw[:, 2:4]) / PUSHT_RENDER_SIZE
        contact_normal = F.normalize(-agent_relative, dim=-1)
        contact_tangent = torch.stack([-contact_normal[:, 1], contact_normal[:, 0]], dim=-1)
        contact_action = torch.stack([
            (action * contact_normal).sum(dim=-1),
            (action * contact_tangent).sum(dim=-1),
        ], dim=-1)
        block_action = torch.stack([
            (action * block_normal).sum(dim=-1),
            (action * block_tangent).sum(dim=-1),
        ], dim=-1)
        distance = torch.linalg.vector_norm(agent_relative, dim=-1, keepdim=True)
        values = [base, agent_relative, contact_action, block_action, distance]
        if self.torque_features:
            signed_torque = (agent_relative[:, 0] * action[:, 1] - agent_relative[:, 1] * action[:, 0]).unsqueeze(-1)
            values.append(signed_torque)
        if self.temporal_dynamics:
            if temporal_context is None:
                temporal_context = torch.zeros(state.shape[0], (11 if self.rich_temporal_dynamics else 4) + self.contact_interface_dim, dtype=state.dtype, device=state.device)
            elif self.rich_temporal_dynamics and not rich_context:
                temporal_context = temporal_context[:, :4 + self.contact_interface_dim]
            values.append(temporal_context)
        if self.previous_action_dynamics:
            if previous_action is None:
                previous_action = torch.zeros_like(action)
            values.append(previous_action)
        if self.contact_interface_dynamics:
            if temporal_context is None:
                raise RuntimeError("contact interface dynamics requires an RGB interface context")
            interface = temporal_context[:, -self.contact_interface_dim:]
            lever = interface[:, 3:5]
            lever = F.normalize(lever, dim=-1)
            tangent = torch.stack([-lever[:, 1], lever[:, 0]], dim=-1)
            interface_action = torch.stack([
                (action * lever).sum(dim=-1),
                (action * tangent).sum(dim=-1),
                lever[:, 0] * action[:, 1] - lever[:, 1] * action[:, 0],
            ], dim=-1)
            values.append(interface_action)
        return torch.cat(values, dim=-1)

    def predict_next(
        self, state: torch.Tensor, action: torch.Tensor, contact_mode: torch.Tensor | None = None,
        temporal_context: torch.Tensor | None = None, previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self._dynamics_input(state, action, temporal_context, previous_action)
        if not self.mode_conditioned_dynamics:
            delta = self.dynamics(features)
        else:
            experts = torch.stack([head(features) for head in self.mode_dynamics], dim=1)
            if contact_mode is not None:
                if contact_mode.ndim != 1 or contact_mode.shape[0] != state.shape[0]:
                    raise ValueError("contact_mode must have shape [batch]")
                weights = F.one_hot(contact_mode.long().clamp(0, 3), num_classes=4).to(experts.dtype)
            else:
                logits = self.contact_mode_head(features)
                weights = torch.softmax(logits, dim=-1)
            delta = (experts * weights.unsqueeze(-1)).sum(dim=1)
        predicted = state + delta
        if self.explicit_angle_effect:
            if self._state_mean is None or self._state_std is None:
                raise RuntimeError("explicit_angle_effect requires set_state_normalization before prediction")
            raw = state * self._state_std + self._state_mean
            current_angle = torch.atan2(raw[:, 4], raw[:, 5])
            angle_features = self._dynamics_input(
                state, action, temporal_context, previous_action,
                rich_context=self.rich_temporal_dynamics,
            )
            next_angle = current_angle + self.angle_effect_head(angle_features).squeeze(-1)
            raw_predicted = predicted * self._state_std + self._state_mean
            raw_predicted = raw_predicted.clone()
            raw_predicted[:, 4] = torch.sin(next_angle)
            raw_predicted[:, 5] = torch.cos(next_angle)
            predicted = (raw_predicted - self._state_mean) / self._state_std
        return predicted

    def predict_angle_delta(
        self, state: torch.Tensor, action: torch.Tensor,
        temporal_context: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.explicit_angle_effect:
            return torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        return self.angle_effect_head(
            self._dynamics_input(
                state, action, temporal_context, previous_action,
                rich_context=self.rich_temporal_dynamics,
            )
        ).squeeze(-1)

    def predict_distribution(
        self, state: torch.Tensor, action: torch.Tensor, contact_mode: torch.Tensor | None = None,
        temporal_context: torch.Tensor | None = None, previous_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step mean and diagonal log-variance in normalized state units."""
        mean = self.predict_next(state, action, contact_mode, temporal_context, previous_action)
        if self.adapter_architecture == "legacy":
            return mean, torch.zeros_like(mean)
        log_variance = self.effect_log_variance(self._dynamics_input(state, action, temporal_context, previous_action)).clamp(-8.0, 5.0)
        return mean, log_variance

    def predict_contact_logits(
        self, state: torch.Tensor, action: torch.Tensor, temporal_context: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.adapter_architecture == "legacy":
            return None
        return self.contact_mode_head(self._dynamics_input(state, action, temporal_context, previous_action))

    def encode_goal_pose(self, goal_image: torch.Tensor) -> torch.Tensor:
        """Decode target block position and orientation from the goal image."""
        features = self.encoder.features(goal_image) if self.uses_mask_moments else self.encoder(goal_image)
        pose = self.goal_head(features)
        if self.explicit_goal_angle_readout:
            if self._state_mean is None or self._state_std is None:
                raise RuntimeError("explicit_goal_angle_readout requires set_state_normalization before goal encoding")
            raw_angle = F.normalize(self.goal_angle_head(features), dim=-1)
            normalized_angle = (raw_angle - self._state_mean[4:6]) / self._state_std[4:6]
            pose = pose.clone()
            pose[:, 2:4] = normalized_angle
        return pose

    def score_action(
        self, state: torch.Tensor, goal_pose: torch.Tensor, action: torch.Tensor,
        temporal_context: torch.Tensor | None = None, previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predicted total normalized one-step pose cost; lower is better."""
        score = self.score_action_components(state, goal_pose, action, temporal_context, previous_action)
        return score.squeeze(-1) if self.utility_mode == "scalar" else score.sum(dim=-1)

    def score_action_components(
        self, state: torch.Tensor, goal_pose: torch.Tensor, action: torch.Tensor,
        temporal_context: torch.Tensor | None = None, previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one scalar cost or [translation, orientation] phase costs."""
        values = [state, goal_pose, action]
        if self.temporal_dynamics:
            if temporal_context is None:
                temporal_context = torch.zeros(state.shape[0], (11 if self.rich_temporal_dynamics else 4) + self.contact_interface_dim, dtype=state.dtype, device=state.device)
            values.append(temporal_context)
        if self.previous_action_dynamics:
            if previous_action is None:
                previous_action = torch.zeros_like(action)
            values.append(previous_action)
        return self.utility_head(torch.cat(values, dim=-1))

    def forward(
        self, image: torch.Tensor, action: torch.Tensor, previous_image: torch.Tensor | None = None,
        contact_mode: torch.Tensor | None = None, previous_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encode_state(image, previous_image)
        temporal_context = self.encode_dynamics_context(image, previous_image)
        return state, self.predict_next(state, action, contact_mode, temporal_context, previous_action)


class ColorGeometryEncoder(nn.Module):
    """RGB-only spatial chart for PushT's rendered agent, block and target.

    The thresholds are applied to pixels, not simulator metadata.  They remove
    the uniform white background and preserve a spatial 16x16 chart on which
    the learned latent/readout and dynamics modules operate.  This is an
    explicit structured-vision ablation appropriate for this colour-coded
    benchmark, not a claim of generic visual perception.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(3 * 16 * 16, 512), nn.SiLU(),
            nn.Linear(512, latent_dim), nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        # The public PushT rendering uses royal-blue pusher, slate-gray T,
        # and light-green target.  Each mask has a margin around antialiased
        # boundaries and excludes the white/background grid.
        agent = ((blue > 0.70) & (blue > green + 0.12) & (blue > red + 0.18)).float()
        block = ((blue < 0.93) & (red < 0.82) & (blue > green + 0.025) & (green > red + 0.025)).float()
        target = ((green > red + 0.14) & (green > blue + 0.14)).float()
        chart = torch.cat([agent, block, target], dim=1)
        chart = F.adaptive_avg_pool2d(chart, (16, 16)).flatten(1)
        return self.project(chart)


class MaskMomentGeometryEncoder(nn.Module):
    """RGB-mask moment chart that preserves object pose before learning.

    PushT renders a blue pusher, gray T block, and green target.  Rather than
    discard these small objects through spatial pooling, this encoder extracts
    their zeroth through third central moments directly from pixels and lets a
    learned map calibrate them to the simulator's pose chart.  It consumes RGB
    images only; it is deliberately a colour-coded toy-domain front end, not a
    generic vision backbone.
    """

    object_feature_dim = 10
    interface_feature_dim = 11
    # Keep the learned pose chart exactly backward compatible.  The interface
    # chart is exposed through ``contact_interface_features`` and is only
    # concatenated into dynamics context when explicitly enabled.
    feature_dim = 3 * object_feature_dim

    def __init__(self, latent_dim: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(30, 128), nn.SiLU(), nn.Linear(128, latent_dim), nn.SiLU(),
        )

    @staticmethod
    def _masks(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        red, green, blue = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        agent = ((blue > 0.70) & (blue > green + 0.12) & (blue > red + 0.18)).float()
        block = ((blue < 0.93) & (red < 0.82) & (blue > green + 0.025) & (green > red + 0.025)).float()
        target = ((green > red + 0.14) & (green > blue + 0.14)).float()
        return agent, block, target

    @staticmethod
    def _moments(mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width = mask.shape
        ys = torch.linspace(-1.0, 1.0, height, device=mask.device, dtype=mask.dtype).view(1, 1, height, 1)
        xs = torch.linspace(-1.0, 1.0, width, device=mask.device, dtype=mask.dtype).view(1, 1, 1, width)
        mass = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean_x = (mask * xs).sum(dim=(2, 3), keepdim=True) / mass
        mean_y = (mask * ys).sum(dim=(2, 3), keepdim=True) / mass
        dx, dy = xs - mean_x, ys - mean_y
        # Normalized central moments carry centre, orientation, and the T's
        # directed asymmetry without using simulator state at runtime.
        central = [
            (mask * dx.square()).sum(dim=(2, 3), keepdim=True) / mass,
            (mask * dx * dy).sum(dim=(2, 3), keepdim=True) / mass,
            (mask * dy.square()).sum(dim=(2, 3), keepdim=True) / mass,
            (mask * dx.pow(3)).sum(dim=(2, 3), keepdim=True) / mass,
            (mask * dx.square() * dy).sum(dim=(2, 3), keepdim=True) / mass,
            (mask * dx * dy.square()).sum(dim=(2, 3), keepdim=True) / mass,
            (mask * dy.pow(3)).sum(dim=(2, 3), keepdim=True) / mass,
        ]
        normalized_mass = mass / float(height * width)
        return torch.cat([normalized_mass, mean_x, mean_y, *central], dim=-1).flatten(1)

    def features(self, image: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._moments(mask) for mask in self._masks(image)], dim=-1)

    @staticmethod
    def _interface_features(agent: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
        """Return a causal RGB chart for the pusher/block interface.

        Central moments identify the objects but do not retain which boundary
        part is nearest the pusher.  This chart uses only thresholded RGB
        masks: relative centroid, nearest block-pixel direction/gap, object
        mass/radius, and a validity bit.  It is deliberately a geometric
        interface descriptor, not a simulator contact label.
        """
        batch, _, height, width = agent.shape
        ys = torch.linspace(-1.0, 1.0, height, device=agent.device, dtype=agent.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=agent.device, dtype=agent.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        # All other RGB moment charts use [x, y].  The previous implementation
        # stacked [y, x], silently rotating the support vector before it was
        # consumed by runtime contact geometry and by contact-interface
        # dynamics.  Keep the chart convention explicit here.
        grid = torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)
        agent_flat = agent[:, 0].reshape(batch, -1)
        block_flat = block[:, 0].reshape(batch, -1)
        agent_mass = agent_flat.sum(dim=-1)
        block_mass = block_flat.sum(dim=-1)
        valid = (agent_mass > 0.5) & (block_mass > 0.5)
        safe_agent_mass = agent_mass.clamp_min(1.0)
        safe_block_mass = block_mass.clamp_min(1.0)
        agent_center = (agent_flat.unsqueeze(-1) * grid).sum(dim=1) / safe_agent_mass.unsqueeze(-1)
        block_center = (block_flat.unsqueeze(-1) * grid).sum(dim=1) / safe_block_mass.unsqueeze(-1)
        relative = agent_center - block_center
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        agent_radius = torch.sqrt(
            ((agent_flat.unsqueeze(-1) * (grid - agent_center.unsqueeze(1)).square()).sum(dim=(1, 2))
             / safe_agent_mass).clamp_min(0.0)
        ).unsqueeze(-1)
        block_radius = torch.sqrt(
            ((block_flat.unsqueeze(-1) * (grid - block_center.unsqueeze(1)).square()).sum(dim=(1, 2))
             / safe_block_mass).clamp_min(0.0)
        ).unsqueeze(-1)
        distances = torch.linalg.vector_norm(grid - agent_center.unsqueeze(1), dim=-1)
        distances = distances.masked_fill(~block_flat.bool(), float("inf"))
        nearest_index = distances.argmin(dim=-1)
        nearest = grid.expand(batch, -1, -1).gather(
            1, nearest_index[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        nearest_gap = distances.gather(1, nearest_index[:, None]).clamp_max(2.0)
        contact_vector = nearest - block_center
        values = torch.cat([
            relative, distance, contact_vector, nearest_gap,
            (agent_mass / float(height * width)).unsqueeze(-1),
            (block_mass / float(height * width)).unsqueeze(-1),
            agent_radius, block_radius, valid.to(agent.dtype).unsqueeze(-1),
        ], dim=-1)
        return torch.nan_to_num(values, nan=0.0, posinf=2.0, neginf=-2.0)

    def contact_interface_features(self, image: torch.Tensor) -> torch.Tensor:
        agent, block, _target = self._masks(image)
        return self._interface_features(agent, block)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.project(self.features(image))


@dataclass
class Normalization:
    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, values: np.ndarray) -> "Normalization":
        mean = np.asarray(values, dtype=np.float32).mean(axis=0)
        std = np.asarray(values, dtype=np.float32).std(axis=0)
        std = np.maximum(std, 1e-3)
        return cls(mean.tolist(), std.tolist())

    def tensors(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.mean, device=device, dtype=torch.float32),
            torch.tensor(self.std, device=device, dtype=torch.float32),
        )


def tensor_images(values: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device) / 255.0


def model_loss(
    model: VisualCausalWorldModel, image: torch.Tensor, action: torch.Tensor,
    state: torch.Tensor, next_state: torch.Tensor, goal_image: torch.Tensor,
    goal_pose: torch.Tensor, next_state_raw: torch.Tensor, goal_pose_raw: torch.Tensor,
    state_raw: torch.Tensor, root_id: torch.Tensor | None = None, utility_informative: torch.Tensor | None = None,
    previous_image: torch.Tensor | None = None, contact_mode: torch.Tensor | None = None,
    previous_action: torch.Tensor | None = None,
    loss_weights: dict[str, float] | None = None,
    raw_mean: torch.Tensor | None = None, raw_std: torch.Tensor | None = None,
    soft_mode_training: bool = False,
    contact_root: torch.Tensor | None = None,
    contact_successor_rank_only: bool = False,
    contact_angle_rank_only: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = {
        "readout": 1.0, "transition": 1.0, "effect": 0.5, "goal": 1.0,
        "block_effect": 0.0,
        "orientation_transition": 0.0, "orientation_effect": 0.0,
        "angle_effect": 0.0,
        "angle_readout": 0.0,
        "contact_angle_readout": 1.0,
        "angle_rank": 0.0,
        "goal_angle": 0.0,
        "utility": 1.0, "rank": 1.0, "successor_rank": 0.0,
        "uncertainty": 1.0, "mode": 0.5,
    }
    if loss_weights is not None:
        weights.update({key: float(value) for key, value in loss_weights.items()})
    # A hard offline contact-mode label is useful as an auxiliary target, but
    # must not route the dynamics during training when runtime has to infer the
    # mode from RGB.  Otherwise the experts are trained under oracle routing
    # and evaluated under a predicted soft mixture, which is a direct
    # train/runtime interface mismatch for action-effect ranking.
    routing_mode = None if soft_mode_training else contact_mode
    decoded, predicted = model(image, action, previous_image, routing_mode, previous_action)
    temporal_context = model.encode_dynamics_context(image, previous_image)
    readout = F.mse_loss(decoded, state)
    transition = F.mse_loss(predicted, next_state)
    # A direct effect loss prevents action information from being discarded by
    # a state-only next-state predictor under matched source triples.
    effect = F.mse_loss(predicted - decoded, next_state - state)
    # Orientation/action-effect supervision is only identifiable when the
    # source intervention has contact support.  On free-space rows the target
    # orientation is mechanically unchanged, so including those rows gives a
    # large, easy no-motion gradient and suppresses the rare contact signal.
    # Mode 3 is retained: it denotes separation after a contact-supported
    # transition and still carries a valid last-contact effect.
    contact_supported = (
        contact_mode > 0 if contact_mode is not None
        else torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    )

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if values.ndim > 1:
            values = values.reshape(values.shape[0], -1).mean(dim=-1)
        if not bool(mask.any().item()):
            return values.new_zeros(())
        return values[mask].mean()

    block_effect = masked_mean(
        ((predicted - decoded)[:, 2:6] - (next_state - state)[:, 2:6]).square(),
        contact_supported,
    )
    orientation_transition = masked_mean(
        (predicted[:, 4:6] - next_state[:, 4:6]).square(), contact_supported
    )
    orientation_effect = masked_mean(
        ((predicted - decoded)[:, 4:6] - (next_state - state)[:, 4:6]).square(),
        contact_supported,
    )
    angle_effect = predicted.new_zeros(())
    angle_rank = predicted.new_zeros(())
    if model.explicit_angle_effect:
        predicted_angle_delta = model.predict_angle_delta(decoded, action, temporal_context, previous_action)
        current_angle = torch.atan2(state_raw[:, 4], state_raw[:, 5])
        next_angle = torch.atan2(next_state_raw[:, 4], next_state_raw[:, 5])
        target_angle_delta = torch.atan2(
            torch.sin(next_angle - current_angle), torch.cos(next_angle - current_angle),
        )
        angle_effect = masked_mean(
            F.smooth_l1_loss(predicted_angle_delta, target_angle_delta, reduction="none"),
            contact_supported,
        )
        if root_id is not None and weights["angle_rank"] > 0.0:
            angle_rank_terms = []
            current_angle_all = torch.atan2(state_raw[:, 4], state_raw[:, 5])
            goal_angle_all = torch.atan2(goal_pose_raw[:, 2], goal_pose_raw[:, 3])
            predicted_angle_cost = 1.0 - torch.cos(current_angle_all + predicted_angle_delta - goal_angle_all)
            true_angle = torch.atan2(next_state_raw[:, 4], next_state_raw[:, 5])
            true_angle_cost = 1.0 - torch.cos(true_angle - goal_angle_all)
            for current_root in torch.unique(root_id):
                idx = torch.nonzero(root_id == current_root, as_tuple=False).flatten()
                if idx.numel() < 2:
                    continue
                if contact_angle_rank_only and contact_root is not None and not bool(contact_root[idx].any().item()):
                    continue
                label = torch.argmin(true_angle_cost[idx]).reshape(1)
                angle_rank_terms.append(F.cross_entropy((-predicted_angle_cost[idx]).reshape(1, -1), label))
            if angle_rank_terms:
                angle_rank = torch.stack(angle_rank_terms).mean()
    angle_readout = predicted.new_zeros(())
    if model.explicit_angle_readout:
        if raw_mean is None or raw_std is None:
            raise ValueError("explicit angle readout requires raw normalization")
        decoded_raw = decoded * raw_std + raw_mean
        decoded_angle = torch.atan2(decoded_raw[:, 4], decoded_raw[:, 5])
        target_angle = torch.atan2(state_raw[:, 4], state_raw[:, 5])
        per_row_angle_readout = 1.0 - torch.cos(decoded_angle - target_angle)
        if contact_root is not None:
            if contact_root.shape != per_row_angle_readout.shape:
                raise ValueError("contact_root must have shape [batch]")
            contact_weight = float(weights["contact_angle_readout"])
            row_weights = torch.where(
                contact_root.bool(),
                torch.as_tensor(contact_weight, dtype=per_row_angle_readout.dtype, device=per_row_angle_readout.device),
                torch.ones_like(per_row_angle_readout),
            )
            angle_readout = (per_row_angle_readout * row_weights).sum() / row_weights.sum().clamp_min(1.0)
        else:
            angle_readout = per_row_angle_readout.mean()
    uncertainty = predicted.new_zeros(())
    mode_loss = predicted.new_zeros(())
    if model.adapter_architecture == "mode_uncertain":
        distribution_mean, log_variance = model.predict_distribution(decoded, action, routing_mode, temporal_context, previous_action)
        residual = next_state - distribution_mean
        uncertainty = 0.5 * (log_variance + residual.square() * torch.exp(-log_variance)).mean()
        logits = model.predict_contact_logits(decoded, action, temporal_context, previous_action)
        if contact_mode is None or logits is None:
            raise ValueError("mode_uncertain training requires offline contact_mode labels")
        mode_loss = F.cross_entropy(logits, contact_mode)
    goal = F.mse_loss(model.encode_goal_pose(goal_image), goal_pose)
    goal_angle = predicted.new_zeros(())
    if model.explicit_goal_angle_readout:
        if raw_mean is None or raw_std is None:
            raise ValueError("explicit goal angle readout requires raw normalization")
        encoded_goal_raw = model.encode_goal_pose(goal_image) * raw_std[2:6] + raw_mean[2:6]
        encoded_goal_angle = torch.atan2(encoded_goal_raw[:, 2], encoded_goal_raw[:, 3])
        target_goal_angle = torch.atan2(goal_pose_raw[:, 2], goal_pose_raw[:, 3])
        goal_angle = (1.0 - torch.cos(encoded_goal_angle - target_goal_angle)).mean()
    orientation = (
        F.normalize(next_state_raw[:, 4:6], dim=-1) - F.normalize(goal_pose_raw[:, 2:4], dim=-1)
    ).square().sum(dim=-1)
    translation = torch.linalg.vector_norm(next_state_raw[:, 2:4] - goal_pose_raw[:, :2], dim=-1)
    # Separate targets preserve the terminal control interface: orientation
    # should dominate only after translation enters its task tube.
    phase_utility_target = torch.stack([torch.log1p(translation) / 6.0, orientation], dim=-1)
    utility_target = (
        phase_utility_target.sum(dim=-1, keepdim=True)
        if model.utility_mode == "scalar" else phase_utility_target
    )
    utility_prediction = model.score_action_components(
        decoded, model.encode_goal_pose(goal_image), action, temporal_context, previous_action,
    )
    if utility_informative is not None and bool(utility_informative.any().item()):
        utility = F.mse_loss(utility_prediction[utility_informative], utility_target[utility_informative])
    else:
        utility = utility_prediction.new_zeros(())
    rank_terms = []
    if root_id is not None:
        # Each source is expanded into +a, -a, and zero.  The ranking term
        # cancels source-specific distance-to-goal and asks the utility head
        # the actual control question: which intervention is best *here*?
        for current_root in torch.unique(root_id):
            idx = torch.nonzero(root_id == current_root, as_tuple=False).flatten()
            if idx.numel() >= 2 and (utility_informative is None or bool(utility_informative[idx].any().item())):
                current_translation = torch.linalg.vector_norm(state_raw[idx[0], 2:4] - goal_pose_raw[idx[0], :2])
                if model.utility_mode == "scalar":
                    target_score = utility_target[idx, 0]
                    predicted_score = utility_prediction[idx, 0]
                elif float(current_translation.item()) > 35.0:
                    target_score = utility_target[idx, 0]
                    predicted_score = utility_prediction[idx, 0]
                else:
                    target_score = utility_target[idx, 1] + 0.03 * utility_target[idx, 0]
                    predicted_score = utility_prediction[idx, 1] + 0.03 * utility_prediction[idx, 0]
                label = torch.argmin(target_score).reshape(1)
                rank_terms.append(F.cross_entropy((-predicted_score).reshape(1, -1), label))
    utility_rank = torch.stack(rank_terms).mean() if rank_terms else utility_prediction.new_zeros(())
    successor_rank_terms = []
    if raw_mean is not None and raw_std is not None and root_id is not None:
        predicted_raw = predicted * raw_std + raw_mean
        for current_root in torch.unique(root_id):
            idx = torch.nonzero(root_id == current_root, as_tuple=False).flatten()
            if idx.numel() >= 2:
                if contact_successor_rank_only and contact_root is not None and not bool(contact_root[idx].any().item()):
                    continue
                pred_orientation = F.normalize(predicted_raw[idx, 4:6], dim=-1)
                goal_orientation = F.normalize(goal_pose_raw[idx, 2:4], dim=-1)
                pred_cost = torch.linalg.vector_norm(
                    predicted_raw[idx, 2:4] - goal_pose_raw[idx, :2], dim=-1
                ) / 20.0
                pred_cost = pred_cost + 10.0 * (pred_orientation - goal_orientation).square().sum(dim=-1)
                true_orientation = F.normalize(next_state_raw[idx, 4:6], dim=-1)
                true_cost = torch.linalg.vector_norm(
                    next_state_raw[idx, 2:4] - goal_pose_raw[idx, :2], dim=-1
                ) / 20.0
                true_cost = true_cost + 10.0 * (true_orientation - goal_orientation).square().sum(dim=-1)
                label = torch.argmin(true_cost).reshape(1)
                successor_rank_terms.append(F.cross_entropy((-pred_cost).reshape(1, -1), label))
    successor_rank = torch.stack(successor_rank_terms).mean() if successor_rank_terms else utility_prediction.new_zeros(())
    loss = (
        weights["readout"] * readout
        + weights["transition"] * transition
        + weights["effect"] * effect
        + weights["block_effect"] * block_effect
        + weights["orientation_transition"] * orientation_transition
        + weights["orientation_effect"] * orientation_effect
        + weights["angle_effect"] * angle_effect
        + weights["angle_rank"] * angle_rank
        + weights["angle_readout"] * angle_readout
        + weights["goal"] * goal
        + weights["goal_angle"] * goal_angle
        + weights["utility"] * utility
        + weights["rank"] * utility_rank
        + weights["successor_rank"] * successor_rank
        + weights["uncertainty"] * uncertainty
        + weights["mode"] * mode_loss
    )
    return loss, {
        "readout": readout.detach(), "transition": transition.detach(),
        "effect": effect.detach(), "block_effect": block_effect.detach(), "goal": goal.detach(), "utility": utility.detach(), "utility_rank": utility_rank.detach(),
        "orientation_transition": orientation_transition.detach(),
        "orientation_effect": orientation_effect.detach(),
        "angle_effect": angle_effect.detach(),
        "angle_rank": angle_rank.detach(),
        "angle_readout": angle_readout.detach(),
        "goal_angle": goal_angle.detach(),
        "successor_rank": successor_rank.detach(),
        "uncertainty": uncertainty.detach(), "mode": mode_loss.detach(),
        "contact_supported_fraction": contact_supported.float().mean().detach(),
    }


def train_model(
    train: dict[str, np.ndarray], val: dict[str, np.ndarray], args: argparse.Namespace,
) -> tuple[VisualCausalWorldModel, Normalization, dict[str, Any], list[dict[str, Any]]]:
    device = args.device
    torch.manual_seed(args.seed)
    model = VisualCausalWorldModel(
        args.latent_dim, args.vision_encoder, args.temporal_context, args.utility_mode, args.adapter_architecture,
        args.structured_readout, args.contact_frame_dynamics, args.mode_conditioned_dynamics,
        args.current_only_readout, args.torque_features, args.temporal_dynamics,
        args.previous_action_dynamics,
        args.explicit_angle_effect,
        args.explicit_angle_readout,
        args.explicit_goal_angle_readout,
        args.rich_temporal_dynamics,
        args.invariant_angle_readout,
        args.contact_interface_dynamics,
        args.angle_readout_current_only,
    ).to(device)
    normalizer = Normalization.fit(np.concatenate([train["state"], train["next_state"]], axis=0))
    mean, std = normalizer.tensors(device)
    model.set_state_normalization(mean, std)
    rng = np.random.default_rng(args.seed + 1)
    history: list[dict[str, float]] = []

    def root_batches(data: dict[str, np.ndarray], roots: np.ndarray) -> list[np.ndarray]:
        """Keep all counterfactual action siblings in the same device batch.

        Moving a complete corpus to CUDA made the dense contact intervention
        arm fail before optimisation began.  Root-preserving host-to-device
        batches retain the ranking contract while making memory scale with the
        batch rather than with the intervention corpus.
        """
        groups = [np.flatnonzero(data["root_id"] == root) for root in roots]
        batches, current, current_size = [], [], 0
        for group in groups:
            if len(group) > args.batch_size:
                raise ValueError("one counterfactual root exceeds batch_size")
            if current and current_size + len(group) > args.batch_size:
                batches.append(np.concatenate(current))
                current, current_size = [], 0
            current.append(group)
            current_size += len(group)
        if current:
            batches.append(np.concatenate(current))
        return batches

    def device_batch(data: dict[str, np.ndarray], index: np.ndarray) -> tuple[torch.Tensor, ...]:
        # Deliberately materialise only one complete-source batch on device.
        image = tensor_images(data["image"][index], device)
        previous = tensor_images(data["previous_image"][index], device)
        goal_image = tensor_images(data["goal_image"][index], device)
        action = torch.as_tensor(data["action"][index], dtype=torch.float32, device=device)
        state_raw = torch.as_tensor(data["state"][index], dtype=torch.float32, device=device)
        next_raw = torch.as_tensor(data["next_state"][index], dtype=torch.float32, device=device)
        goal_raw = torch.as_tensor(data["goal_pose"][index], dtype=torch.float32, device=device)
        return (
            image, action, (state_raw - mean) / std, (next_raw - mean) / std,
            goal_image, (goal_raw - mean[2:6]) / std[2:6], next_raw, goal_raw,
            state_raw, torch.as_tensor(data["root_id"][index], device=device),
            torch.as_tensor(data["utility_informative"][index], device=device), previous,
            torch.as_tensor(data["contact_mode"][index], device=device),
            torch.as_tensor(data["previous_action"][index], dtype=torch.float32, device=device),
        )

    if args.freeze_angle_readout and not args.explicit_angle_readout:
        raise ValueError("freeze_angle_readout requires explicit_angle_readout")
    if args.angle_readout_warmup_steps < 0:
        raise ValueError("angle_readout_warmup_steps must be nonnegative")
    if args.angle_readout_warmup_steps and not args.explicit_angle_readout:
        raise ValueError("angle_readout_warmup_steps requires explicit_angle_readout")
    if args.angle_readout_warmup_steps:
        # The explicit angle head consumes fixed RGB mask moments.  Warm it up
        # against the current-frame angle before the multi-task objective is
        # introduced, so translation/utility gradients cannot erase the rare
        # contact-manifold pose signal.
        warmup_optimizer = torch.optim.AdamW(
            model.angle_readout_head.parameters(), lr=args.lr, weight_decay=1e-4,
        )
        warmup_batches = root_batches(train, np.unique(train["root_id"]))
        for step in range(args.angle_readout_warmup_steps):
            index = warmup_batches[step % len(warmup_batches)]
            image, _action, _state, _next_state, _goal_image, _goal_pose, _next_raw, _goal_raw, state_raw, _root_id, _informative, previous, _mode, _previous_action = device_batch(train, index)
            decoded = model.encode_state(image, previous)
            decoded_raw = decoded * std + mean
            decoded_angle = torch.atan2(decoded_raw[:, 4], decoded_raw[:, 5])
            target_angle = torch.atan2(state_raw[:, 4], state_raw[:, 5])
            per_row = 1.0 - torch.cos(decoded_angle - target_angle)
            root_mask = torch.as_tensor(train["contact_root"][index], device=device).bool()
            row_weights = torch.where(
                root_mask,
                torch.full_like(per_row, float(args.contact_angle_readout_weight)),
                torch.ones_like(per_row),
            )
            warmup_loss = (per_row * row_weights).sum() / row_weights.sum().clamp_min(1.0)
            warmup_optimizer.zero_grad(set_to_none=True)
            warmup_loss.backward()
            warmup_optimizer.step()
    if args.freeze_angle_readout:
        for parameter in model.angle_readout_head.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    snapshot_epochs = sorted(set(int(epoch) for epoch in args.snapshot_epochs))
    if any(epoch < 1 or epoch > args.epochs for epoch in snapshot_epochs):
        raise ValueError("snapshot_epochs must lie in [1, epochs]")
    snapshots: list[dict[str, Any]] = []
    loss_weights = {
        "readout": args.readout_loss_weight,
        "transition": args.transition_loss_weight,
        "effect": args.effect_loss_weight,
        "block_effect": args.block_effect_loss_weight,
        "orientation_transition": args.orientation_transition_loss_weight,
        "orientation_effect": args.orientation_effect_loss_weight,
        "angle_effect": args.angle_effect_loss_weight,
        "angle_readout": args.angle_readout_loss_weight,
        "contact_angle_readout": args.contact_angle_readout_weight,
        "angle_rank": args.angle_rank_loss_weight,
        "goal_angle": args.goal_angle_loss_weight,
        "goal": args.goal_loss_weight,
        "utility": args.utility_loss_weight,
        "rank": args.rank_loss_weight,
        "successor_rank": args.successor_rank_loss_weight,
        "uncertainty": args.uncertainty_loss_weight,
        "mode": args.mode_loss_weight,
    }
    for epoch in range(args.epochs):
        model.train()
        root_order = rng.permutation(np.unique(train["root_id"]))
        losses = []
        for index in root_batches(train, root_order):
            batch = device_batch(train, index)
            loss, _ = model_loss(
                model, *batch,
                loss_weights, raw_mean=mean, raw_std=std,
                soft_mode_training=args.soft_mode_training,
                contact_root=torch.as_tensor(train["contact_root"][index], device=device),
                contact_successor_rank_only=args.contact_successor_rank_only,
                contact_angle_rank_only=args.contact_angle_rank_only,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        validation = []
        with torch.no_grad():
            for index in root_batches(val, np.unique(val["root_id"])):
                val_loss, terms = model_loss(
                    model, *device_batch(val, index), loss_weights, raw_mean=mean, raw_std=std,
                    soft_mode_training=args.soft_mode_training,
                    contact_root=torch.as_tensor(val["contact_root"][index], device=device),
                    contact_successor_rank_only=args.contact_successor_rank_only,
                    contact_angle_rank_only=args.contact_angle_rank_only,
                )
                validation.append((len(index), float(val_loss.cpu()), {key: float(item.cpu()) for key, item in terms.items()}))
        total = sum(item[0] for item in validation)
        value = float(sum(size * loss for size, loss, _terms in validation) / total)
        averaged_terms = {
            key: float(sum(size * terms[key] for size, _loss, terms in validation) / total)
            for key in validation[0][2]
        }
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": value, **averaged_terms})
        if value < best_val:
            best_val = value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch + 1 in snapshot_epochs:
            snapshots.append({
                "epoch": int(epoch + 1),
                "validation_loss": value,
                "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            })
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    # A post-hoc affine fit over raw moments was deliberately rejected here:
    # its design is rank-deficient under rendered-image aliases and can
    # interpolate training target charts while catastrophically failing on
    # held-out goals.  Retain the validation-selected learned readout.
    return model.eval(), normalizer, {
        "history": history,
        "best_val_loss": best_val,
        "goal_readout_calibration": {"method": "validation_selected_gradient", "rejected_posthoc_affine": True},
        "snapshot_epochs_requested": snapshot_epochs,
        "snapshot_epochs_saved": [item["epoch"] for item in snapshots],
    }, snapshots


def source_split_manifest(roots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]]) -> dict[str, Any]:
    """Persist root-level split identities so action siblings cannot leak across splits."""
    rows = [
        {
            "root_id": index,
            "state": np.asarray(state, dtype=np.float32).round(6).tolist(),
            "goal": np.asarray(goal, dtype=np.float32).round(6).tolist(),
            "contact_source": bool(contact),
        }
        for index, (state, goal, _image, _previous, _goal_image, _previous_action, contact) in enumerate(roots)
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"unit": "source root / initial state", "roots": rows, "sha256": hashlib.sha256(encoded).hexdigest()}


@torch.inference_mode()
def physical_metrics(
    model: VisualCausalWorldModel, data: dict[str, np.ndarray], normalizer: Normalization,
    device: str,
) -> dict[str, Any]:
    mean, std = normalizer.tensors(device)
    image = tensor_images(data["image"], device)
    previous_image = tensor_images(data["previous_image"], device)
    action = torch.as_tensor(data["action"], device=device)
    previous_action = torch.as_tensor(data.get("previous_action", np.zeros_like(data["action"])), dtype=torch.float32, device=device)
    decoded, predicted = model(image, action, previous_image, previous_action=previous_action)
    # Audit the train/runtime routing gap without using the oracle result for
    # the reported runtime metric.  The oracle-routed pass is diagnostic only:
    # if it is much better than the soft pass, mode inference remains the
    # bottleneck; if both are poor, the effect/readout interface is at fault.
    contact_mode = torch.as_tensor(data["contact_mode"], dtype=torch.long, device=device)
    _decoded_oracle, predicted_oracle = model(
        image, action, previous_image, contact_mode=contact_mode, previous_action=previous_action,
    )
    goal_image = tensor_images(data["goal_image"], device)
    normalized_goal_pose = model.encode_goal_pose(goal_image)
    goal_pose = (normalized_goal_pose * std[2:6] + mean[2:6]).cpu().numpy()
    goal_state_chart = model.encode_state(goal_image, goal_image)
    goal_state_raw = (goal_state_chart * std + mean).cpu().numpy()
    utility = model.score_action(
        decoded, normalized_goal_pose, action, model.encode_dynamics_context(image, previous_image), previous_action,
    ).cpu().numpy()
    decoded = (decoded * std + mean).cpu().numpy()
    predicted = (predicted * std + mean).cpu().numpy()
    predicted_oracle = (predicted_oracle * std + mean).cpu().numpy()
    state = data["state"]
    target = data["next_state"]
    root = data["root_id"]
    current_angles = np.arctan2(state[:, 4], state[:, 5])
    decoded_angles = np.arctan2(decoded[:, 4], decoded[:, 5])
    target_angles = np.arctan2(target[:, 4], target[:, 5])
    predicted_angles = np.arctan2(predicted[:, 4], predicted[:, 5])
    wrap = lambda values: np.arctan2(np.sin(values), np.cos(values))
    readout_angle_rmse = float(np.sqrt(np.mean(wrap(decoded_angles - current_angles) ** 2)))
    contact_rows = np.asarray(data["contact_root"], dtype=bool)
    contact_angle_readout_rmse = float(np.sqrt(np.mean(wrap(decoded_angles[contact_rows] - current_angles[contact_rows]) ** 2))) if contact_rows.any() else float("nan")
    noncontact_rows = ~contact_rows
    noncontact_angle_readout_rmse = float(np.sqrt(np.mean(wrap(decoded_angles[noncontact_rows] - current_angles[noncontact_rows]) ** 2))) if noncontact_rows.any() else float("nan")
    successor_angle_rmse = float(np.sqrt(np.mean(wrap(predicted_angles - target_angles) ** 2)))
    target_angle_delta = wrap(target_angles - current_angles)
    predicted_angle_delta = wrap(predicted_angles - decoded_angles)
    angle_delta_std = float(np.std(target_angle_delta))
    angle_delta_corr = float(np.corrcoef(target_angle_delta, predicted_angle_delta)[0, 1]) if angle_delta_std > 1e-8 and float(np.std(predicted_angle_delta)) > 1e-8 else 0.0
    goal_angles = np.arctan2(goal_pose[:, 2], goal_pose[:, 3])
    target_goal_angles = np.arctan2(data["goal_pose"][:, 2], data["goal_pose"][:, 3])
    goal_angle_readout_rmse = float(np.sqrt(np.mean(wrap(goal_angles - target_goal_angles) ** 2)))
    goal_state_angles = np.arctan2(goal_state_raw[:, 4], goal_state_raw[:, 5])
    goal_state_angle_rmse = float(np.sqrt(np.mean(wrap(goal_state_angles - target_goal_angles) ** 2)))
    temporal_context = model.encode_dynamics_context(image, previous_image)
    mode_logits = model.predict_contact_logits(
        (torch.as_tensor(decoded, dtype=torch.float32, device=device) - mean) / std,
        action,
        temporal_context=temporal_context, previous_action=previous_action,
    )
    mode_top1 = (
        float((mode_logits.argmax(dim=-1) == contact_mode).float().mean().cpu())
        if mode_logits is not None else float("nan")
    )
    angle_delta_rmse = float("nan")
    if model.explicit_angle_effect:
        normalized_decoded = (torch.as_tensor(decoded, dtype=torch.float32, device=device) - mean) / std
        predicted_delta = model.predict_angle_delta(
            normalized_decoded, action, temporal_context, previous_action,
        )
        current_angle = torch.atan2(
            torch.as_tensor(data["state"][:, 4], device=device),
            torch.as_tensor(data["state"][:, 5], device=device),
        )
        next_angle = torch.atan2(
            torch.as_tensor(data["next_state"][:, 4], device=device),
            torch.as_tensor(data["next_state"][:, 5], device=device),
        )
        target_delta = torch.atan2(torch.sin(next_angle - current_angle), torch.cos(next_angle - current_angle))
        angle_delta_rmse = float(torch.sqrt(torch.mean((predicted_delta - target_delta).square())).cpu())
    # Evaluate action differences on matched +a/-a/zero source triples.
    action_effect_error, action_rank_hits, orientation_rank_hits, oracle_orientation_rank_hits, utility_rank_hits = [], [], [], [], []
    contact_action_effect_error, contact_action_rank_hits, contact_orientation_rank_hits, contact_oracle_orientation_rank_hits = [], [], [], []
    for root_id in np.unique(root):
        idx = np.flatnonzero(root == root_id)
        if len(idx) < 3:
            continue
        plus, minus, zero = idx[:3]
        true_effect = target[plus, 2:6] - target[minus, 2:6]
        pred_effect = predicted[plus, 2:6] - predicted[minus, 2:6]
        action_effect_error.append(float(np.linalg.norm(pred_effect - true_effect)))
        true_motion = np.linalg.norm(target[idx, 2:4] - state[idx, 2:4], axis=1)
        pred_motion = np.linalg.norm(predicted[idx, 2:4] - decoded[idx, 2:4], axis=1)
        hit = float(int(np.argmax(true_motion) == int(np.argmax(pred_motion))))
        action_rank_hits.append(hit)
        true_orientation = np.sum(
            (target[idx, 4:6] / np.maximum(np.linalg.norm(target[idx, 4:6], axis=1, keepdims=True), 1e-6)
             - data["goal_pose"][idx, 2:4] / np.maximum(np.linalg.norm(data["goal_pose"][idx, 2:4], axis=1, keepdims=True), 1e-6)) ** 2,
            axis=1,
        )
        true_goal_cost = np.sum((target[idx, 2:4] - data["goal_pose"][idx, :2]) ** 2, axis=1) + 2400.0 * true_orientation
        predicted_orientation = np.sum(
            (predicted[idx, 4:6] / np.maximum(np.linalg.norm(predicted[idx, 4:6], axis=1, keepdims=True), 1e-6)
             - data["goal_pose"][idx, 2:4] / np.maximum(np.linalg.norm(data["goal_pose"][idx, 2:4], axis=1, keepdims=True), 1e-6)) ** 2,
            axis=1,
        )
        orientation_hit = float(np.argmin(true_orientation) == np.argmin(predicted_orientation))
        orientation_rank_hits.append(orientation_hit)
        oracle_orientation = np.sum(
            (predicted_oracle[idx, 4:6] / np.maximum(np.linalg.norm(predicted_oracle[idx, 4:6], axis=1, keepdims=True), 1e-6)
             - data["goal_pose"][idx, 2:4] / np.maximum(np.linalg.norm(data["goal_pose"][idx, 2:4], axis=1, keepdims=True), 1e-6)) ** 2,
            axis=1,
        )
        oracle_orientation_rank_hits.append(float(np.argmin(true_orientation) == np.argmin(oracle_orientation)))
        if bool(data["utility_informative"][plus]):
            utility_rank_hits.append(float(int(np.argmin(true_goal_cost) == int(np.argmin(utility[idx])))))
        if bool(data["contact_root"][plus]):
            contact_action_effect_error.append(float(np.linalg.norm(pred_effect - true_effect)))
            contact_action_rank_hits.append(hit)
            contact_orientation_rank_hits.append(orientation_hit)
            contact_oracle_orientation_rank_hits.append(oracle_orientation_rank_hits[-1])
    return {
        "state_endpoint_rmse": float(np.sqrt(np.mean(np.square(predicted[:, :6] - target[:, :6])))),
        "state_readout_rmse": float(np.sqrt(np.mean(np.square(decoded[:, :6] - state[:, :6])))),
        "angle_readout_rmse": readout_angle_rmse,
        "contact_angle_readout_rmse": contact_angle_readout_rmse,
        "noncontact_angle_readout_rmse": noncontact_angle_readout_rmse,
        "angle_successor_rmse": successor_angle_rmse,
        "angle_delta_target_std": angle_delta_std,
        "angle_delta_prediction_corr": angle_delta_corr,
        "goal_pose_readout_rmse": float(np.sqrt(np.mean(np.square(goal_pose - data["goal_pose"]))),),
        "goal_angle_readout_rmse": goal_angle_readout_rmse,
        "goal_state_angle_rmse": goal_state_angle_rmse,
        "paired_plus_minus_effect_error": float(np.mean(action_effect_error)),
        "paired_action_motion_top1": float(np.mean(action_rank_hits)),
        "paired_orientation_top1": float(np.mean(orientation_rank_hits)),
        "paired_oracle_mode_orientation_top1": float(np.mean(oracle_orientation_rank_hits)),
        "paired_goal_utility_top1": float(np.mean(utility_rank_hits)) if utility_rank_hits else float("nan"),
        "paired_goal_utility_informative_roots": int(len(utility_rank_hits)),
        "contact_paired_plus_minus_effect_error": float(np.mean(contact_action_effect_error)),
        "contact_paired_action_motion_top1": float(np.mean(contact_action_rank_hits)),
        "contact_paired_orientation_top1": float(np.mean(contact_orientation_rank_hits)),
        "contact_paired_oracle_mode_orientation_top1": float(np.mean(contact_oracle_orientation_rank_hits)),
        "contact_mode_top1": mode_top1,
        "paired_angle_delta_rmse": angle_delta_rmse,
        "contact_paired_roots": int(len(contact_action_effect_error)),
        "paired_roots": int(len(action_effect_error)),
        "finite": bool(np.isfinite(predicted).all() and np.isfinite(decoded).all()),
    }


def raw_from_normalized(state: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return state * std + mean


def task_space_action_cost(
    predicted_raw: torch.Tensor,
    goal_pose_raw: torch.Tensor,
    current_raw: torch.Tensor,
    *,
    contact_distance: float = 42.0,
) -> torch.Tensor:
    """Score predicted successors in the fixed external PushT task metric.

    All arguments are image-derived at runtime.  The orientation term uses the
    normalized sin/cos chart, so it has no angle-wrap discontinuity.  The
    block position and orientation terms define the terminal task; the agent
    term only regularizes approach/contact geometry.
    """
    if predicted_raw.ndim != 2 or predicted_raw.shape[1] < 6:
        raise ValueError("predicted_raw must have shape [batch, >=6]")
    if goal_pose_raw.ndim != 2 or goal_pose_raw.shape[1] < 4:
        raise ValueError("goal_pose_raw must have shape [batch, >=4]")
    if current_raw.ndim != 2 or current_raw.shape[1] < 4:
        raise ValueError("current_raw must have shape [batch, >=4]")
    block_error = torch.linalg.vector_norm(predicted_raw[:, 2:4] - goal_pose_raw[:, :2], dim=-1) / PUSHT_RENDER_SIZE
    predicted_orientation = F.normalize(predicted_raw[:, 4:6], dim=-1)
    goal_orientation = F.normalize(goal_pose_raw[:, 2:4], dim=-1)
    orientation_error = 0.5 * (predicted_orientation - goal_orientation).square().sum(dim=-1)
    push_direction = F.normalize(goal_pose_raw[:, :2] - current_raw[:, 2:4], dim=-1)
    desired_agent = current_raw[:, 2:4] - float(contact_distance) * push_direction
    agent_error = torch.linalg.vector_norm(predicted_raw[:, :2] - desired_agent, dim=-1) / PUSHT_RENDER_SIZE
    return block_error.square() + 4.0 * orientation_error + 0.05 * agent_error.square()


def mask_moment_task_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Convert a mask-moment distance to the declared PushT task chart.

    Every RGB geometry gate is compared against thresholds in the same
    224-unit task chart used by the evaluator.  Keeping this conversion in
    one place prevents terminal holding and phase control from activating at
    incompatible spatial scales.
    """
    return torch.linalg.vector_norm(left - right, dim=-1) * PUSHT_RENDER_SIZE


@torch.inference_mode()
def visual_plan_action(
    model: VisualCausalWorldModel, image: np.ndarray, previous_image: np.ndarray, goal_image: np.ndarray,
    normalizer: Normalization, args: argparse.Namespace, generator: torch.Generator,
) -> np.ndarray:
    """CEM action selection from images only; no simulator successor is queried."""
    device = args.device
    mean, std = normalizer.tensors(device)
    current = tensor_images(resize_image(image, args.image_size)[None], device)
    previous = tensor_images(resize_image(previous_image, args.image_size)[None], device)
    goal = tensor_images(resize_image(goal_image, args.image_size)[None], device)
    current_state = model.encode_state(current, previous)
    temporal_context = model.encode_dynamics_context(current, previous)
    previous_action = torch.as_tensor(
        np.asarray(getattr(args, "_previous_action", np.zeros(2)), dtype=np.float32)[None],
        dtype=torch.float32, device=device,
    )
    raw_current = raw_from_normalized(current_state, mean, std)
    normalized_goal_pose = model.encode_goal_pose(goal)
    goal_pose = normalized_goal_pose * std[2:6] + mean[2:6]
    if model.uses_mask_moments:
        # Runtime terminal latch from RGB only.  Once both pose components
        # enter the declared terminal tube, hold with zero action instead of
        # allowing another learned torque proposal to undo the state.
        current_features = model.encoder.features(current)
        goal_features = model.encoder.features(goal)
        if bool(getattr(args, "geometry_state_readout", False)):
            geometry_current = raw_current.clone()
            geometry_current[:, :2] = (current_features[:, 1:3] + 1.0) * (PUSHT_RENDER_SIZE / 2.0)
            geometry_current[:, 2:4] = (current_features[:, 11:13] + 1.0) * (PUSHT_RENDER_SIZE / 2.0)
            current_state = (geometry_current - mean) / std
            raw_current = geometry_current
            geometry_goal = goal_pose.clone()
            geometry_goal[:, :2] = (goal_features[:, 11:13] + 1.0) * (PUSHT_RENDER_SIZE / 2.0)
            goal_pose = geometry_goal
            normalized_goal_pose = (goal_pose - mean[2:6]) / std[2:6]
        visual_block_xy = mask_moment_task_distance(current_features[0, 11:13], goal_features[0, 11:13])
        current_angle = torch.atan2(raw_current[0, 4], raw_current[0, 5])
        goal_angle = torch.atan2(goal_pose[0, 2], goal_pose[0, 3])
        visual_angle = torch.abs(torch.atan2(
            torch.sin(goal_angle - current_angle), torch.cos(goal_angle - current_angle),
        ))
        if (
            float(visual_block_xy.item()) <= float(getattr(args, "visual_terminal_xy", 20.0))
            and float(visual_angle.item()) <= float(getattr(args, "visual_terminal_angle", math.pi / 9.0))
        ):
            return np.zeros(2, dtype=np.float32)
    contact_geometry: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if args.contact_servo and model.uses_mask_moments:
        # Gate contact from image geometry itself.  The learned state readout
        # can be locally accurate on the validation pool yet misplace a far
        # pusher by enough pixels to falsely declare contact.  These centroids
        # are RGB-only measurements from the current and goal renderings.
        current_features = model.encoder.features(current)
        goal_features = model.encoder.features(goal)
        agent = current_features[0, 1:3]
        block = current_features[0, 11:13]
        goal_block = goal_features[0, 11:13]
        contact_geometry = (agent, block, goal_block)
        target_direction = F.normalize(goal_block - block, dim=-1)
        desired_agent = block - (float(args.contact_distance) / PUSHT_RENDER_SIZE) * target_direction
        approach = desired_agent - agent
        if bool(getattr(args, "visual_contact_gate", False)) and not bool(getattr(args, "_visual_contact_established", False)):
            previous_features = model.encoder.features(previous)
            observed_block_motion = torch.linalg.vector_norm(block - previous_features[0, 11:13]) * PUSHT_RENDER_SIZE
            if float(observed_block_motion.item()) >= float(getattr(args, "visual_contact_motion_threshold", 1.0)):
                setattr(args, "_visual_contact_established", True)
            else:
                return F.normalize(approach, dim=-1).detach().cpu().numpy().astype(np.float32)
        if float(torch.linalg.vector_norm(approach).item()) > float(args.contact_tolerance) / PUSHT_RENDER_SIZE:
            return F.normalize(approach, dim=-1).detach().cpu().numpy().astype(np.float32)
    elif args.contact_servo:
        block = raw_current[0, 2:4]
        agent = raw_current[0, :2]
        target_direction = F.normalize(goal_pose[0, 0:2] - block, dim=-1)
        desired_agent = block - float(args.contact_distance) * target_direction
        approach = desired_agent - agent
        if float(torch.linalg.vector_norm(approach).item()) > float(args.contact_tolerance):
            return F.normalize(approach, dim=-1).detach().cpu().numpy().astype(np.float32)
    if args.direct_utility:
        action_mean = torch.zeros(2, device=device)
        action_std = torch.full((2,), 0.55, device=device)
        orientation_phase_latched = bool(getattr(args, "_orientation_phase_latched", False))
        phase_templates = None
        for iteration in range(args.cem_iterations):
            candidates = (torch.randn(args.cem_samples, 2, device=device, generator=generator) * action_std + action_mean).clamp(-1.0, 1.0)
            if iteration == 0 and contact_geometry is not None:
                agent, block, goal_block = contact_geometry
                push_direction = F.normalize(goal_block - block, dim=-1)
                tangent = torch.stack((-push_direction[1], push_direction[0]))
                # The contact-direction fan mirrors the successful oracle
                # proposal family, but is built only from RGB geometry.  It
                # gives the learned utility head normal and torque-producing
                # alternatives instead of hoping Gaussian samples discover
                # the narrow contact manifold.
                current_angle = torch.atan2(raw_current[0, 4], raw_current[0, 5])
                goal_angle = torch.atan2(goal_pose[0, 2], goal_pose[0, 3])
                signed_delta = torch.atan2(torch.sin(goal_angle - current_angle), torch.cos(goal_angle - current_angle))
                preferred_tangent = tangent if float(signed_delta.item()) >= 0.0 else -tangent
                directions = torch.stack([
                    push_direction, tangent, -tangent, preferred_tangent,
                    F.normalize(push_direction + preferred_tangent, dim=-1),
                    F.normalize(push_direction - preferred_tangent, dim=-1),
                ])
                templates = []
                for radius in (0.34, 0.14):
                    templates.append(((block[None] + radius * directions - agent[None]) / 0.39).clamp(-1.0, 1.0))
                template_actions = torch.cat(templates, dim=0)
                candidates[: min(len(template_actions), args.cem_samples)] = template_actions[: args.cem_samples]
                phase_templates = template_actions
            if contact_geometry is not None:
                _, block, goal_block = contact_geometry
                xy_error = float(mask_moment_task_distance(goal_block, block).item())
            else:
                xy_error = float(torch.linalg.vector_norm(raw_current[0, 2:4] - goal_pose[0, :2]).item())
            phase_distance = float(getattr(args, "orientation_phase_distance", 35.0))
            if bool(getattr(args, "reversible_orientation_phase", False)):
                orientation_phase_latched = xy_error <= phase_distance
            phase_active = model.utility_mode == "phase" and (orientation_phase_latched or xy_error <= phase_distance)
            if phase_active:
                if xy_error <= phase_distance and not bool(getattr(args, "reversible_orientation_phase", False)):
                    setattr(args, "_orientation_phase_latched", True)
                candidates = candidates.clamp(
                    -float(getattr(args, "orientation_action_max", 1.0)),
                    float(getattr(args, "orientation_action_max", 1.0)),
                )
                if bool(getattr(args, "phase_template_only", False)) and phase_templates is not None:
                    repeats = int(math.ceil(args.cem_samples / len(phase_templates)))
                    candidates = phase_templates.repeat((repeats, 1))[: args.cem_samples]
            components = model.score_action_components(
                current_state.expand(args.cem_samples, -1), normalized_goal_pose.expand(args.cem_samples, -1), candidates,
                temporal_context.expand(args.cem_samples, -1) if temporal_context is not None else None,
                previous_action.expand(args.cem_samples, -1),
            )
            use_successor_phase = bool(getattr(args, "successor_orientation_phase", False)) and phase_active
            if bool(getattr(args, "task_space_objective", False)) or use_successor_phase:
                predicted_next = model.predict_next(
                    current_state.expand(args.cem_samples, -1), candidates,
                    temporal_context=temporal_context.expand(args.cem_samples, -1) if temporal_context is not None else None,
                    previous_action=previous_action.expand(args.cem_samples, -1),
                )
                predicted_raw = raw_from_normalized(predicted_next, mean, std)
                if bool(getattr(args, "task_space_objective", False)):
                    scores = task_space_action_cost(
                        predicted_raw,
                        goal_pose.expand(args.cem_samples, -1),
                        raw_current.expand(args.cem_samples, -1),
                    )
                else:
                    predicted_orientation = F.normalize(predicted_raw[:, 4:6], dim=-1)
                    goal_orientation = F.normalize(goal_pose.expand(args.cem_samples, -1)[:, 2:4], dim=-1)
                    orientation_cost = 0.5 * (predicted_orientation - goal_orientation).square().sum(dim=-1)
                    block_cost = torch.linalg.vector_norm(
                        predicted_raw[:, 2:4] - goal_pose.expand(args.cem_samples, -1)[:, :2], dim=-1,
                    ) / PUSHT_RENDER_SIZE
                    scores = orientation_cost + float(getattr(args, "orientation_translation_weight", 0.03)) * block_cost
            elif model.utility_mode == "scalar":
                scores = components[:, 0]
            else:
                scores = components[:, 0] if not phase_active else components[:, 1] + float(getattr(args, "orientation_translation_weight", 0.03)) * components[:, 0]
            if model.adapter_architecture == "mode_uncertain":
                predicted_next, log_variance = model.predict_distribution(
                    current_state.expand(args.cem_samples, -1), candidates,
                    temporal_context=temporal_context.expand(args.cem_samples, -1) if temporal_context is not None else None,
                    previous_action=previous_action.expand(args.cem_samples, -1),
                )
                uncertainty_risk = torch.exp(log_variance).mean(dim=-1)
                logits = model.predict_contact_logits(
                    current_state.expand(args.cem_samples, -1), candidates,
                    temporal_context.expand(args.cem_samples, -1) if temporal_context is not None else None,
                    previous_action.expand(args.cem_samples, -1),
                )
                separation_risk = torch.softmax(logits, dim=-1)[:, 3]
                scores = scores + float(getattr(args, "uncertainty_risk_weight", 0.25)) * uncertainty_risk
                scores = scores + float(getattr(args, "separation_risk_weight", 0.5)) * separation_risk
                # Once visual geometry says the object is inside the terminal
                # XY tube, penalize candidates whose learned successor leaves
                # it.  This is a model-predicted constraint, never a queried
                # simulator successor.
                if phase_active:
                    predicted_raw = raw_from_normalized(predicted_next, mean, std)
                    predicted_xy = torch.linalg.vector_norm(predicted_raw[:, 2:4] - goal_pose[:, :2], dim=-1)
                    scores = scores + float(getattr(args, "contact_preservation_weight", 1.0)) * F.relu(predicted_xy - 35.0) / 35.0
            elite = candidates[torch.topk(scores, k=max(8, args.cem_samples // 12), largest=False).indices]
            action_mean = elite.mean(dim=0)
            action_std = elite.std(dim=0).clamp(0.04, 0.65)
        return action_mean.detach().cpu().numpy().astype(np.float32)
    samples, horizon = args.cem_samples, args.planning_horizon
    action_mean = torch.zeros(horizon, 2, device=device)
    action_std = torch.full((horizon, 2), 0.55, device=device)
    for _ in range(args.cem_iterations):
        candidates = torch.randn(samples, horizon, 2, device=device, generator=generator) * action_std + action_mean
        candidates = candidates.clamp(-1.0, 1.0)
        states = current_state.expand(samples, -1)
        for step in range(horizon):
            states = model.predict_next(
                states, candidates[:, step],
                temporal_context=temporal_context.expand(samples, -1) if temporal_context is not None else None,
                previous_action=previous_action.expand(samples, -1),
            )
        raw = raw_from_normalized(states, mean, std)
        xy = ((raw[:, 2:4] - goal_pose[:, 0:2]) ** 2).sum(dim=-1)
        orientation = ((F.normalize(raw[:, 4:6], dim=-1) - F.normalize(goal_pose[:, 2:4], dim=-1)) ** 2).sum(dim=-1)
        desired_agent = raw[:, 2:4] - 42.0 * F.normalize(goal_pose[:, 0:2] - raw[:, 2:4], dim=-1)
        agent = ((raw[:, :2] - desired_agent) ** 2).sum(dim=-1)
        cost = xy + 2400.0 * orientation + 0.04 * agent + 1.5 * (candidates.square().sum(dim=(1, 2)))
        elite = candidates[torch.topk(cost, k=max(8, samples // 12), largest=False).indices]
        action_mean = elite.mean(dim=0)
        action_std = elite.std(dim=0).clamp(0.04, 0.65)
    return action_mean[0].detach().cpu().numpy().astype(np.float32)


def strict_success(state: np.ndarray, goal: np.ndarray) -> bool:
    vector = state_vector(state)
    target = state_vector(goal)
    return bool(np.linalg.norm(vector[2:4] - target[2:4]) <= 20.0 and angle_error_from_sincos(vector, target) <= math.pi / 9.0)


def evaluate_visual_planning(
    env: gym.Env, model: VisualCausalWorldModel, normalizer: Normalization,
    args: argparse.Namespace, orientation_model: VisualCausalWorldModel | None = None,
    orientation_normalizer: Normalization | None = None,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 999)
    rows = []
    for episode in range(args.planning_episodes):
        args._orientation_phase_latched = False
        args._visual_contact_established = False
        observation, info = env.reset(seed=args.seed + 100_000 + episode)
        goal_image = np.asarray(info["goal"], dtype=np.uint8).copy()
        goal = task_goal_state(info["goal_state"], info["goal_pose"])
        trace = []
        state = np.asarray(observation["state"], dtype=np.float32).copy()
        previous_image = env.render().copy()
        args._previous_action = np.zeros(2, dtype=np.float32)
        for step in range(args.planning_steps):
            # The next line is the complete runtime interface: rendered image
            # and goal image in, action out. State below is evaluator-only.
            image = env.render().copy()
            visual_xy = None
            if orientation_model is not None:
                if not model.uses_mask_moments or orientation_normalizer is None:
                    raise ValueError("orientation expert requires mask-moment RGB geometry and a normalizer")
                current = tensor_images(resize_image(image, args.image_size)[None], args.device)
                goal_tensor = tensor_images(resize_image(goal_image, args.image_size)[None], args.device)
                current_features = model.encoder.features(current)
                goal_features = model.encoder.features(goal_tensor)
                visual_xy = float(mask_moment_task_distance(current_features[0, 11:13], goal_features[0, 11:13]).item())
                phase_now = visual_xy <= float(getattr(args, "orientation_phase_distance", 35.0))
                if bool(getattr(args, "reversible_orientation_phase", False)):
                    args._orientation_phase_latched = phase_now
                elif phase_now:
                    args._orientation_phase_latched = True
                action_model, action_normalizer = (
                    (orientation_model, orientation_normalizer)
                    if bool(getattr(args, "_orientation_phase_latched", False)) else (model, normalizer)
                )
                action_repeat = (
                    int(getattr(args, "orientation_action_repeat", args.action_repeat))
                    if action_model is orientation_model else int(args.action_repeat)
                )
                action_source = "orientation_expert" if action_model is orientation_model else "base"
            else:
                action_model, action_normalizer = model, normalizer
                action_repeat = int(args.action_repeat)
                action_source = "base"
            action = visual_plan_action(
                action_model, image, previous_image, goal_image, action_normalizer, args, generator,
            )
            terminated = truncated = False
            for _ in range(action_repeat):
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            previous_image = image
            args._previous_action = np.asarray(action, dtype=np.float32).copy()
            state = np.asarray(observation["state"], dtype=np.float32).copy()
            trace.append({
                "step": step,
                "action": action.tolist(),
                "action_repeat": int(action_repeat),
                "action_source": action_source,
                "orientation_phase_latched": bool(getattr(args, "_orientation_phase_latched", False)),
                "visual_block_xy_before_action": visual_xy,
                "block_xy": float(np.linalg.norm(state[2:4] - goal[2:4])),
                "angle": angle_error_from_sincos(state_vector(state), state_vector(goal)),
            })
            if strict_success(state, goal) or terminated or truncated:
                break
        rows.append({
            "episode": episode,
            "strict_success": strict_success(state, goal),
            "steps": len(trace),
            "block_xy_distance": float(np.linalg.norm(state[2:4] - goal[2:4])),
            "angle_error": angle_error_from_sincos(state_vector(state), state_vector(goal)),
            "trace": trace,
        })
    return rows


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.action_repeat < 1:
        raise ValueError("action_repeat must be positive")
    env = gym.make(
        "swm/PushT-v1", render_mode="rgb_array",
        max_episode_steps=args.planning_steps * args.action_repeat,
    )
    try:
        data_seed = args.seed if args.data_seed is None else int(args.data_seed)
        if not 0.0 <= args.terminal_root_fraction <= 1.0 or not 0.0 <= args.bootstrap_root_fraction <= 1.0:
            raise ValueError("terminal_root_fraction and bootstrap_root_fraction must lie in [0, 1]")
        if args.terminal_root_fraction + args.bootstrap_root_fraction > 1.0:
            raise ValueError("terminal_root_fraction + bootstrap_root_fraction must not exceed 1")
        if args.bootstrap_root_fraction > 0.0 and args.bootstrap_checkpoint is None:
            raise ValueError("bootstrap_root_fraction requires --bootstrap-checkpoint")
        train_terminal_count = int(round(args.train_roots * args.terminal_root_fraction))
        val_terminal_count = int(round(args.val_roots * args.terminal_root_fraction))
        train_bootstrap_count = int(round(args.train_roots * args.bootstrap_root_fraction))
        val_bootstrap_count = int(round(args.val_roots * args.bootstrap_root_fraction))
        train_roots = balanced_source_cases(
            env, args.train_roots - train_terminal_count - train_bootstrap_count, args.root_rollout_steps, data_seed,
            args.contact_root_fraction, args.contact_approach_steps,
        )
        train_roots.extend(terminal_pose_cases(env, train_terminal_count, data_seed + 400_000))
        if train_bootstrap_count:
            train_roots.extend(bootstrap_closed_loop_source_cases(
                env, args.bootstrap_checkpoint, train_bootstrap_count, data_seed + 500_000,
                args.bootstrap_rollout_episodes, args.bootstrap_rollout_steps,
                args.bootstrap_contact_distance, args.device,
            ))
        val_roots = balanced_source_cases(
            env, args.val_roots - val_terminal_count - val_bootstrap_count, args.root_rollout_steps, data_seed + 10_000,
            args.contact_root_fraction, args.contact_approach_steps,
        )
        val_roots.extend(terminal_pose_cases(env, val_terminal_count, data_seed + 410_000))
        if val_bootstrap_count:
            val_roots.extend(bootstrap_closed_loop_source_cases(
                env, args.bootstrap_checkpoint, val_bootstrap_count, data_seed + 510_000,
                args.bootstrap_rollout_episodes, args.bootstrap_rollout_steps,
                args.bootstrap_contact_distance, args.device,
            ))
        train = make_counterfactual_dataset(
            env, train_roots, args.image_size, data_seed + 20_000, args.action_repeat, args.counterfactual_fan_size,
            args.train_action_support, args.train_action_max_magnitude,
        )
        val = make_counterfactual_dataset(
            env, val_roots, args.image_size, data_seed + 30_000, args.action_repeat, args.counterfactual_fan_size,
            args.train_action_support, args.train_action_max_magnitude,
        )
        print(json.dumps({"phase": "dataset", "train_rows": len(train["action"]), "val_rows": len(val["action"])}), flush=True)
        model, normalizer, training, snapshots = train_model(train, val, args)
        print(json.dumps({"phase": "trained", "best_val_loss": training["best_val_loss"]}), flush=True)
        metrics = physical_metrics(model, val, normalizer, args.device)
        print(json.dumps({"phase": "physical_metrics", "action_top1": metrics["paired_action_motion_top1"]}), flush=True)
        planning = [] if args.skip_planning else evaluate_visual_planning(env, model, normalizer, args)
        print(json.dumps({"phase": "planning_complete", "episodes": len(planning)}), flush=True)
    finally:
        env.close()
    split_manifest = {"train": source_split_manifest(train_roots), "validation": source_split_manifest(val_roots)}
    if set(tuple(row["state"]) for row in split_manifest["train"]["roots"]) & set(tuple(row["state"]) for row in split_manifest["validation"]["roots"]):
        raise RuntimeError("source-root overlap between train and validation splits")
    (output / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True) + "\n")
    torch.save({"model": model.state_dict(), "normalization": asdict(normalizer), "args": vars(args)}, output / "causal_visual_world_model.pt")
    snapshots_dir = output / "snapshots"
    for snapshot in snapshots:
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": snapshot["model"], "normalization": asdict(normalizer), "args": vars(args),
                "snapshot": {"epoch": snapshot["epoch"], "validation_loss": snapshot["validation_loss"]},
            },
            snapshots_dir / f"epoch_{snapshot['epoch']:04d}.pt",
        )
    success = np.asarray([row["strict_success"] for row in planning], dtype=np.float64)
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "training_rows": int(len(train["action"])),
            "validation_rows": int(len(val["action"])),
            "counterfactual_actions_per_visual_source": ["+a", "-a", "zero", f"goal_relative_fan_{args.counterfactual_fan_size}"],
            "physical_steps_per_model_action": int(args.action_repeat),
            "source_train_validation_disjoint": True,
            "source_split_manifest": str((output / "split_manifest.json").resolve()),
            "train_action_support": args.train_action_support,
            "train_action_max_magnitude": float(args.train_action_max_magnitude),
            "policy_runtime_input": "previous/current rendered image and goal image only" if args.temporal_context else "current rendered image and goal image only",
            "policy_runtime_temporal_context": bool(args.temporal_context),
            "policy_runtime_state_access": False,
            "runtime_successor_verification": False,
            "physical_state_role": "offline labels and evaluator only",
            "wall_seconds": time.perf_counter() - started,
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
        "training": training,
        "held_out_counterfactual_metrics": metrics,
        "visual_online_planning": {
            "episodes": planning,
            "strict_success_count": int(success.sum()),
            "strict_success_rate": float(success.mean()) if len(success) else None,
            "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in planning])) if planning else None,
            "mean_angle_error": float(np.mean([row["angle_error"] for row in planning])) if planning else None,
        },
        "sanity_checks": {
            "finite_counterfactual_predictions": metrics["finite"],
            "paired_roots_match_validation": metrics["paired_roots"] == args.val_roots,
            "visual_runtime_only": True,
        },
        "claim_boundary": "This is a newly trained, intervention-supervised visual world model. It is not evidence about the frozen released LeWM checkpoint and does not alone establish universal cross-environment world-model capability or a universal commutator certificate.",
    }
    (output / "causal_visual_world_model.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n"
    )
    (output / "causal_visual_world_model.md").write_text(
        "# Intervention-supervised visual PushT world model\n\n"
        f"- Held-out paired action-effect error: {metrics['paired_plus_minus_effect_error']:.6g}.\n"
        f"- Held-out action-motion top-1: {metrics['paired_action_motion_top1']:.3f}.\n"
        f"- Visual-only strict success: {int(success.sum())}/{len(success)}.\n"
        f"- Mean terminal block XY: {report['visual_online_planning']['mean_block_xy_distance'] if planning else 'not run'}.\n"
        f"- Mean terminal angle error: {report['visual_online_planning']['mean_angle_error'] if planning else 'not run'}.\n\n"
        "Runtime policy input is image + goal image only; physical state is offline supervision/evaluator-only.\n"
    )
    print(json.dumps({
        "output": str(output), "counterfactual": metrics,
        "planning": {key: value for key, value in report["visual_online_planning"].items() if key != "episodes"},
    }, indent=2))


if __name__ == "__main__":
    main()
