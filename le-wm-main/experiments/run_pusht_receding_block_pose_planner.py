"""Matched oracle comparison: joint endpoint CEM vs receding block-pose CEM.

The negative planning result was caused by coupling two different control
objectives in one endpoint: block pose must be closed through contact, while
the agent should be returned only after contact is finished.  This diagnostic
keeps the simulator privileged and compares the coupled baseline with a
receding-horizon repair on exactly the same stable-contact cases.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_obstacle_aware_local_effect import (
    angle_error,
    block_pose_success,
    block_xy_cost,
    jsonable,
    route_to_contact,
    return_agent,
    task_success,
)
from experiments.run_pusht_repaired_predictor_mpc import (
    staged_proposal_sequences,
    stable_contact_bootstrap,
)

gym.logger.min_level = 40


def make_state_only_env() -> gym.Env:
    """Create PushT without rendering a pygame frame on every reset.

    The planner arms in this module consume simulator state only.  The
    upstream environment nevertheless renders the goal image during every
    reset, which makes a large matched CEM run dominated by pygame overhead.
    The stub preserves the reset contract for this state-only diagnostic; it
    must not be used by visual-data collection experiments.
    """
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    env.unwrapped.render = lambda: np.zeros((1, 1, 3), dtype=np.uint8)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8421)
    parser.add_argument("--bootstrap-steps", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument(
        "--phase-action-block",
        type=int,
        default=2,
        help="number of selected actions executed before replanning in staged_phase",
    )
    parser.add_argument(
        "--phase-patience",
        type=int,
        default=4,
        help="stagnant staged steps before contact recovery",
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--cem-iterations", type=int, default=2)
    parser.add_argument(
        "--push-rescue-population",
        type=int,
        default=512,
        help="high-budget push-phase population used only when the baseline push phase misses block closure",
    )
    parser.add_argument(
        "--push-rescue-iterations",
        type=int,
        default=8,
        help="high-budget push-phase CEM iterations used only after a baseline push failure",
    )
    parser.add_argument(
        "--push-rescue-restarts",
        type=int,
        default=2,
        help="independent push-phase rescue searches after a baseline push failure",
    )
    parser.add_argument("--noise", type=float, default=0.18)
    parser.add_argument("--return-clearance", type=float, default=140.0)
    parser.add_argument("--return-steps", type=int, default=45)
    parser.add_argument(
        "--terminal-stability-steps",
        type=int,
        default=1,
        help=(
            "independent zero-action hold steps required after a candidate "
            "meets the strict joint criterion"
        ),
    )
    parser.add_argument(
        "--closure-return-population",
        type=int,
        default=256,
        help="simulator candidate population for the block-closed terminal return phase",
    )
    parser.add_argument(
        "--closure-return-iterations",
        type=int,
        default=2,
        help="CEM iterations for the block-closed terminal return phase",
    )
    parser.add_argument(
        "--closure-return-horizon",
        type=int,
        default=8,
        help="number of one-step actions searched for a closure-preserving return",
    )
    parser.add_argument(
        "--closure-return-rescue-population",
        type=int,
        default=512,
        help="high-budget population used only when the low-budget return search misses full joint success",
    )
    parser.add_argument(
        "--closure-return-rescue-iterations",
        type=int,
        default=8,
        help="high-budget CEM iterations used only for a failed low-budget return search",
    )
    parser.add_argument(
        "--closure-return-rescue-restarts",
        type=int,
        default=2,
        help="independent high-budget random restarts for a failed low-budget return search",
    )
    parser.add_argument(
        "--closure-return-path-xy-margin",
        type=float,
        default=1.0,
        help="small bounded intermediate XY tube beyond the strict 20-pixel endpoint tolerance",
    )
    parser.add_argument(
        "--closure-return-path-angle-margin",
        type=float,
        default=0.02,
        help="small bounded intermediate angle tube beyond the strict endpoint tolerance",
    )
    parser.add_argument(
        "--closure-return-deep-population",
        type=int,
        default=1024,
        help="deep terminal-return population used only after ordinary return rescue fails",
    )
    parser.add_argument(
        "--closure-return-deep-iterations",
        type=int,
        default=16,
        help="deep terminal-return CEM iterations used only after ordinary return rescue fails",
    )
    parser.add_argument(
        "--closure-return-deep-horizon",
        type=int,
        default=24,
        help="deep terminal-return horizon used only after ordinary return rescue fails",
    )
    parser.add_argument(
        "--closure-return-deep-restarts",
        type=int,
        default=2,
        help="independent deep terminal-return restarts after ordinary return rescue fails",
    )
    parser.add_argument("--recovery-clearance", type=float, default=140.0)
    parser.add_argument("--recovery-contact-distance", type=float, default=42.0)
    parser.add_argument("--recovery-settle-steps", type=int, default=6)
    parser.add_argument("--recovery-patience", type=int, default=4)
    force_group = parser.add_mutually_exclusive_group()
    force_group.add_argument(
        "--enable-force-fan",
        dest="disable_force_fan",
        action="store_false",
        help="opt in to independent tangential/force-direction proposals",
    )
    # Keep the historical flag as a backwards-compatible explicit spelling.
    force_group.add_argument(
        "--disable-force-fan",
        dest="disable_force_fan",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(disable_force_fan=True)
    parser.add_argument(
        "--force-fan-xy-threshold",
        type=float,
        default=60.0,
        help="activate the force-direction fan only below this block XY error",
    )
    parser.add_argument(
        "--force-fan-near-goal-only",
        action="store_true",
        help="gate the force-direction fan by near-goal XY and angle error",
    )
    parser.add_argument(
        "--force-fan-on-stagnation",
        action="store_true",
        help="activate the force-direction fan only after observed no-motion",
    )
    parser.add_argument(
        "--disable-adaptive-angle",
        action="store_true",
        help="ablate the near-goal increase in orientation weight",
    )
    parser.add_argument(
        "--disable-reacquisition",
        action="store_true",
        help="ablate bounded multi-normal contact reacquisition",
    )
    parser.add_argument(
        "--cost-mode",
        choices=("weighted_pose", "task_margin"),
        default="weighted_pose",
        help="receding objective; task_margin normalizes XY and angle termination margins",
    )
    parser.add_argument(
        "--hard-contact-gate",
        action="store_true",
        help="during receding CEM, require contact acquisition before block-pose optimization and reject no-motion candidates",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("joint_endpoint", "receding_block_pose", "staged_phase"),
        default=("joint_endpoint", "receding_block_pose", "staged_phase"),
        help="matched arms to execute; use a subset for focused follow-up runs",
    )
    return parser.parse_args()


def joint_cost(state: np.ndarray, goal: np.ndarray) -> float:
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return float(
        np.linalg.norm(state[:4] - goal[:4])
        + 45.0 * angle_error(state[4], goal[4])
    )


def block_pose_cost(
    state: np.ndarray,
    goal: np.ndarray,
    adaptive_angle: bool = True,
    cost_mode: str = "weighted_pose",
) -> float:
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    xy_error = block_xy_cost(state, goal)
    angle = angle_error(state[4], goal[4])
    if cost_mode == "task_margin":
        # PushT success is a conjunction, not a weighted average.  Normalize
        # both block-pose errors by their termination margins and optimize the
        # worst normalized violation, with a small tie-break term.
        xy_margin = xy_error / 20.0
        angle_margin = angle / (np.pi / 9.0)
        return float(max(xy_margin, angle_margin) + 0.01 * (xy_margin + angle_margin))
    # Once translation is already close, the remaining task is primarily a
    # torque/contact problem.  The previous fixed weight made a few pixels of
    # XY improvement dominate a much larger orientation error, so CEM kept
    # selecting no-motion candidates after the block had reached the goal
    # region.  Preserve a translation-first phase while making the terminal
    # orientation phase explicit.
    angle_weight = 90.0 if adaptive_angle and xy_error < 35.0 else 45.0
    return float(xy_error + angle_weight * angle)


def gated_block_pose_cost(
    final: np.ndarray,
    current: np.ndarray,
    goal: np.ndarray,
    args: argparse.Namespace,
) -> float:
    """Score the block while preventing free-space endpoint collapse.

    The ordinary block-pose score is appropriate after contact, but before
    contact it cannot distinguish a useful approach from a hold action because
    both leave the block unchanged.  This gate makes contact acquisition an
    explicit prerequisite and then excludes no-motion candidates while the
    pose remains open.  It is a control-interface repair, not privileged
    physical prediction.
    """
    pose = block_pose_cost(
        final,
        goal,
        adaptive_angle=not args.disable_adaptive_angle,
        cost_mode=args.cost_mode,
    )
    current_distance = float(np.linalg.norm(current[:2] - current[2:4]))
    final_distance = float(np.linalg.norm(final[:2] - final[2:4]))
    block_motion = float(np.linalg.norm(final[2:4] - current[2:4]))
    if current_distance > args.recovery_contact_distance:
        # Acquire contact first; leave pose refinement for the next replanning
        # step.  A small pose term prevents routes that gratuitously drag the T.
        return float(2.0 * final_distance + 0.10 * pose + 0.5 * block_motion)
    if (
        block_motion < 0.5
        and (block_xy_cost(current, goal) > 20.0 or angle_error(current[4], goal[4]) > np.pi / 9.0)
    ):
        return float(pose + 1000.0)
    return float(pose)


def rollout(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, bool]:
    env.reset(seed=0, options={"state": np.asarray(state).copy(), "goal_state": np.asarray(goal).copy()})
    observation = None
    terminated = False
    for action in np.asarray(actions, dtype=np.float32):
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    if observation is None:
        raise RuntimeError("rollout produced no observation")
    return np.asarray(observation["state"], dtype=np.float64).copy(), bool(terminated)


def rollout_trajectory(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Roll out a candidate and retain all states for phase-aware scoring."""
    env.reset(seed=0, options={"state": np.asarray(state).copy(), "goal_state": np.asarray(goal).copy()})
    current = np.asarray(state, dtype=np.float64).copy()
    states = [current.copy()]
    terminated = False
    for action in np.asarray(actions, dtype=np.float32):
        observation, _, terminated, truncated, _ = env.step(action)
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        states.append(current.copy())
        if terminated or truncated:
            break
    return current, np.asarray(states, dtype=np.float64), bool(terminated)


def candidate_bank(
    rng: np.random.Generator,
    state: np.ndarray,
    goal: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    bank = staged_proposal_sequences(
        rng,
        state,
        goal,
        args.population,
        args.horizon,
        1,
        args.noise,
    )
    # The original staged bank mostly perturbs the desired translation normal.
    # That is insufficient when the current agent is already on the wrong
    # side of the T.  Add an explicit contact-direction fan so the receding
    # planner can reacquire a usable contact manifold instead of repeatedly
    # selecting free-space/no-motion candidates.
    state = np.asarray(state, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    agent, block = state[:2], state[2:4]
    direction = goal[2:4] - block
    direction /= max(np.linalg.norm(direction), 1e-8)
    offsets = (0.0, -np.pi / 6.0, np.pi / 6.0, -np.pi / 3.0,
               np.pi / 3.0, -np.pi / 2.0, np.pi / 2.0, np.pi)
    templates = []
    angle_delta = float((goal[4] - state[4] + np.pi) % (2.0 * np.pi) - np.pi)
    for offset in offsets:
        c, s = np.cos(offset), np.sin(offset)
        push_direction = np.asarray([
            c * direction[0] - s * direction[1],
            s * direction[0] + c * direction[1],
        ])
        contact_target = block - 48.0 * push_direction
        approach = np.clip((contact_target - agent) / 180.0, -0.75, 0.75)
        push = np.clip((block + 86.0 * push_direction - agent) / 100.0, -1.0, 1.0)
        for approach_blocks in (2, 3, 4):
            sequence = np.tile(push, (args.horizon, 1)).astype(np.float32)
            sequence[: min(approach_blocks, args.horizon)] = approach
            templates.append(sequence)

        force_fan_active = not getattr(args, "disable_force_fan", False)
        if getattr(args, "force_fan_on_stagnation", False):
            force_fan_active = bool(getattr(args, "_force_fan_active", False))
        if getattr(args, "force_fan_near_goal_only", False):
            force_fan_active = (
                force_fan_active
                and block_xy_cost(state, goal)
                < getattr(args, "force_fan_xy_threshold", 60.0)
                and angle_error(state[4], goal[4]) > np.pi / 9.0
            )
        if not force_fan_active:
            continue
        # Translation direction alone does not span the contact modes needed
        # to rotate the T.  Enumerate force directions independently from the
        # approach normal: tangential pushes can create torque while keeping
        # the block near its target, and the sign of the desired angle change
        # gives a useful ordering of the two tangential directions.  The
        # simulator still scores every candidate; these are only structured
        # proposals, not privileged execution decisions.
        tangential_sign = 1.0 if angle_delta >= 0.0 else -1.0
        force_directions = (
            push_direction,
            np.asarray([-push_direction[1], push_direction[0]]),
            np.asarray([push_direction[1], -push_direction[0]]),
            np.asarray([
                np.cos(tangential_sign * np.pi / 4.0) * push_direction[0]
                - np.sin(tangential_sign * np.pi / 4.0) * push_direction[1],
                np.sin(tangential_sign * np.pi / 4.0) * push_direction[0]
                + np.cos(tangential_sign * np.pi / 4.0) * push_direction[1],
            ]),
        )
        for force_direction in force_directions[1:]:
            tangential_push = np.clip(
                (block + 72.0 * force_direction - agent) / 100.0,
                -0.9,
                0.9,
            )
            for approach_blocks in (2, 3):
                sequence = np.tile(tangential_push, (args.horizon, 1)).astype(np.float32)
                sequence[: min(approach_blocks, args.horizon)] = approach
                templates.append(sequence)
    # Keep the original staged proposals as the warm-start prefix; append the
    # contact fan as coverage rather than allowing it to replace the bank's
    # empirically useful approach--push templates.
    fan_count = min(len(templates), len(bank))
    # Subsample uniformly when the expanded force-direction fan is larger
    # than the population, so both normal and tangential modes survive.
    template_indices = np.linspace(0, len(templates) - 1, fan_count).round().astype(int)
    bank[-fan_count:] = np.asarray(
        [templates[index] for index in template_indices], dtype=np.float32
    )
    # Near the target, large translational pushes can destroy an already good
    # XY placement while correcting orientation.  Add a small-action bank for
    # tangential/fine pose corrections.
    if block_xy_cost(state, goal) < 45.0:
        fine_templates = []
        for offset in (-np.pi / 2.0, -np.pi / 3.0, -np.pi / 6.0,
                       np.pi / 6.0, np.pi / 3.0, np.pi / 2.0):
            c, s = np.cos(offset), np.sin(offset)
            fine_direction = np.asarray([
                c * direction[0] - s * direction[1],
                s * direction[0] + c * direction[1],
            ])
            for scale in (24.0, 38.0):
                fine_push = np.clip(
                    (block + scale * fine_direction - agent) / 100.0,
                    -0.55,
                    0.55,
                )
                sequence = np.zeros((args.horizon, 2), dtype=np.float32)
                sequence[max(0, args.horizon // 2):] = fine_push
                fine_templates.append(sequence)
        fine_count = min(len(fine_templates), len(bank))
        bank[-fine_count:] = np.asarray(fine_templates[:fine_count], dtype=np.float32)
    return bank


def plan_one_shot(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The coupled baseline: optimize a single terminal joint endpoint."""
    bank = candidate_bank(rng, state, goal, args)
    mean = bank.mean(axis=0)
    std = np.maximum(bank.std(axis=0), 0.12)
    best_value = float("inf")
    best_actions = bank[0].copy()
    total_rollouts = 0
    for iteration in range(args.cem_iterations):
        candidates = np.clip(
            mean[None] + std[None] * rng.normal(size=bank.shape), -1.0, 1.0
        ).astype(np.float32)
        candidates[: min(16, len(candidates))] = bank[: min(16, len(candidates))]
        values = []
        for candidate in candidates:
            final, _ = rollout(env, state, goal, candidate)
            values.append(joint_cost(final, goal))
        total_rollouts += len(candidates)
        values_array = np.asarray(values, dtype=np.float64)
        elite = candidates[np.argsort(values_array)[: max(8, len(candidates) // 8)]]
        mean = 0.35 * mean + 0.65 * elite.mean(axis=0)
        std = 0.35 * std + 0.65 * np.maximum(elite.std(axis=0), 0.05)
        index = int(np.argmin(values_array))
        if float(values_array[index]) < best_value:
            best_value = float(values_array[index])
            best_actions = candidates[index].copy()
        bank = candidates
    return best_actions, {
        "objective": "joint_endpoint",
        "objective_value": best_value,
        "simulator_rollouts": total_rollouts,
    }


def probe_contact_routes(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Try several contact normals from the same state and keep the best one."""
    base_direction = np.asarray(goal[2:4] - state[2:4], dtype=np.float64)
    base_direction /= max(np.linalg.norm(base_direction), 1e-8)
    offsets = (0.0, -np.pi / 3.0, np.pi / 3.0, -np.pi / 2.0,
               np.pi / 2.0, np.pi)
    base_contact_distance = float(args.recovery_contact_distance)
    contact_distances = [base_contact_distance]
    if getattr(args, "selection_mode", "pose") == "contact_reacquisition_verified":
        # A single Euclidean stop distance does not identify a usable contact
        # manifold for a non-convex T. Add one closer target and let the
        # simulator probe decide which route actually supports a block effect.
        contact_distances.append(max(12.0, 0.55 * base_contact_distance))
    candidates = []
    current_xy_cost = block_xy_cost(state, goal)
    for offset in offsets:
        c, s = np.cos(offset), np.sin(offset)
        direction = np.asarray([
            c * base_direction[0] - s * base_direction[1],
            s * base_direction[0] + c * base_direction[1],
        ])
        for contact_distance in contact_distances:
            recovery_args = argparse.Namespace(
                clearance=args.recovery_clearance,
                contact_distance=contact_distance,
                settle_steps=args.recovery_settle_steps,
            )
            recovered, info = route_to_contact(
                env, state, goal, direction, recovery_args
            )
            candidates.append({
                "offset": float(offset),
                "contact_distance_target": float(contact_distance),
                "direction": direction,
                "state": recovered,
                "pose_cost": block_pose_cost(
                    recovered,
                    goal,
                    adaptive_angle=not args.disable_adaptive_angle,
                    cost_mode=args.cost_mode,
                ),
                "block_motion": float(np.linalg.norm(recovered[2:4] - state[2:4])),
                "xy_progress": float(current_xy_cost - block_xy_cost(recovered, goal)),
                "block_delta": np.asarray(recovered[2:4] - state[2:4], dtype=np.float64),
                "contact_distance": float(np.linalg.norm(recovered[:2] - recovered[2:4])),
                "info": info,
            })
    # Keep ordinary route selection pose-oriented: the expanded force-
    # direction fan in the receding CEM is responsible for choosing a
    # productive push.  Near the goal, however, a useful recovery may need to
    # preserve the block pose temporarily while moving the agent to a new
    # contact normal.  The explicit ``contact_reacquisition`` mode selects
    # the closest safe contact candidate, while retaining a bounded pose
    # envelope for the caller's acceptance test.
    selection_mode = getattr(args, "selection_mode", "pose")
    if selection_mode in (
        "contact_reacquisition",
        "contact_reacquisition_verified",
    ):
        reacquisition_xy_margin = float(
            getattr(args, "reacquisition_xy_margin", 10.0)
        )
        xy_limit = current_xy_cost + reacquisition_xy_margin
        eligible = [
            item for item in candidates
            if block_xy_cost(item["state"], goal) <= xy_limit
        ]
        if selection_mode == "contact_reacquisition_verified":
            # Proximity to the block is not equivalent to a usable pushing
            # mode for the T-shaped object. Probe each routed state with a
            # small simulator-only fan and prefer a route whose successor
            # improves the block pose. This remains an explicit privileged
            # recovery diagnostic, not a learned-effect prediction.
            for item in candidates:
                probe_state = np.asarray(item["state"], dtype=np.float64)
                block = probe_state[2:4]
                agent = probe_state[:2]
                desired = np.asarray(goal[2:4] - block, dtype=np.float64)
                desired /= max(float(np.linalg.norm(desired)), 1e-8)
                angle_delta = float(
                    (goal[4] - probe_state[4] + np.pi) % (2.0 * np.pi) - np.pi
                )
                rotation_sign = 1.0 if angle_delta >= 0.0 else -1.0
                probe_actions = [np.zeros(2, dtype=np.float32)]
                offsets = np.linspace(-np.pi, np.pi, 24, endpoint=False)
                for magnitude in (0.12, 0.20, 0.35):
                    for offset in offsets:
                        probe_actions.append(
                            np.asarray(
                                [
                                    magnitude * np.cos(offset),
                                    magnitude * np.sin(offset),
                                ],
                                dtype=np.float32,
                            )
                        )
                for offset in offsets:
                    c, s = np.cos(offset), np.sin(offset)
                    force = np.asarray(
                        [
                            c * desired[0] - s * desired[1],
                            s * desired[0] + c * desired[1],
                        ]
                    )
                    if rotation_sign < 0.0:
                        force = np.asarray([force[0], -force[1]])
                    target = block + 82.0 * force
                    probe_actions.append(
                        np.clip((target - agent) / 100.0, -1.0, 1.0).astype(
                            np.float32
                        )
                        * (0.55 / 0.55)
                    )
                probe_actions = np.asarray(probe_actions, dtype=np.float32)
                source_pose = float(item["pose_cost"])
                source_xy = float(block_xy_cost(item["state"], goal))
                source_angle = angle_error(item["state"][4], goal[4])
                probe_rows = []
                for action in probe_actions:
                    env.reset(
                        seed=0,
                        options={
                            "state": np.asarray(item["state"], dtype=np.float64).copy(),
                            "goal_state": np.asarray(goal, dtype=np.float64).copy(),
                        },
                    )
                    observation, _, terminated, truncated, _ = env.step(action)
                    successor = np.asarray(observation["state"], dtype=np.float64).copy()
                    successor_pose = block_pose_cost(successor, goal)
                    successor_xy = float(block_xy_cost(successor, goal))
                    successor_angle = angle_error(successor[4], goal[4])
                    probe_rows.append({
                        "action": np.asarray(action, dtype=np.float32).copy(),
                        "pose_cost": float(successor_pose),
                        "pose_improvement": float(source_pose - successor_pose),
                        "safe": bool(
                            successor_xy <= source_xy + 10.0
                            and successor_angle <= source_angle + 0.25
                        ),
                        "global_safe": bool(
                            successor_xy <= current_xy_cost + reacquisition_xy_margin
                            and successor_angle <= angle_error(state[4], goal[4]) + 0.25
                        ),
                        "productive": bool(
                            successor_pose < source_pose - 1e-5
                            and np.linalg.norm(
                                successor[2:4] - item["state"][2:4]
                            ) > 0.5
                        ),
                        # A probe can be useful even when the weighted pose
                        # score is numerically flat (for example, a pure
                        # contact-mode change).  It must still move the block
                        # by a measurable amount and avoid worsening the
                        # source pose.  This prevents a zero-action probe
                        # from consuming a recovery budget merely because it
                        # is globally safe.
                        "useful": bool(
                            successor_xy <= current_xy_cost + reacquisition_xy_margin
                            and successor_angle <= angle_error(state[4], goal[4]) + 0.25
                            and np.linalg.norm(
                                successor[2:4] - item["state"][2:4]
                            ) > 0.5
                            and successor_pose <= source_pose + 1e-5
                        ),
                        "after_xy": successor_xy,
                        "after_angle": successor_angle,
                        "block_motion": float(
                            np.linalg.norm(successor[2:4] - item["state"][2:4])
                        ),
                        "terminated": bool(terminated or truncated),
                    })
                probe_rows.sort(
                    key=lambda row: (
                        not row["global_safe"],
                        not row["useful"],
                        not row["productive"],
                        row["pose_cost"],
                        -row["block_motion"],
                    )
                )
                best_probe = probe_rows[0]
                item["verified_probe"] = {
                    "probe_count": int(len(probe_rows)),
                    "best_pose_cost": float(best_probe["pose_cost"]),
                    "best_pose_improvement": float(best_probe["pose_improvement"]),
                    "best_safe": bool(best_probe["safe"]),
                    "best_global_safe": bool(best_probe["global_safe"]),
                    "best_useful": bool(best_probe["useful"]),
                    "best_productive": bool(best_probe["productive"]),
                    "best_after_xy": float(best_probe["after_xy"]),
                    "best_after_angle": float(best_probe["after_angle"]),
                    "best_block_motion": float(best_probe["block_motion"]),
                    "best_action": best_probe["action"].copy(),
                    "all_finite": bool(
                        np.isfinite(
                            [row["pose_cost"] for row in probe_rows]
                        ).all()
                    ),
                }
            selected = min(
                candidates,
                key=lambda item: (
                    not item.get("verified_probe", {}).get("best_global_safe", False),
                    not item.get("verified_probe", {}).get("best_productive", False),
                    item.get("verified_probe", {}).get("best_pose_cost", np.inf),
                    -item.get("verified_probe", {}).get("best_block_motion", 0.0),
                    item["contact_distance"],
                ),
            )
        else:
            selected = min(
                eligible or candidates,
                key=lambda item: (item["contact_distance"], item["pose_cost"]),
            )
    else:
        selected = min(
            candidates,
            key=lambda item: (item["pose_cost"], item["contact_distance"]),
        )
    # Each route probe executes from the same source state, so the simulator
    # is left at the last candidate rather than necessarily at the selected
    # one.  Restore the selected state before returning; otherwise the caller
    # plans from one state and executes the next action from another, which
    # can look like an unstable contact mode or an unexplained pose jump.
    env.reset(
        seed=0,
        options={"state": np.asarray(selected["state"], dtype=np.float64).copy(),
                 "goal_state": np.asarray(goal, dtype=np.float64).copy()},
    )
    return np.asarray(selected["state"], dtype=np.float64), {
        "selected_offset": selected["offset"],
        "selected_pose_cost": selected["pose_cost"],
        "selected_block_motion": selected["block_motion"],
        "selected_contact_distance": selected["contact_distance"],
        "selection_mode": selection_mode,
        "verified_probe": selected.get("verified_probe"),
        "candidate_scores": [
            {
                "offset": item["offset"],
                "contact_distance_target": item["contact_distance_target"],
                "pose_cost": item["pose_cost"],
                "block_motion": item["block_motion"],
                "xy_progress": item["xy_progress"],
                "contact_distance": item["contact_distance"],
                **(
                    {"verified_probe": item["verified_probe"]}
                    if "verified_probe" in item
                    else {}
                ),
            }
            for item in candidates
        ],
    }


def _plan_receding_block_pose_core(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Replan block pose at every step and execute only the first action."""
    current = np.asarray(state, dtype=np.float64).copy()
    trace = []
    no_motion_steps = 0
    recovery_count = 0
    for step in range(args.max_steps):
        args._force_fan_active = no_motion_steps >= 2
        bank = candidate_bank(rng, current, goal, args)
        mean = bank.mean(axis=0)
        std = np.maximum(bank.std(axis=0), 0.12)
        best_value = float("inf")
        best_actions = bank[0].copy()
        total_rollouts = 0
        for iteration in range(args.cem_iterations):
            candidates = np.clip(
                mean[None] + std[None] * rng.normal(size=bank.shape), -1.0, 1.0
            ).astype(np.float32)
            candidates[: min(16, len(candidates))] = bank[: min(16, len(candidates))]
            values, finals = [], []
            for candidate in candidates:
                final, _ = rollout(env, current, goal, candidate)
                values.append(
                    gated_block_pose_cost(final, current, goal, args)
                    if args.hard_contact_gate
                    else block_pose_cost(
                        final,
                        goal,
                        adaptive_angle=not args.disable_adaptive_angle,
                        cost_mode=args.cost_mode,
                    )
                )
                finals.append(final)
            total_rollouts += len(candidates)
            values_array = np.asarray(values, dtype=np.float64)
            elite = candidates[np.argsort(values_array)[: max(8, len(candidates) // 8)]]
            mean = 0.35 * mean + 0.65 * elite.mean(axis=0)
            std = 0.35 * std + 0.65 * np.maximum(elite.std(axis=0), 0.05)
            index = int(np.argmin(values_array))
            if float(values_array[index]) < best_value:
                best_value = float(values_array[index])
                best_actions = candidates[index].copy()
            bank = candidates
        next_state, terminated = rollout(env, current, goal, best_actions[:1])
        block_motion = float(np.linalg.norm(next_state[2:4] - current[2:4]))
        trace.append({
            "step": step,
            "state_before": current.copy(),
            "action": best_actions[0].copy(),
            "state_after": next_state.copy(),
            "predicted_block_pose_cost": best_value,
            "actual_block_pose_cost": block_pose_cost(
                next_state,
                goal,
                adaptive_angle=not args.disable_adaptive_angle,
                cost_mode=args.cost_mode,
            ),
            "actual_block_motion": block_motion,
            "simulator_rollouts": total_rollouts,
        })
        current = next_state
        if terminated:
            break
        # Once the block pose satisfies the task tolerance, stop the push
        # phase immediately.  Continuing to optimize after closure can turn a
        # successful pose into an artificial contact-recovery failure.
        if (
            block_xy_cost(current, goal) < 20.0
            and angle_error(current[4], goal[4]) < np.pi / 9.0
        ):
            break
        if block_motion < 0.5:
            no_motion_steps += 1
        else:
            no_motion_steps = 0
        if (
            no_motion_steps >= args.recovery_patience
            and recovery_count < 2
            and not args.disable_reacquisition
        ):
            # A static candidate bank can repeatedly choose free-space actions
            # when the agent is on the wrong side of the T.  Reacquire the
            # desired contact normal only after observing this failure online.
            recovered, recovery_info = probe_contact_routes(
                env, current, goal, args
            )
            recovery_count += 1
            trace.append({
                "step": step,
                "event": "contact_reacquisition",
                "state_before": current.copy(),
                "state_after": recovered.copy(),
                "block_motion": float(np.linalg.norm(recovered[2:4] - current[2:4])),
                "recovery_info": recovery_info,
            })
            current = recovered
            no_motion_steps = 0
    return current, trace


def plan_receding_block_pose(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run the normal receding push, then rescue only an unclosed push.

    The rescue is an alternate push-phase search from the same initial
    simulator state, not extra post-hoc return tuning.  It is triggered by a
    pre-registered block-pose failure and records both candidates so the
    resulting gain can be audited as adaptive search coverage rather than
    silently folded into the baseline.
    """
    initial = np.asarray(state, dtype=np.float64).copy()
    baseline_state, baseline_trace = _plan_receding_block_pose_core(
        env, initial, goal, rng, args
    )
    if block_pose_success(baseline_state, goal):
        return baseline_state, baseline_trace
    if int(getattr(args, "push_rescue_restarts", 0)) <= 0:
        return baseline_state, baseline_trace
    if int(getattr(args, "push_rescue_population", 0)) < 8:
        raise ValueError("push-rescue population is too small")
    if int(getattr(args, "push_rescue_iterations", 0)) < 1:
        raise ValueError("push-rescue iterations must be positive")

    def rank(candidate: np.ndarray) -> tuple[int, float]:
        return (
            0 if block_pose_success(candidate, goal) else 1,
            block_pose_cost(
                candidate,
                goal,
                adaptive_angle=not args.disable_adaptive_angle,
                cost_mode=args.cost_mode,
            ),
        )

    best_state = baseline_state
    best_trace = baseline_trace
    selected_rescue = False
    rescue_records = []
    for restart in range(int(args.push_rescue_restarts)):
        rescue_args = argparse.Namespace(**vars(args))
        rescue_args.population = int(args.push_rescue_population)
        rescue_args.cem_iterations = int(args.push_rescue_iterations)
        rescue_args.push_rescue_restarts = 0
        rescue_seed = int(rng.integers(0, 2**32 - 1))
        candidate_state, candidate_trace = _plan_receding_block_pose_core(
            env,
            initial,
            goal,
            np.random.default_rng(rescue_seed),
            rescue_args,
        )
        candidate_success = block_pose_success(candidate_state, goal)
        rescue_records.append({
            "restart": restart,
            "seed": rescue_seed,
            "population": int(rescue_args.population),
            "iterations": int(rescue_args.cem_iterations),
            "block_pose_success": bool(candidate_success),
            "final_block_pose_cost": float(block_pose_cost(candidate_state, goal)),
        })
        if rank(candidate_state) < rank(best_state):
            best_state = candidate_state
            best_trace = candidate_trace
            selected_rescue = True
        if candidate_success:
            break
    best_trace = list(best_trace) + [{
        "step": len(best_trace),
        "event": "push_rescue_search",
        "baseline_block_pose_success": False,
        "selected_rescue": selected_rescue,
        "rescue_records": rescue_records,
    }]
    return best_state, best_trace


def closure_return_cost(
    state: np.ndarray,
    goal: np.ndarray,
) -> float:
    """Score a terminal return using the complete PushT joint criterion."""
    joint_position_distance = float(
        np.linalg.norm(np.asarray(state)[:4] - np.asarray(goal)[:4])
    )
    if block_pose_success(state, goal):
        return joint_position_distance
    return 1000.0 + joint_position_distance + block_pose_cost(state, goal)


def closure_return_trajectory_cost(
    states: np.ndarray,
    goal: np.ndarray,
    path_xy_margin: float = 0.0,
    path_angle_margin: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    """Score a return sequence with a pathwise closure constraint.

    The block is already solved when this phase starts.  A terminal-only
    score is insufficient because a candidate can leave the PushT block-pose
    tolerance and re-enter it at the last step, while transferring contact
    momentum that makes the reported endpoint non-reproducible under a safe
    return.  We therefore require block-pose closure at every observed state
    and rank feasible candidates by the exact joint-position distance used by
    ``task_success``.  We use a narrow, explicit intermediate safety tube to
    avoid treating a sub-pixel numerical excursion as a physical failure; the
    final state must still satisfy the strict PushT criterion.  This is still
    privileged simulator-state planning.
    """
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 5:
        raise ValueError("closure return requires a non-empty state trajectory")
    xy = np.asarray([block_xy_cost(row, goal) for row in values], dtype=np.float64)
    angles = np.asarray([angle_error(row[4], goal[4]) for row in values], dtype=np.float64)
    joint = np.linalg.norm(values[:, :4] - np.asarray(goal, dtype=np.float64)[:4], axis=1)
    xy_limit = 20.0 + max(float(path_xy_margin), 0.0)
    angle_limit = np.pi / 9.0 + max(float(path_angle_margin), 0.0)
    normalized_violation = np.maximum(
        np.maximum(xy / xy_limit - 1.0, angles / angle_limit - 1.0),
        0.0,
    )
    closure_preserved = bool(np.all(normalized_violation <= 1e-9))
    final = values[-1]
    metrics = {
        "closure_preserved": closure_preserved,
        "max_closure_violation": float(np.max(normalized_violation)),
        "path_xy_limit": float(xy_limit),
        "path_angle_limit": float(angle_limit),
        "final_block_xy_distance": float(xy[-1]),
        "final_angle_error": float(angles[-1]),
        "final_agent_xy_distance": float(np.linalg.norm(final[:2] - np.asarray(goal)[:2])),
        "final_joint_position_distance": float(joint[-1]),
        "task_success": bool(task_success(final, goal)),
    }
    if not closure_preserved:
        # Keep the hard gate dominant, but retain a smooth signal for CEM to
        # move back toward the closed tube rather than selecting arbitrary
        # free-space trajectories.
        score = 100000.0 + 1000.0 * float(np.max(normalized_violation)) + float(joint[-1])
    else:
        # This is the complete joint endpoint quantity underlying PushT
        # success, not an agent-only proxy.
        score = float(joint[-1])
    return score, metrics


def terminal_stability_certificate(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    steps: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Verify that a strict endpoint remains closed under a short hold.

    A one-frame threshold crossing is not a reliable terminal interface for a
    dynamic simulator: residual contact velocity can immediately reopen the
    block pose.  This certificate replays zero action from the candidate
    endpoint and requires the strict joint criterion and block-pose criterion
    at every hold state.  It uses only the observed simulator successor and is
    therefore explicitly a privileged state diagnostic.
    """
    current = np.asarray(state, dtype=np.float64).copy()
    trajectory = [current.copy()]
    if int(steps) < 0:
        raise ValueError("terminal stability steps must be non-negative")
    terminated = False
    for _ in range(int(steps)):
        env.reset(seed=0, options={"state": current.copy(), "goal_state": np.asarray(goal).copy()})
        observation, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        current = np.asarray(observation["state"], dtype=np.float64).copy()
        trajectory.append(current.copy())
        if not task_success(current, goal) or not block_pose_success(current, goal):
            return current, {
                "stable": False,
                "steps": len(trajectory) - 1,
                "terminated": bool(terminated or truncated),
                "trajectory": np.asarray(trajectory, dtype=np.float64),
                "failure": "strict_criterion_lost_during_hold",
            }
    return current, {
        "stable": True,
        "steps": len(trajectory) - 1,
        "terminated": bool(terminated),
        "trajectory": np.asarray(trajectory, dtype=np.float64),
        "failure": None,
    }


def _run_closure_return_cem(
    env: gym.Env,
    current: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    return_args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None, float, dict[str, Any]]:
    """Run one independently seeded closure-preserving terminal CEM."""
    bank = candidate_bank(rng, current, goal, return_args)
    mean = bank.mean(axis=0)
    std = np.maximum(bank.std(axis=0), 0.12)
    best_value = float("inf")
    best_actions: np.ndarray | None = None
    best_state: np.ndarray | None = None
    best_metrics: dict[str, Any] = {}
    total_rollouts = 0
    for iteration in range(int(return_args.cem_iterations)):
        candidates = np.clip(
            mean[None] + std[None] * rng.normal(size=bank.shape), -1.0, 1.0
        ).astype(np.float32)
        warm = min(32, len(candidates))
        candidates[:warm] = bank[:warm]
        values = []
        finals = []
        metrics_list = []
        for candidate in candidates:
            final, states, terminated = rollout_trajectory(env, current, goal, candidate)
            stability_steps = int(getattr(return_args, "terminal_stability_steps", 0))
            hold_state = np.asarray(final, dtype=np.float64).copy()
            hold_states = np.asarray([hold_state], dtype=np.float64)
            hold_info = {
                "stable": True,
                "steps": 0,
                "terminated": bool(terminated),
                "failure": None,
            }
            if stability_steps > 0:
                hold_state, hold_info = terminal_stability_certificate(
                    env,
                    final,
                    goal,
                    steps=stability_steps,
                )
                hold_states = np.asarray(hold_info["trajectory"], dtype=np.float64)
            # Optimize the endpoint *after* the required hold.  This prevents
            # CEM from selecting a high-velocity threshold crossing that is
            # accepted by the ordinary endpoint score but immediately reopens
            # the joint criterion under zero action.
            scored_states = np.concatenate(
                [np.asarray(states, dtype=np.float64), hold_states[1:]], axis=0
            )
            value, metrics = closure_return_trajectory_cost(
                scored_states,
                goal,
                path_xy_margin=float(getattr(return_args, "closure_return_path_xy_margin", 0.0)),
                path_angle_margin=float(getattr(return_args, "closure_return_path_angle_margin", 0.0)),
            )
            metrics["terminal_stability"] = bool(hold_info["stable"])
            metrics["terminal_stability_steps"] = stability_steps
            # A simulator termination is only a valid early stop when it
            # satisfies the complete joint criterion and the pathwise closure
            # gate.  Otherwise it remains an ordinary scored candidate.
            if metrics["task_success"] and metrics["closure_preserved"] and hold_info["stable"]:
                value = 0.0
            values.append(value)
            finals.append(hold_state)
            metrics_list.append(metrics)
        total_rollouts += len(candidates)
        values_array = np.asarray(values, dtype=np.float64)
        elite_index = np.argsort(values_array)[: max(8, len(candidates) // 8)]
        selected = int(elite_index[0])
        if float(values_array[selected]) < best_value:
            best_value = float(values_array[selected])
            best_actions = np.concatenate(
                [
                    candidates[selected].copy(),
                    np.zeros((stability_steps, 2), dtype=np.float32),
                ],
                axis=0,
            )
            best_state = np.asarray(finals[selected], dtype=np.float64).copy()
            best_metrics = dict(metrics_list[selected])
        elite = candidates[elite_index]
        mean = 0.35 * mean + 0.65 * elite.mean(axis=0)
        std = 0.35 * std + 0.65 * np.maximum(elite.std(axis=0), 0.05)
        bank = candidates
        if best_metrics.get("task_success") and best_metrics.get("closure_preserved"):
            break
    return best_actions, best_state, best_value, {
        "population": int(return_args.population),
        "iterations": int(return_args.cem_iterations),
        "horizon": int(return_args.horizon),
        "simulator_rollouts": int(total_rollouts),
        "best_metrics": best_metrics,
    }


def plan_closure_preserving_return(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Search a short terminal trajectory after block-pose closure.

    A geometric return route can be infeasible from some contact sides even
    though a short simulator-replayed action sequence can move the agent to
    its goal without leaving the block termination region.  This terminal
    CEM is invoked only after block closure and uses a hard block-pose gate;
    it does not reintroduce the original joint endpoint objective into the
    push phase.
    """
    current = np.asarray(state, dtype=np.float64).copy()
    if not block_pose_success(current, goal):
        fallback, terminated = return_agent(
            env, current, goal, steps=args.return_steps, clearance=args.return_clearance
        )
        return fallback, {
            "mode": "geometric_fallback_before_block_closure",
            "planned": False,
            "terminated": bool(terminated),
        }
    # Agent proximity alone is not the endpoint criterion.  The environment
    # can place the agent within 20 px while the residual block translation
    # still makes the four-dimensional joint position norm exceed the strict
    # task margin.  Only skip terminal search when the complete criterion is
    # already satisfied.
    if task_success(current, goal):
        stable_state, stability = terminal_stability_certificate(
            env,
            current,
            goal,
            steps=int(getattr(args, "terminal_stability_steps", 1)),
        )
        if stability["stable"]:
            return stable_state, {
                "mode": "already_task_closed",
                "planned": False,
                "terminated": True,
                "terminal_stability": stability,
            }

    return_args = argparse.Namespace(**vars(args))
    return_args.population = int(args.closure_return_population)
    return_args.horizon = int(args.closure_return_horizon)
    return_args.action_block = 1
    return_args.disable_force_fan = True
    if return_args.population < 8 or return_args.horizon < 1:
        raise ValueError("closure-return population and horizon are too small")

    low_args = argparse.Namespace(**vars(return_args))
    low_args.cem_iterations = int(args.closure_return_iterations)
    best_actions, best_state, best_value, low_info = _run_closure_return_cem(
        env, current, goal, rng, low_args
    )
    search_records = [{"budget": "low", **low_info}]
    if (
        best_actions is not None
        and best_state is not None
        and low_info["best_metrics"].get("task_success")
        and low_info["best_metrics"].get("closure_preserved")
    ):
        committed, committed_states, terminated = rollout_trajectory(
            env, current, goal, best_actions
        )
        _, committed_metrics = closure_return_trajectory_cost(
            committed_states,
            goal,
            path_xy_margin=float(args.closure_return_path_xy_margin),
            path_angle_margin=float(args.closure_return_path_angle_margin),
        )
        if committed_metrics["task_success"] and committed_metrics["closure_preserved"]:
            stable_state, stability = terminal_stability_certificate(
                env,
                committed,
                goal,
                steps=int(getattr(args, "terminal_stability_steps", 1)),
            )
            if stability["stable"]:
                committed = stable_state
                return committed, {
                    "mode": "closure_preserving_terminal_cem",
                    "planned": True,
                    "rescue_triggered": False,
                    "terminated": True,
                    "objective_value": float(best_value),
                    "agent_goal_distance": float(np.linalg.norm(committed[:2] - goal[:2])),
                    "joint_position_distance": float(np.linalg.norm(committed[:4] - goal[:4])),
                    "block_pose_preserved": True,
                    "pathwise_block_pose_preserved": True,
                    "terminal_stability": stability,
                    "search_records": search_records,
                }

    rescue_args = argparse.Namespace(**vars(return_args))
    rescue_args.population = int(args.closure_return_rescue_population)
    rescue_args.cem_iterations = int(args.closure_return_rescue_iterations)
    rescue_records = []
    for restart in range(max(0, int(args.closure_return_rescue_restarts))):
        restart_seed = int(rng.integers(0, 2**32 - 1))
        rescue_rng = np.random.default_rng(restart_seed)
        rescue_actions, rescue_state, rescue_value, rescue_info = _run_closure_return_cem(
            env, current, goal, rescue_rng, rescue_args
        )
        rescue_info = {"restart": restart, "seed": restart_seed, **rescue_info}
        rescue_records.append(rescue_info)
        if (
            rescue_actions is not None
            and rescue_state is not None
            and rescue_info["best_metrics"].get("task_success")
            and rescue_info["best_metrics"].get("closure_preserved")
        ):
            committed, committed_states, terminated = rollout_trajectory(
                env, current, goal, rescue_actions
            )
            _, committed_metrics = closure_return_trajectory_cost(
                committed_states,
                goal,
                path_xy_margin=float(args.closure_return_path_xy_margin),
                path_angle_margin=float(args.closure_return_path_angle_margin),
            )
            if committed_metrics["task_success"] and committed_metrics["closure_preserved"]:
                stable_state, stability = terminal_stability_certificate(
                    env,
                    committed,
                    goal,
                    steps=int(getattr(args, "terminal_stability_steps", 1)),
                )
                if stability["stable"]:
                    search_records.extend({"budget": "rescue", **record} for record in rescue_records)
                    committed = stable_state
                    return committed, {
                        "mode": "closure_preserving_terminal_cem_rescue",
                        "planned": True,
                        "rescue_triggered": True,
                        "rescue_success": True,
                        "rescue_restart": restart,
                        "terminated": True,
                        "objective_value": float(rescue_value),
                        "agent_goal_distance": float(np.linalg.norm(committed[:2] - goal[:2])),
                        "joint_position_distance": float(np.linalg.norm(committed[:4] - goal[:4])),
                        "block_pose_preserved": True,
                        "pathwise_block_pose_preserved": True,
                        "terminal_stability": stability,
                        "search_records": search_records,
                    }

    search_records.extend({"budget": "rescue", **record} for record in rescue_records)

    deep_args = argparse.Namespace(**vars(return_args))
    deep_args.population = int(args.closure_return_deep_population)
    deep_args.horizon = int(args.closure_return_deep_horizon)
    deep_args.cem_iterations = int(args.closure_return_deep_iterations)
    deep_records = []
    for restart in range(max(0, int(args.closure_return_deep_restarts))):
        restart_seed = int(rng.integers(0, 2**32 - 1))
        deep_rng = np.random.default_rng(restart_seed)
        deep_actions, deep_state, deep_value, deep_info = _run_closure_return_cem(
            env, current, goal, deep_rng, deep_args
        )
        deep_info = {"restart": restart, "seed": restart_seed, **deep_info}
        deep_records.append(deep_info)
        if (
            deep_actions is not None
            and deep_state is not None
            and deep_info["best_metrics"].get("task_success")
            and deep_info["best_metrics"].get("closure_preserved")
        ):
            committed, committed_states, terminated = rollout_trajectory(
                env, current, goal, deep_actions
            )
            _, committed_metrics = closure_return_trajectory_cost(
                committed_states,
                goal,
                path_xy_margin=float(args.closure_return_path_xy_margin),
                path_angle_margin=float(args.closure_return_path_angle_margin),
            )
            if committed_metrics["task_success"] and committed_metrics["closure_preserved"]:
                stable_state, stability = terminal_stability_certificate(
                    env,
                    committed,
                    goal,
                    steps=int(getattr(args, "terminal_stability_steps", 1)),
                )
                if stability["stable"]:
                    search_records.extend({"budget": "deep_rescue", **record} for record in deep_records)
                    committed = stable_state
                    return committed, {
                        "mode": "closure_preserving_terminal_cem_deep_rescue",
                        "planned": True,
                        "rescue_triggered": True,
                        "rescue_success": True,
                        "deep_rescue_success": True,
                        "deep_rescue_restart": restart,
                        "terminated": True,
                        "objective_value": float(deep_value),
                        "agent_goal_distance": float(np.linalg.norm(committed[:2] - goal[:2])),
                        "joint_position_distance": float(np.linalg.norm(committed[:4] - goal[:4])),
                        "block_pose_preserved": True,
                        "pathwise_block_pose_preserved": True,
                        "terminal_stability": stability,
                        "search_records": search_records,
                    }
    search_records.extend({"budget": "deep_rescue", **record} for record in deep_records)

    fallback, terminated = return_agent(
        env, current, goal, steps=args.return_steps, clearance=args.return_clearance
    )
    # A geometric fallback is allowed to move the agent, but it must never
    # silently spend the block-pose closure while trying to solve the agent
    # endpoint.  If it does, reject the candidate and restore the exact
    # pre-return state so the caller can retry or resume pushing.
    fallback_block_preserved = bool(block_pose_success(fallback, goal))
    if not fallback_block_preserved:
        env.reset(seed=0, options={"state": current.copy(), "goal_state": goal.copy()})
        fallback = current.copy()
        terminated = False
    return fallback, {
        "mode": "geometric_fallback_after_terminal_cem",
        "planned": False,
        "rescue_triggered": bool(rescue_records),
        "rescue_success": False,
        "deep_rescue_triggered": bool(deep_records),
        "deep_rescue_success": False,
        "terminated": bool(terminated),
        "objective_value": float(best_value),
        "block_pose_preserved": fallback_block_preserved,
        "fallback_rejected_block_regression": not fallback_block_preserved,
        "pathwise_block_pose_preserved": None,
        "search_records": search_records,
    }


def phase_cost(states: np.ndarray, goal: np.ndarray, phase: str) -> float:
    """Score a candidate according to the active control phase.

    The translation and rotation subproblems have incompatible local optima.
    A single weighted endpoint score can therefore prefer an action that
    improves one coordinate while making the other phase unrecoverable.  The
    staged arm uses a lexicographic-style score: translation is protected
    until it is close, then angle is optimized while keeping translation
    inside a bounded tube.
    """
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("phase_cost expects a non-empty state trajectory")
    xy = np.asarray([block_xy_cost(row, goal) for row in values], dtype=np.float64)
    angle = np.asarray([angle_error(row[4], goal[4]) for row in values], dtype=np.float64)
    terminal_xy = float(xy[-1])
    terminal_angle = float(angle[-1])
    # The path term makes candidates that briefly make progress and then lose
    # contact less attractive than trajectories with the same endpoint but a
    # recoverable monotone approach.
    regressions = np.maximum(0.0, np.diff(xy))
    if phase == "translate":
        return float(terminal_xy + 8.0 * terminal_angle + 0.20 * np.mean(xy) + 0.35 * np.sum(regressions))
    if phase == "rotate":
        # Once translation is close, a small XY increase is acceptable, but a
        # candidate that leaves the translation tube is rejected softly.
        tube_penalty = max(0.0, terminal_xy - 35.0) ** 2
        return float(120.0 * terminal_angle + 0.80 * terminal_xy + 0.25 * tube_penalty + 0.20 * np.mean(angle))
    raise ValueError(f"unknown planning phase: {phase}")


def phase_candidate_plan(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
    phase: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """CEM proposal for one phase; only its short prefix is executed."""
    bank = candidate_bank(rng, state, goal, args)
    mean = bank.mean(axis=0)
    std = np.maximum(bank.std(axis=0), 0.12)
    best_value = float("inf")
    best_actions = bank[0].copy()
    total_rollouts = 0
    for iteration in range(args.cem_iterations):
        candidates = np.clip(
            mean[None] + std[None] * rng.normal(size=bank.shape), -1.0, 1.0
        ).astype(np.float32)
        candidates[: min(16, len(candidates))] = bank[: min(16, len(candidates))]
        values = []
        for candidate in candidates:
            _, trajectory, _ = rollout_trajectory(env, state, goal, candidate)
            values.append(phase_cost(trajectory, goal, phase))
        total_rollouts += len(candidates)
        values_array = np.asarray(values, dtype=np.float64)
        elite = candidates[np.argsort(values_array)[: max(8, len(candidates) // 8)]]
        mean = 0.35 * mean + 0.65 * elite.mean(axis=0)
        std = 0.35 * std + 0.65 * np.maximum(elite.std(axis=0), 0.05)
        index = int(np.argmin(values_array))
        if float(values_array[index]) < best_value:
            best_value = float(values_array[index])
            best_actions = candidates[index].copy()
        bank = candidates
    return best_actions, {
        "phase": phase,
        "objective": "phase_lexicographic_trajectory",
        "objective_value": best_value,
        "simulator_rollouts": total_rollouts,
    }


def plan_staged_phase(
    env: gym.Env,
    state: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Explicit acquire/translate/rotate controller with bounded recovery.

    This is still a privileged simulator-state diagnostic.  Its purpose is
    to test the planning-interface hypothesis with a controller whose active
    objective changes only after the preceding subgoal is satisfied.
    """
    current = np.asarray(state, dtype=np.float64).copy()
    trace: list[dict[str, Any]] = []
    stale = 0
    recovery_count = 0
    for step in range(args.max_steps):
        xy_before = block_xy_cost(current, goal)
        angle_before = angle_error(current[4], goal[4])
        phase = "translate" if xy_before > 28.0 else "rotate"
        actions, plan_info = phase_candidate_plan(env, current, goal, rng, args, phase)
        execute_count = min(max(1, args.phase_action_block), len(actions))
        previous = current.copy()
        for action_index, action in enumerate(actions[:execute_count]):
            next_state, terminated = rollout(env, current, goal, action[None])
            block_motion = float(np.linalg.norm(next_state[2:4] - current[2:4]))
            trace.append({
                "step": step,
                "action_index": action_index,
                "phase": phase,
                "state_before": current.copy(),
                "action": np.asarray(action).copy(),
                "state_after": next_state.copy(),
                "block_xy_before": block_xy_cost(current, goal),
                "block_xy_after": block_xy_cost(next_state, goal),
                "angle_before": angle_error(current[4], goal[4]),
                "angle_after": angle_error(next_state[4], goal[4]),
                "actual_block_motion": block_motion,
                "plan": plan_info,
            })
            current = next_state
            if terminated:
                break
        xy_after = block_xy_cost(current, goal)
        angle_after = angle_error(current[4], goal[4])
        if phase == "translate":
            improved = xy_after < xy_before - 0.05
        else:
            improved = angle_after < angle_before - 0.002 and xy_after < xy_before + 8.0
        stale = 0 if improved else stale + 1
        if (
            xy_after < 20.0
            and angle_after < np.pi / 9.0
        ):
            break
        if (
            stale >= args.phase_patience
            and recovery_count < 3
            and not args.disable_reacquisition
        ):
            recovered, recovery_info = probe_contact_routes(env, current, goal, args)
            trace.append({
                "step": step,
                "event": "phase_contact_reacquisition",
                "phase": phase,
                "state_before": current.copy(),
                "state_after": recovered.copy(),
                "block_motion": float(np.linalg.norm(recovered[2:4] - current[2:4])),
                "recovery_info": recovery_info,
            })
            current = recovered
            recovery_count += 1
            stale = 0
        if np.array_equal(previous, current):
            break
    return current, trace


def episode_row(
    state: np.ndarray,
    goal: np.ndarray,
    arm_state: np.ndarray,
    final_state: np.ndarray,
    trace: list[dict[str, Any]],
    arm: str,
    return_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "pre_return_state": arm_state,
        "final_state": final_state,
        "pre_return_block_xy_distance": block_xy_cost(arm_state, goal),
        "pre_return_angle_error": angle_error(arm_state[4], goal[4]),
        "final_block_xy_distance": block_xy_cost(final_state, goal),
        "final_agent_xy_distance": float(np.linalg.norm(final_state[:2] - goal[:2])),
        "final_angle_error": angle_error(final_state[4], goal[4]),
        "final_joint_position_distance": float(np.linalg.norm(final_state[:4] - goal[:4])),
        "success": task_success(final_state, goal),
        "block_pose_success": bool(block_xy_cost(final_state, goal) < 20.0 and angle_error(final_state[4], goal[4]) < np.pi / 9.0),
        "planning_steps": len(trace),
        "trace": trace,
        "return_info": return_info or {},
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "block_pose_success_rate": float(np.mean([row["block_pose_success"] for row in rows])),
        "mean_pre_return_block_xy_distance": float(np.mean([row["pre_return_block_xy_distance"] for row in rows])),
        "mean_pre_return_angle_error": float(np.mean([row["pre_return_angle_error"] for row in rows])),
        "mean_final_block_xy_distance": float(np.mean([row["final_block_xy_distance"] for row in rows])),
        "mean_final_agent_xy_distance": float(np.mean([row["final_agent_xy_distance"] for row in rows])),
        "mean_final_angle_error": float(np.mean([row["final_angle_error"] for row in rows])),
        "mean_final_joint_position_distance": float(np.mean([row["final_joint_position_distance"] for row in rows])),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = make_state_only_env()
    reports: dict[str, list[dict[str, Any]]] = {name: [] for name in args.arms}
    try:
        cases = [
            stable_contact_bootstrap(env, args.seed + 10000 + index, args.bootstrap_steps)
            for index in range(args.episodes)
        ]
        for index, case in enumerate(cases):
            initial = np.asarray(case["state"], dtype=np.float64)
            goal = np.asarray(case["goal"], dtype=np.float64)
            if "joint_endpoint" in reports:
                joint_actions, joint_info = plan_one_shot(
                    env, initial, goal, np.random.default_rng(args.seed + 1000 * index + 11), args
                )
                joint_pose, _ = rollout(env, initial, goal, joint_actions)
                joint_final, _ = return_agent(env, joint_pose, goal, clearance=args.return_clearance)
                reports["joint_endpoint"].append(
                    episode_row(initial, goal, joint_pose, joint_final, [], "joint_endpoint") | {"plan": joint_info}
                )

            if "receding_block_pose" in reports:
                receding_pose, trace = plan_receding_block_pose(
                    env, initial, goal, np.random.default_rng(args.seed + 1000 * index + 29), args
                )
                receding_final, receding_return_info = plan_closure_preserving_return(
                    env,
                    receding_pose,
                    goal,
                    np.random.default_rng(args.seed + 1000 * index + 59),
                    args,
                )
                reports["receding_block_pose"].append(
                    episode_row(
                        initial, goal, receding_pose, receding_final, trace,
                        "receding_block_pose", receding_return_info,
                    )
                )

            if "staged_phase" in reports:
                staged_pose, staged_trace = plan_staged_phase(
                    env, initial, goal, np.random.default_rng(args.seed + 1000 * index + 47), args
                )
                staged_final, _ = return_agent(env, staged_pose, goal, clearance=args.return_clearance)
                reports["staged_phase"].append(
                    episode_row(initial, goal, staged_pose, staged_final, staged_trace, "staged_phase")
                )

    finally:
        env.close()
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "episodes": args.episodes,
            "seed": args.seed,
            "same_cases": True,
            "bootstrap": "stable contact; same initial simulator states reused by both arms",
            "joint_endpoint": "one-shot simulator CEM scored by final joint agent/block endpoint",
            "receding_block_pose": "simulator CEM scores block pose, executes one action, replans; safe return after block closure",
            "closure_return": "terminal simulator CEM with hard block-pose preservation, followed by geometric fallback",
            "closure_return_population": int(args.closure_return_population),
            "closure_return_iterations": int(args.closure_return_iterations),
            "closure_return_horizon": int(args.closure_return_horizon),
            "closure_return_rescue_population": int(args.closure_return_rescue_population),
            "closure_return_rescue_iterations": int(args.closure_return_rescue_iterations),
            "closure_return_rescue_restarts": int(args.closure_return_rescue_restarts),
            "closure_return_path_xy_margin": float(args.closure_return_path_xy_margin),
            "closure_return_path_angle_margin": float(args.closure_return_path_angle_margin),
            "closure_return_deep_population": int(args.closure_return_deep_population),
            "closure_return_deep_iterations": int(args.closure_return_deep_iterations),
            "closure_return_deep_horizon": int(args.closure_return_deep_horizon),
            "closure_return_deep_restarts": int(args.closure_return_deep_restarts),
            "horizon": args.horizon,
            "population": args.population,
            "cem_iterations": args.cem_iterations,
            "push_rescue_population": int(args.push_rescue_population),
            "push_rescue_iterations": int(args.push_rescue_iterations),
            "push_rescue_restarts": int(args.push_rescue_restarts),
            "max_steps": args.max_steps,
            "phase_action_block": args.phase_action_block,
            "phase_patience": args.phase_patience,
            "wall_seconds": time.perf_counter() - started,
            "disable_force_fan": bool(args.disable_force_fan),
            "force_fan_xy_threshold": float(args.force_fan_xy_threshold),
            "force_fan_near_goal_only": bool(args.force_fan_near_goal_only),
            "force_fan_on_stagnation": bool(args.force_fan_on_stagnation),
            "cost_mode": args.cost_mode,
            "disable_adaptive_angle": bool(args.disable_adaptive_angle),
            "disable_reacquisition": bool(args.disable_reacquisition),
            "hard_contact_gate": bool(args.hard_contact_gate),
        },
        "reports": reports,
        "summaries": {name: summarize(rows) for name, rows in reports.items()},
        "sanity_checks": {
            "same_case_count": all(len(rows) == args.episodes for rows in reports.values()),
            "all_finite": bool(all(np.isfinite(value) for rows in reports.values() for row in rows for value in [row["final_block_xy_distance"], row["final_agent_xy_distance"], row["final_angle_error"]])),
            "trace_present": bool(all(len(row["trace"]) > 0 for name, rows in reports.items() if name != "joint_endpoint" for row in rows)),
        },
        "claim_boundary": (
            "Privileged simulator-state matched planning diagnosis. The repaired arm tests objective decoupling and receding "
            "replanning in the simulator. The staged arm isolates translation and orientation phases and records bounded "
            "contact recovery; neither arm is a frozen-LeWM visual planner, learned physical adapter, or evidence of general "
            "physical causality."
        ),
    }
    (output / "receding_block_pose_planning.json").write_text(json.dumps(jsonable(report), indent=2, sort_keys=True))
    lines = [
        "# PushT matched planning repair",
        "",
        "| arm | task success | block-pose success | pre-return block XY | final block XY | final agent XY | final angle |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in reports:
        row = report["summaries"][name]
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (name, row["success_rate"], row["block_pose_success_rate"], row["mean_pre_return_block_xy_distance"], row["mean_final_block_xy_distance"], row["mean_final_agent_xy_distance"], row["mean_final_angle_error"]))
    lines += ["", report["claim_boundary"], ""]
    (output / "receding_block_pose_planning.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "summaries": report["summaries"], "sanity_checks": report["sanity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
