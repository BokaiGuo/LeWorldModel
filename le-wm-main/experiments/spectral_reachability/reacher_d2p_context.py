"""Executed temporal contexts for the frozen Reacher D2P protocol.

LeWM consumes one action token for every observed RGB frame and predicts the
next frame after the final token.  A planning root is consequently the state
*after* the recorded history, rather than the reset image used to start that
history.  These helpers make that causal boundary executable and replayable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


# The frozen Reacher checkpoint consumes one token containing five *separate*
# environment actions.  Do not manufacture a lower-rank token by repeating a
# single gentle action five times: planner candidates are full packets and the
# random-action checkpoint statistics describe full-range raw actions.
WARMUP_POLICY = "seeded_full_packet_uniform_v2"


def require_new_output(path: Path, *, receipt: str) -> None:
    """Prevent a failed or repeated launch from silently replacing evidence."""
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite immutable {receipt}: {path}")


def rgb_digest(value: np.ndarray) -> str:
    image = np.ascontiguousarray(np.asarray(value, dtype=np.uint8))
    return hashlib.sha256(image.tobytes()).hexdigest()


def warmup_action_blocks(
    *,
    seed: int,
    history_len: int,
    action_block: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> list[np.ndarray]:
    """Return deterministic full-support raw action tokens for warmup.

    The schedule is fixed by root seed before planning and is independent of
    simulator state, goal, method, and all planner outputs.  Each token is
    expanded into the checkpoint's time-major ``[action_block, action_dim]``
    layout before environment execution.  Each raw action in a token is drawn
    separately from the environment Box.  This preserves the token geometry
    used by the full-packet candidate bank and avoids conditioning a
    five-step checkpoint on an artificial constant-action submanifold.
    """
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    if history_len < 1 or action_block < 1 or low.shape != high.shape or not np.all(low < high):
        raise ValueError("invalid executed-warmup action specification")
    generator = np.random.default_rng(int(seed) ^ 0xD2F0)
    packets = generator.uniform(
        low, high, size=(history_len, action_block, low.size),
    ).astype(np.float32)
    return [packet.reshape(-1) for packet in packets]


def _execute_raw_steps(env: Any, raw_steps: np.ndarray) -> None:
    for action in np.asarray(raw_steps, dtype=np.float32):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError("executed warmup ended the Reacher episode before the planning root")


def capture_execution_snapshot(env: Any) -> dict[str, list[float]] | None:
    """Capture evaluator-only physical state at the observed planning root.

    The planner consumes only RGB, goal RGB, and executed action history. A
    physical snapshot is retained exclusively so every candidate can be
    executed from the exact RGB root after a long GPU planning call, rather
    than relying on a second simulator reset to recreate that root.
    """
    try:
        raw = env.unwrapped
        physics = raw.env.physics
        target = raw.env.task.target_qpos
    except AttributeError:
        # Lightweight unit-test environments do not expose simulator state.
        return None
    return {
        "qpos": np.asarray(physics.data.qpos, dtype=np.float64).tolist(),
        "qvel": np.asarray(physics.data.qvel, dtype=np.float64).tolist(),
        "target_qpos": np.asarray(target, dtype=np.float64).tolist(),
    }


def restore_execution_snapshot(
    env: Any, *, seed: int, snapshot: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Reset bookkeeping, then restore an evaluator-only observed-root state.

    A long planner call exposed a rare reset-and-replay RGB-identity failure.
    Restoring the root actually observed by the planner removes that fragile
    reconstruction dependency while preserving a strict rendered-RGB check.
    """
    _, info = env.reset(seed=int(seed))
    raw = env.unwrapped
    physics = raw.env.physics
    qpos = np.asarray(snapshot["qpos"], dtype=np.float64)
    qvel = np.asarray(snapshot["qvel"], dtype=np.float64)
    target = np.asarray(snapshot["target_qpos"], dtype=np.float64)
    if qpos.shape != physics.data.qpos.shape or qvel.shape != physics.data.qvel.shape:
        raise ValueError("execution snapshot state shape is incompatible with Reacher")
    if target.shape != np.asarray(raw.env.task.target_qpos).shape:
        raise ValueError("execution snapshot target shape is incompatible with Reacher")
    raw.env.task.target_qpos = target.copy()
    physics.data.qpos[:] = qpos
    physics.data.qvel[:] = qvel
    physics.forward()
    root = np.asarray(env.render(), dtype=np.uint8).copy()
    # The task target was generated at reset, not from the state after the
    # warmup.  Render the stored target directly; calling the wrapper helper
    # with ``qpos`` would incorrectly synthesize ``root + target_delta``.
    physics.data.qpos[:] = target
    physics.data.qvel[:] = 0.0
    physics.forward()
    try:
        goal = np.asarray(raw.render(width=224, height=224), dtype=np.uint8).copy()
    except TypeError:
        # Lightweight unit-test environments need not implement sized renders.
        goal = np.asarray(info["goal"], dtype=np.uint8).copy()
    physics.data.qpos[:] = qpos
    physics.data.qvel[:] = qvel
    physics.forward()
    if hasattr(env, "_goal"):
        env._goal = goal.copy()
    return root, goal


def collect_executed_context(
    env: Any,
    *,
    seed: int,
    history_len: int,
    action_block: int,
) -> dict[str, Any]:
    """Reset, execute a causal history, and return the resulting plan root."""
    _, info = env.reset(seed=int(seed))
    goal = np.asarray(info["goal"], dtype=np.uint8).copy()
    blocks = warmup_action_blocks(
        seed=seed,
        history_len=history_len,
        action_block=action_block,
        action_low=np.asarray(env.action_space.low),
        action_high=np.asarray(env.action_space.high),
    )
    frames: list[np.ndarray] = []
    raw_steps: list[np.ndarray] = []
    action_dim = int(np.prod(env.action_space.shape))
    for block in blocks:
        frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
        steps = np.asarray(block, dtype=np.float32).reshape(action_block, action_dim)
        _execute_raw_steps(env, steps)
        raw_steps.extend(step.copy() for step in steps)
    root_frame = np.asarray(env.render(), dtype=np.uint8).copy()
    return {
        "frames": frames,
        "goal": goal,
        "history_action_blocks": [np.asarray(block, dtype=np.float32).copy() for block in blocks],
        "warmup_raw_steps": np.asarray(raw_steps, dtype=np.float32),
        "execution_snapshot": capture_execution_snapshot(env),
        "root_frame": root_frame,
        "context_mode": "executed_warmup",
        "warmup_policy": WARMUP_POLICY,
        "warmup_steps": int(len(raw_steps)),
        "history_rgb_sha256": [rgb_digest(frame) for frame in frames],
        "root_rgb_sha256": rgb_digest(root_frame),
        "goal_rgb_sha256": rgb_digest(goal),
        "warmup_actions_sha256": hashlib.sha256(np.ascontiguousarray(np.asarray(raw_steps, dtype=np.float32)).tobytes()).hexdigest(),
    }


def advance_executed_context(
    env: Any,
    *,
    context: dict[str, Any],
    raw_action_block: np.ndarray,
    action_block: int,
) -> dict[str, Any]:
    """Execute one observed block and shift a causal RGB/action context.

    This is deliberately an evaluator-side operation.  It captures the newly
    observed RGB successor and a snapshot only for later *physical* root
    restoration; the planner receives the returned RGB/action history, never
    qpos, qvel, or target state.  The preceding root is checked before the
    action executes so a stale context cannot silently be advanced.
    """

    frames = [np.asarray(frame, dtype=np.uint8).copy() for frame in context["frames"]]
    blocks = [np.asarray(block, dtype=np.float32).copy() for block in context["history_action_blocks"]]
    history_len = len(frames)
    if history_len < 2 or len(blocks) != history_len:
        raise ValueError("cannot advance an incompatible executed context")
    root_before = np.asarray(env.render(), dtype=np.uint8).copy()
    if not np.array_equal(root_before, np.asarray(context["root_frame"], dtype=np.uint8)):
        raise RuntimeError("execution context root differs before reconfiguration")
    action_dim = int(np.prod(env.action_space.shape))
    raw_steps = np.asarray(raw_action_block, dtype=np.float32)
    if raw_steps.shape != (int(action_block), action_dim):
        raise ValueError("reconfiguration raw action block has incompatible shape")
    _execute_raw_steps(env, raw_steps)
    root_after = np.asarray(env.render(), dtype=np.uint8).copy()
    flattened = raw_steps.reshape(-1).copy()
    return {
        "frames": frames[1:] + [root_before],
        "goal": np.asarray(context["goal"], dtype=np.uint8).copy(),
        "history_action_blocks": blocks[1:] + [flattened],
        "execution_snapshot": capture_execution_snapshot(env),
        "root_frame": root_after,
        "context_mode": "executed_warmup_plus_observed_action",
        "parent_context_mode": context.get("context_mode"),
        "history_rgb_sha256": [rgb_digest(frame) for frame in frames[1:] + [root_before]],
        "root_rgb_sha256": rgb_digest(root_after),
        "goal_rgb_sha256": rgb_digest(np.asarray(context["goal"], dtype=np.uint8)),
        "advanced_action_sha256": hashlib.sha256(np.ascontiguousarray(raw_steps).tobytes()).hexdigest(),
    }


def replay_executed_warmup(env: Any, *, seed: int, warmup_raw_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recreate a recorded planning root and return its RGB plus goal RGB."""
    _, info = env.reset(seed=int(seed))
    _execute_raw_steps(env, warmup_raw_steps)
    return np.asarray(env.render(), dtype=np.uint8).copy(), np.asarray(info["goal"], dtype=np.uint8).copy()
