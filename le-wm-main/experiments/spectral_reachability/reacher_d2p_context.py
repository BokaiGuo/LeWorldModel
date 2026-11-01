"""Executed temporal contexts for the frozen Reacher D2P protocol.

LeWM consumes one action token for every observed RGB frame and predicts the
next frame after the final token.  A planning root is consequently the state
*after* the recorded history, rather than the reset image used to start that
history.  These helpers make that causal boundary executable and replayable.
"""

from __future__ import annotations

import hashlib
import base64
import zlib
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


def pack_rgb_observation(value: np.ndarray) -> dict[str, Any]:
    pixels = np.asarray(value)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError("snapshot observation must be uint8 HWC RGB")
    return {"shape": list(pixels.shape), "sha256": rgb_digest(pixels),
            "uint8_zlib_base64": base64.b64encode(zlib.compress(np.ascontiguousarray(pixels).tobytes(), level=1)).decode("ascii")}


def unpack_rgb_observation(saved: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(value) for value in saved["shape"])
    if len(shape) != 3 or shape[-1] != 3 or min(shape) < 1:
        raise ValueError("stored snapshot observation shape is invalid")
    payload = zlib.decompress(base64.b64decode(saved["uint8_zlib_base64"], validate=True))
    image = np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()
    if rgb_digest(image) != saved.get("sha256"):
        raise ValueError("stored snapshot observation RGB hash mismatch")
    return image


def assert_exact_execution_state(expected: dict[str, Any], current: dict[str, Any] | None) -> None:
    required = ("physics_integration_state", "integration_state_spec", "model_binary_sha256", "target_qpos", "target_marker_pos", "rendered_goal_rgb")
    if current is None or any(key not in expected or key not in current for key in required):
        raise ValueError("sealed root observation needs a complete integration/scene snapshot")
    if any(current[key] != expected[key] for key in required):
        raise RuntimeError("sealed root observation has stale integration/scene state")


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


def capture_execution_snapshot(env: Any) -> dict[str, Any] | None:
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
    snapshot: dict[str, Any] = {
        "qpos": np.asarray(physics.data.qpos, dtype=np.float64).tolist(),
        "qvel": np.asarray(physics.data.qvel, dtype=np.float64).tolist(),
        "target_qpos": np.asarray(target, dtype=np.float64).tolist(),
    }
    # qpos/qvel can reproduce an RGB root but are not the complete causal
    # MuJoCo simulator state.  A checkpoint can appear to replay
    # from the same RGB root while an unrecorded actuator state changes the
    # next packet.  Preserve the full state whenever dm_control exposes it;
    # retain the component fields so lightweight tests and old receipts stay
    # readable.
    get_state = getattr(physics, "get_state", None)
    if callable(get_state):
        state = np.asarray(get_state(), dtype=np.float64).reshape(-1)
        if not np.isfinite(state).all():
            raise RuntimeError("execution snapshot contains a non-finite MuJoCo state")
        snapshot["physics_state"] = state.tolist()
        # dm_control.get_state() without a signature contains only qpos/qvel/
        # act/plugin state: it omits time, controls, applied forces and solver
        # warm-start.  Use the explicit integration signature for real MuJoCo
        # objects; retain the legacy vector for compatibility and diagnostics.
        if hasattr(getattr(physics, "model", None), "ptr") and hasattr(physics.data, "ptr"):
            import mujoco
            spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
            integration = np.asarray(get_state(sig=spec), dtype=np.float64).reshape(-1)
            if not np.isfinite(integration).all():
                raise RuntimeError("execution snapshot has non-finite integration state")
            snapshot["integration_state_spec"] = spec
            snapshot["physics_integration_state"] = integration.tolist()
            snapshot["model_binary_sha256"] = hashlib.sha256(physics.model.to_bytes()).hexdigest()
    # Model geometry is not part of ``physics.get_state()``.  The Reacher
    # goal wrapper binds the visible scene target to its qpos goal, so replay
    # must preserve that marker as well as the dynamic MuJoCo state.
    try:
        marker = np.asarray(physics.named.model.geom_pos["target"], dtype=np.float64).reshape(-1)
    except (AttributeError, KeyError, IndexError):
        marker = None
    if marker is not None:
        if not np.isfinite(marker).all():
            raise RuntimeError("execution snapshot contains a non-finite visible target marker")
        snapshot["target_marker_pos"] = marker.tolist()
    # Goal RGB is an immutable observation in this task. Re-rendering it
    # between two root identity checks needlessly changes graphics state and
    # can change a root pixel by one quantization unit, despite exact physics.
    # Bind the originally observed goal instead of synthesizing it on replay.
    goal = getattr(env, "_goal", None)
    if goal is not None:
        snapshot["rendered_goal_rgb"] = pack_rgb_observation(goal)
    return snapshot


def restore_execution_snapshot(
    env: Any, *, seed: int, snapshot: dict[str, Any],
    root_identity: str = "exact_render",
) -> tuple[np.ndarray, np.ndarray]:
    """Reset bookkeeping, then restore an evaluator-only observed-root state.

    A long planner call exposed a rare reset-and-replay RGB-identity failure.
    Restoring the root actually observed by the planner removes that fragile
    reconstruction dependency while preserving a strict rendered-RGB check.
    """
    if root_identity not in ("exact_render", "sealed_observation_exact_state"):
        raise ValueError("unknown root identity protocol")
    if root_identity == "sealed_observation_exact_state":
        # Validate the bound observation before mutating the simulator. This
        # branch never accepts a substitute or a near-equal rendered image.
        if "observed_root_rgb" not in snapshot or "rendered_goal_rgb" not in snapshot:
            raise ValueError("sealed root observation missing from execution snapshot")
        sealed_root = unpack_rgb_observation(snapshot["observed_root_rgb"])
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
    full_state = snapshot.get("physics_state")
    set_state = getattr(physics, "set_state", None)
    integration_state = snapshot.get("physics_integration_state")
    integration_spec = snapshot.get("integration_state_spec")
    if (integration_state is None) != (integration_spec is None):
        raise ValueError("execution snapshot has an incomplete integration-state contract")
    if integration_state is not None and (full_state is None or not callable(set_state)):
        raise ValueError("integration snapshot requires the complete physics-state restore interface")
    if full_state is not None and callable(set_state):
        state = np.asarray(full_state, dtype=np.float64).reshape(-1)
        if not np.isfinite(state).all():
            raise ValueError("execution snapshot full MuJoCo state is non-finite")
        if integration_state is not None:
            packed = np.asarray(integration_state, dtype=np.float64).reshape(-1)
            if not np.isfinite(packed).all():
                raise ValueError("execution snapshot integration state is non-finite")
            set_state(packed, sig=int(integration_spec))
            if not np.array_equal(np.asarray(physics.get_state(sig=int(integration_spec))), packed):
                raise RuntimeError("integration snapshot did not restore exactly")
        else:
            set_state(state)
        # A malformed or version-incompatible full-state vector must never
        # silently restore only an RGB-equivalent state.
        if not np.allclose(np.asarray(physics.data.qpos), qpos, rtol=0.0, atol=1.0e-12) or not np.allclose(np.asarray(physics.data.qvel), qvel, rtol=0.0, atol=1.0e-12):
            raise RuntimeError("full execution snapshot does not reproduce stored qpos/qvel")
        replayed_state = getattr(physics, "get_state", None)
        if callable(replayed_state) and not np.allclose(
            np.asarray(replayed_state(), dtype=np.float64).reshape(-1), state, rtol=0.0, atol=1.0e-12,
        ):
            raise RuntimeError("full execution snapshot does not reproduce the complete MuJoCo state")
    else:
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel
    marker = snapshot.get("target_marker_pos")
    if marker is not None:
        try:
            restored_marker = np.asarray(marker, dtype=np.float64).reshape(-1)
            live_marker = np.asarray(physics.named.model.geom_pos["target"], dtype=np.float64).reshape(-1)
        except (AttributeError, KeyError, IndexError) as error:
            raise RuntimeError("execution snapshot target marker is incompatible with Reacher") from error
        if restored_marker.shape != live_marker.shape or not np.isfinite(restored_marker).all():
            raise ValueError("execution snapshot target marker is incompatible with Reacher")
        physics.named.model.geom_pos["target"] = restored_marker
    def refresh_restored_root():
        # mj_forward recomputes derived fields but can also change the causal
        # qacc_warmstart integration field by floating-point roundoff. Restore
        # the full integration vector AFTER refreshing derived state, without
        # a second forward. qpos/qvel and scene are unchanged, so derived
        # positions used by rendering remain valid while the next mj_step
        # starts from the exact recorded solver state.
        physics.forward()
        if integration_state is not None:
            set_state(packed, sig=int(integration_spec))
            if not np.array_equal(np.asarray(physics.get_state(sig=int(integration_spec))), packed):
                raise RuntimeError("post-forward integration snapshot did not restore exactly")
    refresh_restored_root()
    root = np.asarray(env.render(), dtype=np.uint8).copy() if root_identity == "exact_render" else sealed_root
    saved_goal = snapshot.get("rendered_goal_rgb")
    if saved_goal is not None:
        goal = unpack_rgb_observation(saved_goal)
        if hasattr(env, "_goal"):
            env._goal = goal.copy()
        if root_identity == "sealed_observation_exact_state":
            assert_exact_execution_state(snapshot, capture_execution_snapshot(env))
        return root, goal
    # The task target was generated at reset, not from the state after the
    # warmup.  Render the stored target directly; calling the wrapper helper
    # with ``qpos`` would incorrectly synthesize ``root + target_delta``.
    # Rendering the task goal temporarily changes the simulator state.  Do
    # not leave a partially restored root behind afterwards: qpos/qvel alone
    # are insufficient for a general MuJoCo state (for example, an actuator
    # or auxiliary dynamic state is not represented there).  This matters in
    # P1, where the first physical prefix is executed only after the goal image
    # has been reconstructed.
    physics.data.qpos[:] = target
    physics.data.qvel[:] = 0.0
    physics.forward()
    try:
        goal = np.asarray(raw.render(width=224, height=224), dtype=np.uint8).copy()
    except TypeError:
        # Lightweight unit-test environments need not implement sized renders.
        goal = np.asarray(info["goal"], dtype=np.uint8).copy()
    if full_state is not None and callable(set_state):
        # Reapply the exact state captured at the observed planning root,
        # rather than relying on the qpos/qvel approximation used solely for
        # the temporary goal render above.
        if integration_state is not None:
            set_state(packed, sig=int(integration_spec))
            if not np.array_equal(np.asarray(physics.get_state(sig=int(integration_spec))), packed):
                raise RuntimeError("goal rendering changed restored integration state")
        else:
            set_state(state)
        replayed_state = getattr(physics, "get_state", None)
        if callable(replayed_state) and not np.allclose(
            np.asarray(replayed_state(), dtype=np.float64).reshape(-1), state,
            rtol=0.0, atol=1.0e-12,
        ):
            raise RuntimeError("goal rendering changed the restored complete MuJoCo state")
    else:
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = qvel
    # Model geometry lies outside ``get_state``; preserving it is required to
    # retain the marker/goal RGB contract after a full-state restore.
    if marker is not None:
        physics.named.model.geom_pos["target"] = restored_marker
    refresh_restored_root()
    if hasattr(env, "_goal"):
        env._goal = goal.copy()
    return root, goal


def collect_executed_context(
    env: Any,
    *,
    seed: int,
    history_len: int,
    action_block: int,
    warmup_packets: np.ndarray | None = None,
    warmup_policy: str = WARMUP_POLICY,
) -> dict[str, Any]:
    """Reset, execute a causal history, and return the resulting plan root."""
    _, info = env.reset(seed=int(seed))
    goal = np.asarray(info["goal"], dtype=np.uint8).copy()
    action_dim = int(np.prod(env.action_space.shape))
    if warmup_packets is None:
        blocks = warmup_action_blocks(
            seed=seed,
            history_len=history_len,
            action_block=action_block,
            action_low=np.asarray(env.action_space.low),
            action_high=np.asarray(env.action_space.high),
        )
    else:
        packets = np.asarray(warmup_packets, dtype=np.float32)
        if packets.shape != (int(history_len), int(action_block), action_dim) or not np.isfinite(packets).all():
            raise ValueError("supported Reacher warmup packets have an incompatible full-packet layout")
        low, high = np.asarray(env.action_space.low), np.asarray(env.action_space.high)
        if np.any(packets < low[None, None] - 1.0e-6) or np.any(packets > high[None, None] + 1.0e-6):
            raise ValueError("supported Reacher warmup packet leaves the live action box")
        blocks = [packet.reshape(-1).copy() for packet in packets]
    frames: list[np.ndarray] = []
    raw_steps: list[np.ndarray] = []
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
        "warmup_policy": str(warmup_policy),
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
    verified_root_frame: np.ndarray | None = None,
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
    if verified_root_frame is None:
        root_before = np.asarray(env.render(), dtype=np.uint8).copy()
        root_validation = "fresh_strict_rgb"
    else:
        # The caller has just compared an actual restored render byte-for-byte
        # with this context. Do not render it twice: repeated OpenGL renders
        # can differ by one quantization unit even at exactly the same state.
        # Instead prove no physical OR compiled scene state changed since that
        # verified observation. This is not an image tolerance or a bypass for
        # legacy/incomplete snapshots.
        root_before = np.asarray(verified_root_frame, dtype=np.uint8).copy()
        expected = context.get("execution_snapshot")
        current = capture_execution_snapshot(env)
        required = ("physics_integration_state", "integration_state_spec", "model_binary_sha256", "target_qpos", "rendered_goal_rgb")
        if expected is None or current is None or any(key not in expected or key not in current for key in required):
            raise ValueError("verified RGB root needs a complete integration/scene snapshot")
        if any(current[key] != expected[key] for key in required):
            raise RuntimeError("verified RGB root has stale integration/scene state before reconfiguration")
        root_validation = "strict_restored_rgb_then_fresh_exact_integration_and_scene"
    if not np.array_equal(root_before, np.asarray(context["root_frame"], dtype=np.uint8)):
        expected_root = np.asarray(context["root_frame"], dtype=np.uint8)
        details = {"expected_shape": list(expected_root.shape), "actual_shape": list(root_before.shape)}
        if expected_root.shape == root_before.shape:
            details.update(max_pixel_error=int(np.abs(root_before.astype(np.int16) - expected_root.astype(np.int16)).max()),
                           changed_components=int(np.count_nonzero(root_before != expected_root)))
        snapshot = context.get("execution_snapshot")
        current = capture_execution_snapshot(env)
        if snapshot is not None and current is not None:
            for key in ("qpos", "qvel", "physics_state", "physics_integration_state", "target_marker_pos", "target_qpos"):
                if key in snapshot and key in current:
                    left, right = np.asarray(snapshot[key]), np.asarray(current[key])
                    details[key + "_max_error"] = float(np.max(np.abs(left - right))) if left.shape == right.shape else None
        raise RuntimeError("execution context root differs before reconfiguration: " + str(details))
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
        "root_identity_validation": root_validation,
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
