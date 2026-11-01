"""Build a CUDA-gated, fully replayable live Cube panel for D2P P1.

The official Cube HDF5 is an admissible *training* source, but it lacks the
runtime manifest and solver state required to replay its archived qpos/qvel
transitions exactly.  This builder does not retrofit those missing facts.  It
creates new roots under one explicitly recorded runtime, using only
training-supported physical action packets for the causal RGB history.  It
then records the complete MuJoCo physics state and verifies the held-out next
packet by exact same-runtime replay.  The result is a P1 input contract, not a
candidate selector, mechanism result, or outcome experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import stable_worldmodel  # noqa: F401 -- environment registration

from experiments.gpu_preflight import require_cuda_allocation
from experiments.spectral_reachability import cube_d2p_training_source as source


SCHEMA = "cube_d2p_live_runtime_panel.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _integration_state_spec() -> mujoco.mjtState:
    """Capture all state which can affect the next integration step.

    ``mjSTATE_FULLPHYSICS`` omits controls and mocap values.  Those omissions
    are harmless for a static state dump but break an action-conditioned
    successor replay.  ``mjSTATE_INTEGRATION`` includes them along with the
    physics/solver fields, so it is the minimum honest live-root receipt.
    """

    return mujoco.mjtState.mjSTATE_INTEGRATION


def _capture_state(raw: Any) -> dict[str, Any]:
    spec = _integration_state_spec()
    packed = np.empty(mujoco.mj_stateSize(raw.model, spec), dtype=np.float64)
    mujoco.mj_getState(raw.model, raw.data, packed, spec)
    return {
        "mujoco_state_spec": int(spec),
        "packed_integration_state": packed.tolist(),
        "previous_qpos": np.asarray(raw._prev_qpos, dtype=np.float64).tolist(),
        "previous_qvel": np.asarray(raw._prev_qvel, dtype=np.float64).tolist(),
        "target_block": int(raw._target_block),
        "target_task": str(raw._target_task),
        "target_pos": np.asarray(raw.data.mocap_pos[raw._cube_target_mocap_ids[raw._target_block]], dtype=np.float64).tolist(),
        "target_quat": np.asarray(raw.data.mocap_quat[raw._cube_target_mocap_ids[raw._target_block]], dtype=np.float64).tolist(),
        # MuJoCo state serialization intentionally excludes mutable model
        # appearance.  CubeEnv.post_step mutates these colors, so they are
        # part of the rendered root contract even though they are not physics.
        "model_geom_rgba": np.asarray(raw.model.geom_rgba, dtype=np.float64).tolist(),
        "success": bool(raw._success),
        "reset_next_step": bool(raw._reset_next_step),
        "restore_reference": {
            "qpos": np.asarray(raw.data.qpos, dtype=np.float64).tolist(),
            "qvel": np.asarray(raw.data.qvel, dtype=np.float64).tolist(),
            "ctrl": np.asarray(raw.data.ctrl, dtype=np.float64).tolist(),
            "qacc_warmstart": np.asarray(raw.data.qacc_warmstart, dtype=np.float64).tolist(),
            # OGBench uses mj_step(..., nstep=25), whose final integration
            # leaves these controller inputs at the preceding forward stage.
            # ManipSpaceEnv.set_control deliberately consumes them before the
            # next physics forward pass, so they are causal runtime state.
            "site_xpos": np.asarray(raw.data.site_xpos, dtype=np.float64).tolist(),
            "site_xmat": np.asarray(raw.data.site_xmat, dtype=np.float64).tolist(),
        },
    }


def _restore_state(raw: Any, snapshot: dict[str, Any]) -> None:
    spec = mujoco.mjtState(int(snapshot["mujoco_state_spec"]))
    packed = np.asarray(snapshot["packed_integration_state"], dtype=np.float64)
    expected = mujoco.mj_stateSize(raw.model, spec)
    if packed.shape != (expected,) or not np.isfinite(packed).all():
        raise ValueError("Cube live panel has an invalid integration-state snapshot")
    raw._target_block = int(snapshot["target_block"])
    raw._target_task = str(snapshot["target_task"])
    target_mocap = raw._cube_target_mocap_ids[raw._target_block]
    raw.data.mocap_pos[target_mocap] = np.asarray(snapshot["target_pos"], dtype=np.float64)
    raw.data.mocap_quat[target_mocap] = np.asarray(snapshot["target_quat"], dtype=np.float64)
    mujoco.mj_setState(raw.model, raw.data, packed, spec)
    raw._prev_qpos = np.asarray(snapshot["previous_qpos"], dtype=np.float64).copy()
    raw._prev_qvel = np.asarray(snapshot["previous_qvel"], dtype=np.float64).copy()
    raw.model.geom_rgba[:] = np.asarray(snapshot["model_geom_rgba"], dtype=np.float64)
    raw._success = bool(snapshot["success"])
    raw._reset_next_step = bool(snapshot["reset_next_step"])
    # Do not call mj_forward here.  Under OGBench's 25-substep mj_step, the
    # final integration advances qpos/qvel while leaving site transforms from
    # the preceding forward pass.  The next set_control() reads precisely
    # those cached site transforms.  Recomputing them would silently turn a
    # same-runtime restore into a different controller trajectory.  Restore
    # the causal cache verbatim; the following mj_step will perform its own
    # ordinary forward pass after set_control has populated ctrl.
    controller_cache = snapshot["restore_reference"]
    raw.data.site_xpos[:] = np.asarray(controller_cache["site_xpos"], dtype=np.float64)
    raw.data.site_xmat[:] = np.asarray(controller_cache["site_xmat"], dtype=np.float64)


def _restore_diagnostics(raw: Any, snapshot: dict[str, Any]) -> dict[str, float]:
    """Report every field the next Cube action can consume after restore.

    These are diagnostics, not tolerances: exact replay remains the only
    admission gate.  Their role is to distinguish an incomplete restore from
    numerical divergence inside a fully restored MuJoCo step.
    """

    reference = snapshot["restore_reference"]
    fields = {
        "qpos": raw.data.qpos,
        "qvel": raw.data.qvel,
        "ctrl": raw.data.ctrl,
        "qacc_warmstart": raw.data.qacc_warmstart,
        "site_xpos": raw.data.site_xpos,
        "site_xmat": raw.data.site_xmat,
    }
    return {
        name + "_max_abs_error": float(
            np.max(np.abs(np.asarray(value, dtype=np.float64) - np.asarray(reference[name], dtype=np.float64)))
        )
        for name, value in fields.items()
    }


def _step_packet(env: gym.Env, packet: np.ndarray) -> None:
    for action in np.asarray(packet, dtype=np.float32):
        _observation, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            raise RuntimeError("live Cube panel packet terminated unexpectedly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--training-source-receipt", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202610290)
    parser.add_argument("--history-packets", type=int, default=3)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if min(args.roots, args.history_packets) < 1:
        parser.error("roots and history-packets must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.training_dataset = args.training_dataset.resolve()
    args.training_source_receipt = args.training_source_receipt.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable Cube live panel: {args.output}")
    cuda = require_cuda_allocation(args.device)
    receipt = source.validate_receipt(args.training_source_receipt)
    receipt_source = json.loads(args.training_source_receipt.read_text(encoding="utf-8"))["source"]
    if args.training_dataset != Path(receipt_source["dataset"]).resolve():
        raise ValueError("live Cube panel dataset differs from its SHA-bound training receipt")
    sequences, provenance = source.sample_contiguous_packet_sequences(
        args.training_dataset,
        expected_sha256=str(receipt_source["dataset_sha256"]),
        packets_per_sequence=args.history_packets + 1,
        count=args.roots,
        seed=args.seed,
    )
    env = gym.make("swm/OGBCube-v0", mode="data_collection", render_mode="rgb_array", terminate_at_goal=False)
    rows: list[dict[str, Any]] = []
    model_history_rgb: list[np.ndarray] = []
    model_root_rgb: list[np.ndarray] = []
    model_goal_rgb: list[np.ndarray] = []
    model_history_packets: list[np.ndarray] = []
    model_heldout_packets: list[np.ndarray] = []
    try:
        for index, (sequence, packet_provenance) in enumerate(zip(sequences, provenance)):
            root_seed = int(args.seed + index)
            env.reset(seed=root_seed)
            frames: list[np.ndarray] = [np.asarray(env.render(), dtype=np.uint8).copy()]
            for packet in sequence[:args.history_packets]:
                _step_packet(env, packet)
                frames.append(np.asarray(env.render(), dtype=np.uint8).copy())
            snapshot = _capture_state(env.unwrapped)
            root_rgb = frames[-1]
            future_packet = np.asarray(sequence[-1], dtype=np.float32)
            _step_packet(env, future_packet)
            expected_qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64).copy()
            expected_qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64).copy()
            expected_rgb = np.asarray(env.render(), dtype=np.uint8).copy()
            expected_rgb_repeat = np.asarray(env.render(), dtype=np.uint8).copy()
            _restore_state(env.unwrapped, snapshot)
            restore_diagnostics = _restore_diagnostics(env.unwrapped, snapshot)
            _step_packet(env, future_packet)
            replay_qpos = np.asarray(env.unwrapped.data.qpos, dtype=np.float64)
            replay_qvel = np.asarray(env.unwrapped.data.qvel, dtype=np.float64)
            replay_rgb = np.asarray(env.render(), dtype=np.uint8)
            replay_rgb_repeat = np.asarray(env.render(), dtype=np.uint8)
            qpos_error = float(np.max(np.abs(replay_qpos - expected_qpos)))
            qvel_error = float(np.max(np.abs(replay_qvel - expected_qvel)))
            rgb_error = float(np.max(np.abs(replay_rgb.astype(np.int16) - expected_rgb.astype(np.int16))))
            expected_repeat_error = float(np.max(np.abs(expected_rgb_repeat.astype(np.int16) - expected_rgb.astype(np.int16))))
            replay_repeat_error = float(np.max(np.abs(replay_rgb_repeat.astype(np.int16) - replay_rgb.astype(np.int16))))
            renderer_envelope = max(expected_repeat_error, replay_repeat_error)
            model_history_rgb.append(np.stack(frames[:-1], axis=0))
            model_root_rgb.append(root_rgb)
            model_goal_rgb.append(expected_rgb)
            model_history_packets.append(np.asarray(sequence[:args.history_packets], dtype=np.float32))
            model_heldout_packets.append(future_packet)
            rows.append({
                "root_id": _array_hash(np.frombuffer((str(root_seed) + _array_hash(root_rgb)).encode(), dtype=np.uint8)),
                "root_seed": root_seed,
                "training_packet_provenance": packet_provenance,
                "history_frame_sha256": [_array_hash(frame) for frame in frames[:-1]],
                "root_rgb_sha256": _array_hash(root_rgb),
                "history_packets_sha256": _array_hash(sequence[:args.history_packets]),
                "heldout_next_packet_sha256": _array_hash(future_packet),
                "goal_rgb_sha256": _array_hash(expected_rgb),
                "evaluator_only_root_state": snapshot,
                "heldout_successor": {
                    "qpos": expected_qpos.tolist(), "qvel": expected_qvel.tolist(),
                    "rgb_sha256": _array_hash(expected_rgb),
                },
                "exact_replay": {
                    "qpos_max_abs_error": qpos_error,
                    "qvel_max_abs_error": qvel_error,
                    "rgb_max_abs_error": rgb_error,
                    "expected_render_repeat_max_abs_error": expected_repeat_error,
                    "replay_render_repeat_max_abs_error": replay_repeat_error,
                    "same_state_renderer_envelope_max_abs_error": renderer_envelope,
                },
                "restore_before_heldout_packet": restore_diagnostics,
            })
    finally:
        env.close()
    gates = {
        "cuda_only": True,
        "training_receipt_valid": True,
        "source_supported_contiguous_packet_history": True,
        "live_root_state_never_enters_model_input": True,
        "all_heldout_packets_physics_exactly_replay": bool(rows) and all(
            row["exact_replay"]["qpos_max_abs_error"] == 0.0
            and row["exact_replay"]["qvel_max_abs_error"] == 0.0
            for row in rows
        ),
        "all_successor_rgb_within_same_state_renderer_envelope": bool(rows) and all(
            row["exact_replay"]["rgb_max_abs_error"]
            <= row["exact_replay"]["same_state_renderer_envelope_max_abs_error"]
            for row in rows
        ),
        "no_p1_candidate_or_outcome_selected": True,
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "CUDA-only live Cube P1 root contract; no candidate selection, mechanism claim, or outcome roots opened.",
        "cuda_host_preflight": {**cuda, "python": platform.python_version()},
        "inputs": {
            "training_dataset": str(args.training_dataset),
            "training_dataset_sha256": str(receipt_source["dataset_sha256"]),
            "training_source_receipt": str(args.training_source_receipt),
            "training_source_receipt_sha256": _sha256(args.training_source_receipt),
            "mujoco_version": mujoco.__version__,
            "environment": source.ENVIRONMENT,
            "action_block": source.ACTION_BLOCK,
            "raw_action_dim": source.RAW_ACTION_DIM,
        },
        "protocol": {
            "model_input_if_consumed_by_later_p1": "live RGB history, root RGB, held-out live goal RGB, and source-supported causal action packets only",
            "evaluator_only": "complete MuJoCo physics state and successor qpos/qvel",
            "history_packets": args.history_packets,
            "root_seeds": [row["root_seed"] for row in rows],
            "root_family": "fresh live runtime; disjoint from official archive roots",
        },
        "gates": gates,
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "model_inputs.npz",
        history_rgb=np.stack(model_history_rgb, axis=0),
        root_rgb=np.stack(model_root_rgb, axis=0),
        goal_rgb=np.stack(model_goal_rgb, axis=0),
        history_packets=np.stack(model_history_packets, axis=0),
        heldout_packets=np.stack(model_heldout_packets, axis=0),
    )
    report["model_inputs"] = {
        "path": "model_inputs.npz",
        "sha256": _sha256(args.output / "model_inputs.npz"),
        "arrays": ["history_rgb", "root_rgb", "goal_rgb", "history_packets", "heldout_packets"],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
