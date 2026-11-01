"""CUDA-audited source-to-runtime contract for Cube D2P P1.

P0 can establish that a frozen checkpoint differentiates through its action
tokens using source-native HDF5 observations.  P1 additionally needs to
execute a selected packet from the *same* physical root.  This audit is the
gate between those claims: it restores source qpos/qvel and target provenance
into OGB-Cube, executes the logged next five actions, and compares the live
transition with the official source rows.  It never selects a candidate or
opens a P1 outcome.
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
import stable_worldmodel  # noqa: F401
import torch
from ogbench.manipspace import lie
from ogbench.manipspace.envs.cube_env import CubeEnv as VanillaCubeEnv

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import _image_transform
from experiments.spectral_reachability import cube_d2p_training_source as source


SCHEMA = "cube_d2p_source_runtime_transition.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Cube source/runtime state arrays are incompatible")
    return float(np.max(np.abs(actual - expected)))


def _restore_source_root(
    env: gym.Env, context: dict[str, Any], *, solver_warmstart: str = "zero",
) -> np.ndarray:
    """Restore evaluator-only physical state and target without model leakage."""

    evaluator = context["evaluator_only_source_state"]
    raw = env.unwrapped
    qpos = np.asarray(evaluator["qpos"], dtype=np.float64)
    qvel = np.asarray(evaluator["qvel"], dtype=np.float64)
    prev_qpos = np.asarray(evaluator["prev_qpos"], dtype=np.float64)
    prev_qvel = np.asarray(evaluator["prev_qvel"], dtype=np.float64)
    control = np.asarray(evaluator["control"], dtype=np.float64)
    time_value = np.asarray(evaluator["time"], dtype=np.float64)
    # The stable-worldmodel Cube subclass accepts a state option, but its reset
    # path immediately calls ``pre_step`` and therefore overwrites the
    # archived previous-state cache.  Initialise the compiled environment, then
    # restore the complete source state explicitly below.
    env.reset(seed=0)
    if prev_qpos.shape != qpos.shape or prev_qvel.shape != qvel.shape:
        raise ValueError("Cube source previous-state cache does not match qpos/qvel")
    if control.shape != raw.data.ctrl.shape or time_value.shape != (1,):
        raise ValueError("Cube source control/time snapshot differs from deployed simulator")
    target_block = int(np.asarray(evaluator["target_block"]).item())
    if target_block < 0 or target_block >= int(raw._num_cubes):
        raise ValueError("Cube source target block is outside the deployed environment")
    raw._target_block = target_block
    raw._target_task = str(evaluator["target_task"])
    target_yaw = float(np.asarray(evaluator["target_yaw"], dtype=np.float64).reshape(-1)[0])
    target_pos = np.asarray(evaluator["target_pos"], dtype=np.float64)
    target_quat = lie.SO3.from_z_radians(target_yaw).wxyz.tolist()
    if hasattr(raw, "set_target_pos"):
        raw.set_target_pos(target_block, target_pos, target_quat)
    else:
        target_mocap_id = int(raw._cube_target_mocap_ids[target_block])
        raw.data.mocap_pos[target_mocap_id] = target_pos
        raw.data.mocap_quat[target_mocap_id] = target_quat
    raw.set_state(qpos, qvel)
    # The archive does not expose MuJoCo's transient solver warm start.  Zero
    # the reset-time residue so the first replay step is a deterministic
    # function of the archived physical state rather than the unrelated reset
    # episode.  A later step recomputes this field normally.
    if solver_warmstart not in {"zero", "reconstruct_from_logged_control"}:
        raise ValueError("unsupported Cube solver warm-start strategy")
    raw.data.qacc_warmstart.fill(0.0)
    raw._prev_qpos = prev_qpos.copy()
    raw._prev_qvel = prev_qvel.copy()
    raw.data.ctrl[:] = control
    raw.data.time = float(time_value[0])
    mujoco.mj_forward(raw.model, raw.data)
    if solver_warmstart == "reconstruct_from_logged_control":
        # ``control[t]`` is the controller state that arrived with x_t.  A
        # source step has already solved the corresponding acceleration, but
        # HDF5 does not persist that solver cache.  Reconstruct the only
        # source-derived approximation available before replacing control with
        # a_t in env.step(); no model input or candidate selection uses it.
        raw.data.qacc_warmstart[:] = raw.data.qacc
    raw.post_step()
    return np.asarray(env.render(), dtype=np.uint8).copy()


def _source_successor(dataset: Path, root_row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the trainer-aligned successor, action packet, and controller trace.

    The HDF5 trainer packs ``action[t:t+5]`` alongside its five-frame RGB
    stride.  A CUDA offset probe binds each outgoing ``action[t+i]`` to the
    runtime controller snapshot stored as ``control[t+i+1]``: the control
    outgoing transition token.  Keep the trainer-aligned action packet and
    shift only its evaluator-only control trace by one row.  This prevents a
    false adapter mismatch from being mistaken for missing simulator
    provenance.
    """

    row = int(root_row) + source.ACTION_BLOCK
    with source.h5py.File(dataset, "r", swmr=True) as handle:
        return (
            np.asarray(handle["pixels"][row], dtype=np.uint8),
            np.asarray(handle["qpos"][row], dtype=np.float64),
            np.asarray(handle["qvel"][row], dtype=np.float64),
            np.asarray(handle["action"][root_row:row], dtype=np.float32),
            np.asarray(handle["control"][root_row + 1:row + 1], dtype=np.float64),
        )


def _source_predecessor_transition(
    dataset: Path, root_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the archived one-step transition that arrives at ``root_row``.

    This is a diagnostic only.  It tests whether an action/control recorded at
    row ``t`` belongs to the transition ``x[t-1] -> x[t]`` rather than to a
    transition leaving ``x[t]``.  That distinction is decisive for any frozen
    action-token deployment contract.
    """
    if int(root_row) < 1:
        raise ValueError("Cube predecessor transition needs a non-initial source row")
    with source.h5py.File(dataset, "r", swmr=True) as handle:
        previous = int(root_row) - 1
        return (
            np.asarray(handle["qpos"][previous], dtype=np.float64),
            np.asarray(handle["qvel"][previous], dtype=np.float64),
            np.asarray(handle["time"][previous], dtype=np.float64),
            np.asarray(handle["action"][root_row], dtype=np.float32),
            np.asarray(handle["control"][root_row], dtype=np.float64),
            np.asarray(handle["qpos"][root_row], dtype=np.float64),
            np.asarray(handle["qvel"][root_row], dtype=np.float64),
        )


def _source_successor_trace(dataset: Path, root_row: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the five archived arrival states for outgoing actions at x_t."""

    start = int(root_row) + 1
    stop = start + source.ACTION_BLOCK
    with source.h5py.File(dataset, "r", swmr=True) as handle:
        return (
            np.asarray(handle["qpos"][start:stop], dtype=np.float64),
            np.asarray(handle["qvel"][start:stop], dtype=np.float64),
        )


def _restore_source_root_with_episode_warmstart(
    env: gym.Env,
    context: dict[str, Any],
    *,
    dataset: Path,
    solver_warmstart: str,
) -> np.ndarray:
    """Restore a root, optionally reconstructing its unarchived solver cache.

    MuJoCo's ``qacc_warmstart`` is not an HDF5 observation column. It is part
    of the numerical state passed between constrained steps. The
    teacher-forced variant advances that cache from the source episode start
    while pinning each logged physical state before its action. It is
    evaluator-only: the reconstructed value never reaches the frozen model or
    candidate selector.
    """

    if solver_warmstart != "teacher_forced_episode":
        return _restore_source_root(env, context, solver_warmstart=solver_warmstart)

    root_row = int(context["source_rows"]["root_row"])
    with source.h5py.File(dataset, "r", swmr=True) as handle:
        episode_index = int(handle["ep_idx"][root_row])
        episode_start = int(handle["ep_offset"][episode_index])
        episode_stop = episode_start + int(handle["ep_len"][episode_index])
        if not (episode_start <= root_row < episode_stop):
            raise ValueError("Cube root is outside its declared source episode")
        rows = slice(episode_start, root_row)
        qpos = np.asarray(handle["qpos"][rows], dtype=np.float64)
        qvel = np.asarray(handle["qvel"][rows], dtype=np.float64)
        prev_qpos = np.asarray(handle["prev_qpos"][rows], dtype=np.float64)
        prev_qvel = np.asarray(handle["prev_qvel"][rows], dtype=np.float64)
        controls = np.asarray(handle["control"][rows], dtype=np.float64)
        times = np.asarray(handle["time"][rows], dtype=np.float64)
        actions = np.asarray(handle["action"][rows], dtype=np.float32)
    if not len(actions) or not all(np.isfinite(value).all() for value in (
        qpos, qvel, prev_qpos, prev_qvel, controls, times, actions,
    )):
        raise ValueError("Cube source episode prefix is empty or non-finite")

    _restore_source_root(env, context, solver_warmstart="zero")
    raw = env.unwrapped
    for state_qpos, state_qvel, state_prev_qpos, state_prev_qvel, control, time_value, action in zip(
        qpos, qvel, prev_qpos, prev_qvel, controls, times, actions,
    ):
        warmstart = np.asarray(raw.data.qacc_warmstart, dtype=np.float64).copy()
        raw.set_state(state_qpos, state_qvel)
        raw.data.qacc_warmstart[:] = warmstart
        raw._prev_qpos = state_prev_qpos.copy()
        raw._prev_qvel = state_prev_qvel.copy()
        raw.data.ctrl[:] = control
        raw.data.time = float(np.asarray(time_value).reshape(-1)[0])
        mujoco.mj_forward(raw.model, raw.data)
        raw.data.qacc_warmstart[:] = warmstart
        _observation, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            raise RuntimeError("Cube teacher-forced warm-start replay truncated")
    reconstructed_warmstart = np.asarray(raw.data.qacc_warmstart, dtype=np.float64).copy()
    root_rgb = _restore_source_root(env, context, solver_warmstart="zero")
    raw.data.qacc_warmstart[:] = reconstructed_warmstart
    return root_rgb


def _encode(model: torch.nn.Module, image: np.ndarray, *, transform: Any, device: torch.device) -> torch.Tensor:
    tensor = transform(image).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        value = model.encode({"pixels": tensor})["emb"][0, 0]
    if value.ndim != 1 or not torch.isfinite(value).all():
        raise RuntimeError("Cube frozen encoder returned invalid RGB embedding")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-dataset", type=Path, required=True)
    parser.add_argument("--training-source-receipt", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("/home/gbk/.stable_worldmodel/checkpoints/cube/lewm"))
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202610280)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--goal-offset-packets", type=int, default=1)
    parser.add_argument(
        "--solver-warmstart", choices=("zero", "reconstruct_from_logged_control", "teacher_forced_episode"), default="zero",
        help="Evaluator-only reconstruction for the unarchived MuJoCo solver cache.",
    )
    parser.add_argument("--runtime-backend", choices=("stable", "vanilla_ogbench"), default="stable")
    parser.add_argument(
        "--allow-legacy-runtime-training-contract", action="store_true",
        help="Read a SHA-bound existing receipt under a legacy physics runtime without importing its trainer API.",
    )
    parser.add_argument(
        "--skip-encoder-diagnostics", action="store_true",
        help="Run a physical-runtime-only audit when a legacy environment cannot instantiate the modern LeWM class.",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    if min(args.roots, args.history_len, args.goal_offset_packets) < 1:
        parser.error("roots and source context dimensions must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output, args.training_dataset = args.output.resolve(), args.training_dataset.resolve()
    args.training_source_receipt, args.policy = args.training_source_receipt.resolve(), args.policy.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite Cube source/runtime receipt: {args.output}")
    runtime = require_cuda_allocation(args.device)
    receipt = source.validate_receipt(
        args.training_source_receipt,
        verify_local_training_contract=not args.allow_legacy_runtime_training_contract,
    )
    source_meta = json.loads(args.training_source_receipt.read_text(encoding="utf-8"))["source"]
    if str(args.training_dataset) != str(Path(source_meta["dataset"]).resolve()):
        raise ValueError("Cube transition audit dataset differs from its receipt")
    device = torch.device("cuda")
    model = None
    transform = None
    if not args.skip_encoder_diagnostics:
        model = load_compatible_pretrained(args.policy, device=device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        transform = _image_transform()
    contexts, provenance = source.sample_source_native_contexts(
        args.training_dataset, expected_sha256=str(source_meta["dataset_sha256"]),
        history_len=args.history_len, goal_offset_packets=args.goal_offset_packets,
        count=args.roots, seed=args.seed,
    )
    # The offline trainer packs every five logged low-level actions into one
    # token.  Its dataset can contain an action suffix after the environment's
    # online success flag would fire, so an evaluation default of
    # ``terminate_at_goal=True`` truncates a source-valid token mid-packet.
    # Disable only that evaluator termination for this contract replay; the
    # success event remains recorded and never selects a candidate.
    # A fresh CUDA audit established that the archive aligns materially better
    # with the default ``front`` renderer than with ``front_pixels``.  Preserve
    # that empirically selected renderer rather than inferring it from the
    # HDF5 column name.
    # The source HDF5 includes ``privileged_target_*`` fields, which are
    # emitted only by OGBench's data-collection branch.  Replaying those rows
    # through the default task-mode constructor is therefore a runtime-domain
    # change even when qpos/qvel happen to restore exactly.  Bind the deployed
    # environment to the source mode before restoring any evaluator state.
    if args.runtime_backend == "stable":
        env = gym.make(
            "swm/OGBCube-v0", mode="data_collection", render_mode="rgb_array",
            terminate_at_goal=False,
        )
    else:
        env = VanillaCubeEnv(
            env_type="single", mode="data_collection", ob_type="pixels", width=224, height=224,
            terminate_at_goal=False,
        )
    rows: list[dict[str, Any]] = []
    try:
        for context, index in zip(contexts, provenance):
            root_rgb = _restore_source_root_with_episode_warmstart(
                env, context, dataset=args.training_dataset, solver_warmstart=args.solver_warmstart,
            )
            raw = env.unwrapped
            evaluator = context["evaluator_only_source_state"]
            root_qpos_error = _state_error(raw.data.qpos, np.asarray(evaluator["qpos"], dtype=np.float64))
            root_qvel_error = _state_error(raw.data.qvel, np.asarray(evaluator["qvel"], dtype=np.float64))
            expected_successor_rgb, expected_qpos, expected_qvel, successor_packet, successor_controls = _source_successor(
                args.training_dataset, int(context["source_rows"]["root_row"]),
            )
            if successor_packet.shape != (source.ACTION_BLOCK, source.RAW_ACTION_DIM) or not np.isfinite(successor_packet).all():
                raise ValueError("Cube source successor packet is invalid")
            if successor_controls.shape != (source.ACTION_BLOCK, raw.data.ctrl.size) or not np.isfinite(successor_controls).all():
                raise ValueError("Cube source successor controller trace is invalid")
            expected_qpos_trace, expected_qvel_trace = _source_successor_trace(
                args.training_dataset, int(context["source_rows"]["root_row"]),
            )
            if expected_qpos_trace.shape != (source.ACTION_BLOCK, raw.data.qpos.size) or expected_qvel_trace.shape != (source.ACTION_BLOCK, raw.data.qvel.size):
                raise ValueError("Cube source successor state trace is invalid")
            success_events: list[bool] = []
            action_control_errors: list[float] = []
            action_qpos_errors: list[float] = []
            action_qvel_errors: list[float] = []
            for action, expected_control, trace_qpos, trace_qvel in zip(
                successor_packet, successor_controls, expected_qpos_trace, expected_qvel_trace,
            ):
                _observation, _reward, terminated, truncated, _info = env.step(action)
                success_events.append(bool(_info.get("success", False)))
                action_control_errors.append(_state_error(raw.data.ctrl, expected_control))
                action_qpos_errors.append(_state_error(raw.data.qpos, trace_qpos))
                action_qvel_errors.append(_state_error(raw.data.qvel, trace_qvel))
                if terminated or truncated:
                    raise RuntimeError("Cube source replay truncated despite terminate_at_goal=False")
            successor_rgb = np.asarray(env.render(), dtype=np.uint8).copy()
            action_successor_qpos = np.asarray(raw.data.qpos, dtype=np.float64).copy()
            action_successor_qvel = np.asarray(raw.data.qvel, dtype=np.float64).copy()
            source_root = np.asarray(context["root_frame"], dtype=np.uint8)
            root_mae = float(np.mean(np.abs(root_rgb.astype(np.float32) - source_root.astype(np.float32))))
            successor_mae = float(np.mean(np.abs(successor_rgb.astype(np.float32) - expected_successor_rgb.astype(np.float32))))
            root_embedding_l2 = None if model is None else float(torch.linalg.vector_norm(
                _encode(model, root_rgb, transform=transform, device=device)
                - _encode(model, source_root, transform=transform, device=device),
            ))
            successor_embedding_l2 = None if model is None else float(torch.linalg.vector_norm(
                _encode(model, successor_rgb, transform=transform, device=device)
                - _encode(model, expected_successor_rgb, transform=transform, device=device),
            ))
            # This replay is evaluator-only: it intentionally bypasses the
            # public five-dimensional action interface with the logged 7-D
            # controller trace.  It does not license a planner to use control
            # values; it isolates whether any remaining mismatch is caused by
            # the runtime action adapter or by missing source physics state.
            _restore_source_root_with_episode_warmstart(
                env, context, dataset=args.training_dataset, solver_warmstart=args.solver_warmstart,
            )
            control_raw = env.unwrapped
            for control in successor_controls:
                control_raw.data.ctrl[:] = control
                control_raw.pre_step()
                mujoco.mj_step(control_raw.model, control_raw.data, nstep=control_raw._n_steps)
                mujoco.mj_rnePostConstraint(control_raw.model, control_raw.data)
                control_raw.post_step()
            # Determine the archive's raw-step convention independently of
            # the five-step token.  This is intentionally not a P1 execution:
            # it uses a logged predecessor state solely to bind the temporal
            # meaning of one stored action/control row.
            previous_qpos, previous_qvel, previous_time, arrival_action, arrival_control, arrival_qpos, arrival_qvel = _source_predecessor_transition(
                args.training_dataset, int(context["source_rows"]["root_row"]),
            )
            _restore_source_root_with_episode_warmstart(
                env, context, dataset=args.training_dataset, solver_warmstart=args.solver_warmstart,
            )
            predecessor_raw = env.unwrapped
            predecessor_raw.set_state(previous_qpos, previous_qvel)
            predecessor_raw.data.qacc_warmstart.fill(0.0)
            predecessor_raw.data.time = float(np.asarray(previous_time).reshape(-1)[0])
            mujoco.mj_forward(predecessor_raw.model, predecessor_raw.data)
            _observation, _reward, terminated, truncated, _info = env.step(arrival_action)
            if terminated or truncated:
                raise RuntimeError("Cube predecessor diagnostic truncated despite terminate_at_goal=False")
            rows.append({
                "source_provenance": index,
                "source_rows": context["source_rows"],
                "root_rgb_mae": root_mae,
                "root_qpos_max_abs_error": root_qpos_error,
                "root_qvel_max_abs_error": root_qvel_error,
                "successor_rgb_mae": successor_mae,
                "root_encoder_l2": root_embedding_l2,
                "successor_encoder_l2": successor_embedding_l2,
                "successor_qpos_max_abs_error": _state_error(action_successor_qpos, expected_qpos),
                "successor_qvel_max_abs_error": _state_error(action_successor_qvel, expected_qvel),
                "action_to_logged_control_max_abs_error": max(action_control_errors),
                "action_successor_qpos_max_abs_errors": action_qpos_errors,
                "action_successor_qvel_max_abs_errors": action_qvel_errors,
                "logged_control_successor_qpos_max_abs_error": _state_error(np.asarray(control_raw.data.qpos), expected_qpos),
                "logged_control_successor_qvel_max_abs_error": _state_error(np.asarray(control_raw.data.qvel), expected_qvel),
                "predecessor_action_to_logged_control_max_abs_error": _state_error(predecessor_raw.data.ctrl, arrival_control),
                "predecessor_action_arrival_qpos_max_abs_error": _state_error(predecessor_raw.data.qpos, arrival_qpos),
                "predecessor_action_arrival_qvel_max_abs_error": _state_error(predecessor_raw.data.qvel, arrival_qvel),
                "source_successor_packet_sha256": hashlib.sha256(np.ascontiguousarray(successor_packet).tobytes()).hexdigest(),
                "logged_packet_success_events": success_events,
                "target_provenance": context["evaluator_only_source_state"],
            })
    finally:
        env.close()
    # Exact state replay is non-negotiable.  RGB equality is deliberately a
    # separate diagnostic because the archive omits visual variation values;
    # P1 cannot claim source-native physical execution unless a later bound
    # establishes a suitable visual/render equivalence.
    max_qpos = max(row["successor_qpos_max_abs_error"] for row in rows)
    max_qvel = max(row["successor_qvel_max_abs_error"] for row in rows)
    max_root_qpos = max(row["root_qpos_max_abs_error"] for row in rows)
    max_root_qvel = max(row["root_qvel_max_abs_error"] for row in rows)
    report = {
        "schema": SCHEMA,
        "status": "pass" if max_root_qpos <= 1.0e-8 and max_root_qvel <= 1.0e-8 and max_qpos <= 1.0e-8 and max_qvel <= 1.0e-8 else "fail",
        "scope": "CUDA-only Cube source-to-runtime transition contract; no candidate selection, P1 mechanism, or efficacy outcome.",
        "cuda_host_preflight": {**runtime, "python": platform.python_version()},
        "inputs": {
            "training_dataset": str(args.training_dataset),
            "dataset_sha256": str(source_meta["dataset_sha256"]),
            "training_source_receipt": str(args.training_source_receipt),
            "training_source_receipt_sha256": _sha256(args.training_source_receipt),
            "checkpoint_weights_sha256": _sha256(args.policy / "weights.pt"),
            "training_action_mean_bound": receipt["calibration_action_mean"],
        },
        "protocol": {
            "root_construction": "same-episode HDF5 RGB/action/future-goal source-native contexts",
            "runtime_renderer": "default front renderer; a fresh CUDA comparison rejected front_pixels for this archive",
            "physical_replay": "construct OGBench in source data_collection mode; restore HDF5 qpos/qvel/prev-qpos/prev-qvel/control/time plus target block/position/yaw/task at x_t; reconstruct the declared evaluator-only solver warm-start strategy (including a teacher-forced source-episode cache when selected); execute frozen trainer-aligned logged a_t:a_(t+4); compare each resulting controller state to arrival-row control[t+1:t+6], and replay that evaluator-only control trace to diagnose adapter versus simulator provenance",
            "solver_warmstart": args.solver_warmstart,
            "runtime_backend": args.runtime_backend,
            "local_training_contract_verified": not args.allow_legacy_runtime_training_contract,
            "encoder_diagnostics": "skipped for legacy physics-only audit" if args.skip_encoder_diagnostics else "frozen published checkpoint",
            "online_termination": "terminate_at_goal=False so the source-valid five-action token is never cut short; success events are evaluator-only provenance",
            "model_input": "source RGB only; restored qpos/qvel/target remain evaluator-only",
            "visual_variation_contract": "archive has no explicit variation-values field; source/live RGB error is reported and is a required downstream P1 admission decision",
        },
        "gates": {
            "cuda_only": True,
            "source_receipt_validated": True,
            "same_episode_contexts": True,
            "restored_root_qpos_qvel_matches_source": bool(max_root_qpos <= 1.0e-8 and max_root_qvel <= 1.0e-8),
            "restored_root_qpos_qvel_matches_source_successor": bool(max_qpos <= 1.0e-8 and max_qvel <= 1.0e-8),
            "no_candidate_or_outcome_roots_opened": True,
            "evaluator_state_not_used_as_model_input": True,
        },
        "aggregate": {
            "max_root_qpos_abs_error": max_root_qpos,
            "max_root_qvel_abs_error": max_root_qvel,
            "max_successor_qpos_abs_error": max_qpos,
            "max_successor_qvel_abs_error": max_qvel,
            "max_action_to_logged_control_abs_error": max(row["action_to_logged_control_max_abs_error"] for row in rows),
            "max_logged_control_successor_qpos_abs_error": max(row["logged_control_successor_qpos_max_abs_error"] for row in rows),
            "max_logged_control_successor_qvel_abs_error": max(row["logged_control_successor_qvel_max_abs_error"] for row in rows),
            "max_predecessor_action_to_logged_control_abs_error": max(row["predecessor_action_to_logged_control_max_abs_error"] for row in rows),
            "max_predecessor_action_arrival_qpos_abs_error": max(row["predecessor_action_arrival_qpos_max_abs_error"] for row in rows),
            "max_predecessor_action_arrival_qvel_abs_error": max(row["predecessor_action_arrival_qvel_max_abs_error"] for row in rows),
            "root_rgb_mae": [row["root_rgb_mae"] for row in rows],
            "successor_rgb_mae": [row["successor_rgb_mae"] for row in rows],
            "root_encoder_l2": [row["root_encoder_l2"] for row in rows],
            "successor_encoder_l2": [row["successor_encoder_l2"] for row in rows],
        },
        "rows": rows,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": report["gates"], "aggregate": report["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
