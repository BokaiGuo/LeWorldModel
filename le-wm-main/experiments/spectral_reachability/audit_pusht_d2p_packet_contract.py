"""Neural-free, bounded official-HDF5/physical PushT contract sidecar.

Default selection is EXACT episodes 0..15, first 20 dense transitions each.
Invalid declared data fails closed: no clipping, dropping, or replacement by
better replay outcomes. No checkpoint is opened; this is not P0/P1 admission.
Run with ``python -m experiments.spectral_reachability.audit_pusht_d2p_packet_contract``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability.pusht_d2p_packet_contract import (
    ACTION_BLOCK, REQUIRED_GATES, VERSION, BoundaryHistory, contract_spec,
    packet_commands, projection_identity, render_without_advancing,
    repeat_command, require_shared_suffix_projection, state_of,
    update_observed_context,
)


DEFAULT_DATA = Path("/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5")
# Fixed diagnostic reset goal, NOT a recovered HDF5 training-goal annotation.
# Its block pose agrees with the official default target marker.
DIAGNOSTIC_GOAL = np.array([256., 400., 256., 256., np.pi / 4, 0., 0.])
DIAGNOSTIC_GOAL.setflags(write=False)
WINDOW_COMMANDS = 20  # three warmup packets + one prefix, plus recorded successor
BOUNDARIES = (0, 5, 10, 15, 20)
FIRST_TRANSITION_ATOL = 1e-3  # fixed diagnostic precision; no CLI gate relaxation
PUBLIC_CONVENTION = {
    "mean": [-0.0087, 0.0068], "std": [0.2019, 0.2002],
    "status": "unverified_public_stats_convention_only",
    "source": "experiments/run_online_pusht_planning.py:PublicPushTActionTransform",
    "checkpoint_training_truth_verified": False,
    "used_to_select_rows_or_outcomes": False,
    "used_for_neural_experiment": False,
}


def _array_hash(value):
    value = np.asarray(value)
    digest = hashlib.sha256(json.dumps([value.dtype.str, value.shape]).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _file_hash(path):
    """Only for bounded source/receipt/artifact files; NEVER the HDF5 archive."""
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError(f"refusing full hash of large source/receipt: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_stat(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _validate_indices(indices):
    if (not indices or any(type(i) is not int or i < 0 for i in indices) or
            len(set(indices)) != len(indices)):
        raise ValueError("predeclared episode indices must be unique nonnegative integers")


def _stored_terminal(states):
    pos = np.linalg.norm(states[:, :4] - DIAGNOSTIC_GOAL[:4], axis=1)
    angle = np.abs(states[:, 4] - DIAGNOSTIC_GOAL[4])
    angle = np.minimum(angle, 2 * np.pi - angle)
    return (pos < 20) & (angle < np.pi / 9)


def select_rows(path, indices):
    """Bounded hyperslabs only. Validate ALL declared episodes before physics.

    Terminal filtering is solely a predeclared stored-state predicate relative
    to the fixed diagnostic goal; failures abort, never select a replacement.
    Selected-row digests are not full-archive provenance or checkpoint stats.
    """
    _validate_indices(indices)
    import h5py
    try:
        import hdf5plugin  # noqa: F401 -- needed by the official pixel compression
    except ModuleNotFoundError:
        pass  # uncompressed synthetic data works; compressed reads fail explicitly

    path = Path(path)
    before = _source_stat(path)
    rows, hashes, transitions = [], [], []
    with h5py.File(path, "r") as data:
        shapes = {key: list(data[key].shape) for key in (
            "action", "state", "pixels", "episode_idx", "step_idx", "ep_len", "ep_offset")}
        n = shapes["state"][0]
        if (shapes["state"] != [n, 7] or shapes["action"] != [n, 2] or
                shapes["pixels"] != [n, 224, 224, 3] or
                shapes["episode_idx"] != [n] or shapes["step_idx"] != [n] or
                len(shapes["ep_len"]) != 1 or shapes["ep_offset"] != shapes["ep_len"]):
            raise ValueError("official HDF5 field layout mismatch")
        for episode in indices:
            if episode >= shapes["ep_len"][0]:
                raise ValueError(f"declared episode {episode} does not exist")
            offset, length = int(data["ep_offset"][episode]), int(data["ep_len"][episode])
            if length <= WINDOW_COMMANDS or offset < 0 or offset + length > n:
                raise ValueError(f"declared episode {episode}: insufficient/invalid length")
            stop = offset + WINDOW_COMMANDS + 1
            state = data["state"][offset:stop]
            action = data["action"][offset:stop - 1]
            episode_ids = data["episode_idx"][offset:stop]
            step_ids = data["step_idx"][offset:stop]
            if (not np.array_equal(episode_ids, np.full(21, episode)) or
                    not np.array_equal(step_ids, np.arange(21))):
                raise ValueError(f"declared episode {episode}: noncontiguous transition IDs")
            if not np.isfinite(state).all():
                raise ValueError(f"declared episode {episode}: nonfinite states")
            for packet in action.reshape(4, ACTION_BLOCK, 2):
                packet_commands(packet)  # strict finite/bounds gate, never clip
            if _stored_terminal(state).any():
                raise ValueError(f"declared episode {episode}: terminal stored window for fixed diagnostic goal")
            pixels = np.stack([data["pixels"][offset + k] for k in BOUNDARIES])
            if pixels.dtype != np.uint8:
                raise ValueError("official archived pixels must be uint8")
            rows.append({"episode": episode, "offset": offset, "states": state,
                         "actions": action, "pixels": pixels})
            hashes.append({"episode": episode, "state": _array_hash(state),
                           "action": _array_hash(action), "pixels_at_boundaries": _array_hash(pixels),
                           "episode_idx": _array_hash(episode_ids), "step_idx": _array_hash(step_ids)})
            transitions.extend({"episode": episode, "step": k, "row": offset + k,
                                "next_row": offset + k + 1} for k in range(WINDOW_COMMANDS))
    if before != _source_stat(path):
        raise ValueError("source data changed during bounded selection")
    return rows, {
        "source": before, "source_binding": "selected_rows_only_not_full_archive_hash",
        "selection": "predeclared_exact_episode_indices_no_outcome_replacement",
        "episode_indices": list(indices), "commands_per_episode": WINDOW_COMMANDS,
        "dataset_shapes": shapes, "transition_ids": transitions, "selected_hashes": hashes,
    }


def _write_json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)


def _snapshot_implementation(output):
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(__file__).with_name("pusht_d2p_packet_contract.py"),
               root / "tests/test_pusht_d2p_packet_contract.py",
               root / "experiments/pusht_protocol_contracts.py",
               root / "experiments/run_online_pusht_planning.py",
               root / "config/train/data/pusht.yaml"]
    spec = importlib.util.find_spec("stable_worldmodel")
    if spec is not None and spec.submodule_search_locations:
        installed = Path(next(iter(spec.submodule_search_locations)))
        sources.extend(installed / p for p in (
            "envs/pusht/env.py", "data/dataset.py", "data/formats/hdf5.py"))
    dest = output / "implementation"
    dest.mkdir()
    manifest = {"sources": [], "missing_sources": [],
                "note": "bounded code snapshots only; no dataset/checkpoint full hash"}
    for i, source in enumerate(sources):
        if not source.is_file():
            manifest["missing_sources"].append(str(source))
            continue
        original_hash = _file_hash(source)
        target = dest / f"{i:02d}_{source.name}"
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        target.chmod(0o444)
        if _file_hash(target) != original_hash:
            raise ValueError(f"implementation source changed during snapshot: {source}")
        manifest["sources"].append({"source": str(source), "snapshot": str(target.relative_to(output)),
                                    "sha256": original_hash})
    _write_json_new(output / "implementation.json", manifest)


def official_env_factory():
    """Lazy official simulator registration only; never constructs/loads a model."""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401 -- registers the official environment

    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                   relative=True, render_action=False, with_target=True)
    base = env.unwrapped
    if (not base.relative or base.action_scale != 100 or base.dt != .01 or base.control_hz != 10 or
            base.action_space.shape != (2,) or not np.all(base.action_space.low == -1) or
            not np.all(base.action_space.high == 1)):
        env.close()
        raise ValueError("official physical environment contract differs from declaration")
    return env


@contextmanager
def _reset(factory, row, *, at_goal=False):
    env = factory()
    try:
        observation, info = env.reset(seed=row["episode"], options={
            "state": DIAGNOSTIC_GOAL.copy() if at_goal else row["states"][0].copy(),
            "goal_state": DIAGNOSTIC_GOAL.copy(),
        })
        yield env, observation, info
    finally:
        env.close()


def _goal_contract(env, info, row, factory):
    if (not np.array_equal(info.get("goal_state"), DIAGNOSTIC_GOAL) or
            not np.array_equal(info.get("goal_proprio"), DIAGNOSTIC_GOAL[[0, 1, 5, 6]])):
        raise ValueError("official goal metadata differs from fixed diagnostic reset goal")
    # The target marker is separate metadata in the official renderer. Record it
    # explicitly and reject a marker/state disagreement rather than hiding it.
    marker = np.asarray(info.get("goal_pose"))
    if not np.array_equal(marker, DIAGNOSTIC_GOAL[2:5]):
        raise ValueError("official goal_pose target marker disagrees with goal_state")
    root_frame = render_without_advancing(env)
    before = state_of(env)
    goal_image = np.asarray(info.get("goal"))
    if goal_image.shape != (224, 224, 3) or goal_image.dtype != np.uint8:
        raise ValueError("official reset did not supply uint8 goal RGB")
    # Separate environment: never set/reset the live root to render a goal.
    with _reset(factory, row, at_goal=True) as (goal_env, _, _):
        independently_rendered = render_without_advancing(goal_env)
    if not np.array_equal(goal_image, independently_rendered):
        raise ValueError("official reset goal RGB does not match independent official render")
    if not np.array_equal(before, state_of(env)):
        raise ValueError("goal rendering advanced the live root state")
    return {"goal_state": DIAGNOSTIC_GOAL.tolist(), "goal_pose": marker.tolist(),
            "goal_rgb_sha256": _array_hash(goal_image),
            "root_rgb_sha256": _array_hash(root_frame),
            "independent_goal_rgb_equal": True, "live_state_unchanged": True}


def _replay(row, factory, *, extra_div100=False, verify_goal=False):
    with _reset(factory, row) as (env, _, info):
        goal = _goal_contract(env, info, row, factory) if verify_goal else None
        history = BoundaryHistory(env, root_state=row["states"][0], goal=DIAGNOSTIC_GOAL)
        reset_error = float(np.max(np.abs(state_of(env) - row["states"][0])))
        commands = row["actions"] / 100 if extra_div100 else row["actions"]
        for packet in commands[:15].reshape(3, 5, 2):
            history.execute(packet)
        context = history.warmup_context()
        history.execute(commands[15:20])
        observed = update_observed_context(context, history)
        actual = np.stack(history.step_states)
        errors = np.abs(actual - row["states"][1:21])
        result = {
            "max_state_errors": errors.max(axis=1).tolist(),
            "agent_xy_l2_errors": np.linalg.norm(errors[:, :2], axis=1).tolist(),
            "reset_max_state_error": reset_error,
            "boundary_rgb_archive_mae": [float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
                                          for a, b in zip(history.frames, row["pixels"])],
            "archive_rgb_equivalence_claimed": False,
        }
        if not np.array_equal(history.packets.reshape(20, 2), commands):
            raise ValueError("runtime packet order differs from stored dense commands")
        steps = list(history.boundary_steps)
        if steps != list(BOUNDARIES) or observed.frame_steps != (10, 15, 20):
            raise ValueError("runtime boundary alignment failed")
        result["command_trace_sha256"] = _array_hash(history.packets.reshape(20, 2))
        trace = {"boundary_steps": steps, "warmup_frame_steps": list(context.frame_steps),
                 "observed_frame_steps": list(observed.frame_steps), "physical_step_count": history.step,
                 "root_id": history.root_id,
                 "observed_action_packet_sha256": _array_hash(observed.action_packets),
                 "boundary_rgb_sha256": [_array_hash(frame) for frame in history.frames],
                 "no_second_bootstrap": True}
        return result, trace, goal


def _repeat_equality(row, factory):
    packet = repeat_command(row["actions"][0])
    with _reset(factory, row) as (env, _, _):
        history = BoundaryHistory(env, root_state=row["states"][0], goal=DIAGNOSTIC_GOAL)
        history.execute(packet)
        packet_states, packet_frame = np.stack(history.step_states), history.frames[-1]
    # Independent reference: five raw calls, not execute_packet or action*5.
    with _reset(factory, row) as (env, _, _):
        direct = []
        for command in packet:
            _, _, terminated, truncated, _ = env.step(command.copy())
            if terminated or truncated:
                raise ValueError("repeated-command reference terminated within packet")
            direct.append(state_of(env))
        equal = (np.array_equal(packet_states, np.stack(direct)) and
                 np.array_equal(packet_frame, render_without_advancing(env)))
    return bool(equal)


def run_audit(data_path, output, indices=tuple(range(16)), *, env_factory=None, source_receipt=None):
    """Reserve a fresh directory; seal protocol before invoking ANY simulator.

    Existing directories are never reused/resumed. A failed report is also
    immutable. Injected factories are for synthetic tests, not physical evidence.
    """
    output, data_path = Path(output), Path(data_path)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema": VERSION, "contract": contract_spec(), "scope": "physical_contract_only",
        "selection": "predeclared_exact_episode_indices_no_outcome_replacement",
        "episode_indices": list(indices), "transition_steps": list(range(WINDOW_COMMANDS)),
        "fixed_diagnostic_goal": DIAGNOSTIC_GOAL.tolist(),
        "goal_source": "fixed_diagnostic_only_not_archived_training_goal",
        "validity": "all_21_stored_states_finite_nonterminal_all_20_commands_finite_in_bounds_dense_ids",
        "no_clipping_no_replacement": True,
        "first_transition_atol": FIRST_TRANSITION_ATOL,
        "later_replay_errors": "descriptive_only_seven_vector_omits_block_velocity",
        "normalization": PUBLIC_CONVENTION,
        "no_checkpoint_no_posttraining_no_neural_inference": True,
        "no_p0_p1_admission_or_causality": True,
        "simulator": "official" if env_factory is None else "injected_synthetic_not_physical_evidence",
    }
    report = {
        "schema": VERSION, "contract": contract_spec(), "scope": "physical_contract_only",
        "status": "failed", "checkpoint_loaded": False, "normalization": PUBLIC_CONVENTION,
        "gates": {key: False for key in REQUIRED_GATES}, "episodes": [],
        "limitations": ["No original checkpoint training-statistics proof.",
                        "No neural efficacy, P0/P1 admission, or causal finding.",
                        "No historical prediction/outcome files changed or admitted.",
                        "Selected rows only; no full archive hash or RGB equivalence claim."],
        "simulator": protocol["simulator"],
    }
    try:
        _snapshot_implementation(output)
        if source_receipt is not None:
            source_receipt = Path(source_receipt)
            protocol["source_existing_receipt"] = {**_source_stat(source_receipt),
                                                    "sha256": _file_hash(source_receipt),
                                                    "authority": "bound_reference_not_independently_verified"}
        rows, selected = select_rows(data_path, indices)
        protocol["data"] = selected
        _write_json_new(output / "protocol.json", protocol)
        factory = official_env_factory if env_factory is None else env_factory
        for row in rows:
            dense, history, goal = _replay(row, factory, verify_goal=True)
            divided, _, _ = _replay(row, factory, extra_div100=True)
            repeated = _repeat_equality(row, factory)
            report["episodes"].append({"episode": row["episode"], "dense": dense,
                                       "extra_div100": divided, "history": history, "goal": goal,
                                       "repeated_packet_equal": repeated})
        if selected["source"] != _source_stat(data_path):
            raise ValueError("source data changed during physical audit")
        report["gates"] = {
            "strict_packet_execution": True,
            "contiguous_boundary_history": True,
            "repeated_packet_equivalence": all(e["repeated_packet_equal"] for e in report["episodes"]),
            "goal_metadata_and_render_purity": True,
            "first_stored_transition_replay": all(e["dense"]["max_state_errors"][0] <= FIRST_TRANSITION_ATOL
                                                  for e in report["episodes"]),
        }
        # Exercise the shared guard without fabricating neural suffix evidence.
        identity = projection_identity(np.eye(2, dtype=np.float64))
        require_shared_suffix_projection(identity, identity)
        report["suffix_projection_guard"] = "available_only_no_neural_suffix_comparison_performed"
        report["status"] = "passed" if all(report["gates"].values()) else "failed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    if not (output / "protocol.json").exists():
        protocol["preflight_error"] = report.get("error", "unknown preflight failure")
        _write_json_new(output / "protocol.json", protocol)
    report["protocol_sha256"] = _file_hash(output / "protocol.json")
    implementation = output / "implementation.json"
    report["implementation_sha256"] = _file_hash(implementation) if implementation.exists() else None
    _write_json_new(output / "report.json", report)
    return report


def _indices(value):
    try:
        result = tuple(int(i) for i in value.split(","))
        _validate_indices(result)
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=DEFAULT_DATA)
    result.add_argument("--output", type=Path, required=True, help="NEW immutable output directory")
    result.add_argument("--episode-indices", type=_indices, default=tuple(range(16)),
                        help="exact predeclared IDs; default 0,1,...,15; invalid rows abort")
    result.add_argument("--source-receipt", type=Path, help="optional existing data provenance receipt to bind")
    return result


def main():
    args = parser().parse_args()
    report = run_audit(args.data, args.output, args.episode_indices, source_receipt=args.source_receipt)
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json"),
                      "error": report.get("error")}, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
