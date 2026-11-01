"""Source-locked training-action receipts for frozen Cube D2P.

The released Cube checkpoint advertises ``quentinll/lewm-cube`` as its
training source and has a 25-dimensional action encoder.  This is not enough
to let a runtime silently average a post-packet action statistic: the trainer
normalises a physical five-dimensional action before packing five consecutive
actions into each model token.  This module makes that assumption executable.

It serves two deliberately narrow purposes:

* derive a SHA-bound 25-wide normalisation receipt from the official HDF5;
* construct finite, within-episode, time-major five-action packets for
  source-support checks.

Neither operation evaluates a model or supplies a planning outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


SCHEMA = "cube_d2p_training_source.v1"
ENVIRONMENT = "swm/OGBCube-v0"
RAW_ACTION_DIM = 5
ACTION_BLOCK = 5
SOURCE_NATIVE_COLUMNS = (
    "pixels",
    "action",
    "qpos",
    "qvel",
    "prev_qpos",
    "prev_qvel",
    "control",
    "time",
    "privileged_target_block",
    "privileged_target_block_pos",
    "privileged_target_block_yaw",
    "privileged_target_task",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_contract_hashes() -> dict[str, str]:
    """Bind the source receipt to the exact pre-packetisation trainer path."""

    import stable_worldmodel.data.formats.hdf5 as hdf5_format

    project = Path(__file__).resolve().parents[2]
    files = {
        "train_entrypoint": project / "train.py",
        "column_normalizer": project / "utils.py",
        "hdf5_dataset": Path(hdf5_format.__file__).resolve(),
    }
    if not all(path.is_file() for path in files.values()):
        raise RuntimeError("Cube training-contract source file is missing")
    return {name: _sha256(path) for name, path in files.items()}


def _validate_complete_extent(*, logical_bytes: int, allocated_bytes: int, stored_eof: int) -> None:
    if logical_bytes < stored_eof:
        raise ValueError("Cube HDF5 logical size is shorter than its stored EOF")
    if allocated_bytes < stored_eof:
        raise ValueError(
            "Cube HDF5 is a partially written sparse file: "
            f"allocated_bytes={allocated_bytes} stored_eof={stored_eof}"
        )


def require_complete_hdf5(path: Path) -> int:
    """Reject tar-created sparse placeholders that merely *look* readable.

    Streaming ``tar`` may apply the member's final logical size before its
    payload has arrived.  HDF5 can then open the sparse prefix and report its
    stored EOF even though most dataset chunks are holes.  A source receipt
    made from that file would turn an interrupted download into a scientific
    input.  The official Cube export is a regular, non-sparse HDF5, so its
    allocated bytes must cover its HDF5 EOF before it is admitted.
    """

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cube source HDF5 not found: {path}")
    stat = path.stat()
    allocated = int(stat.st_blocks) * 512
    with h5py.File(path, "r", swmr=True) as handle:
        stored_eof = int(handle.id.get_filesize())
    _validate_complete_extent(
        logical_bytes=int(stat.st_size), allocated_bytes=allocated, stored_eof=stored_eof,
    )
    return stored_eof


def _episode_columns(handle: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """Return validated episode offsets and lengths without guessing layout."""

    if "ep_offset" not in handle or "ep_len" not in handle:
        raise ValueError("Cube source HDF5 must expose ep_offset and ep_len")
    offsets = np.asarray(handle["ep_offset"], dtype=np.int64)
    lengths = np.asarray(handle["ep_len"], dtype=np.int64)
    if offsets.ndim != 1 or lengths.shape != offsets.shape or len(offsets) == 0:
        raise ValueError("Cube episode columns have incompatible shapes")
    if np.any(offsets < 0) or np.any(lengths < 1):
        raise ValueError("Cube episode columns contain invalid values")
    if np.any(offsets[1:] < offsets[:-1] + lengths[:-1]):
        raise ValueError("Cube episode ranges overlap or are unordered")
    return offsets, lengths


def training_packet_statistics(actions: np.ndarray, *, action_block: int = ACTION_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce pre-packet raw-action z-score statistics, tiled time-major."""

    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != RAW_ACTION_DIM or action_block != ACTION_BLOCK:
        raise ValueError("expected Cube actions with shape [rows, 5] and action_block=5")
    finite = values[np.isfinite(values).all(axis=1)]
    if finite.shape[0] < 2:
        raise ValueError("Cube source has fewer than two finite physical actions")
    physical = torch.from_numpy(np.ascontiguousarray(finite, dtype=np.float32))
    mean, std = physical.mean(dim=0), physical.std(dim=0)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0.0):
        raise ValueError("Cube training action statistics are non-finite or degenerate")
    return mean.repeat(action_block).numpy(), std.repeat(action_block).numpy()


def load_packet_pool(
    dataset: Path, *, expected_sha256: str, action_block: int = ACTION_BLOCK,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load all finite, consecutive action packets contained in one episode."""

    dataset = dataset.resolve()
    if action_block != ACTION_BLOCK or not dataset.is_file() or _sha256(dataset) != expected_sha256:
        raise ValueError("Cube packet source does not match its SHA-bound receipt")
    chunks: list[np.ndarray] = []
    with h5py.File(dataset, "r", swmr=True) as handle:
        if "action" not in handle:
            raise ValueError("Cube source HDF5 has no action column")
        actions = handle["action"]
        if actions.ndim != 2 or actions.shape[1] != RAW_ACTION_DIM:
            raise ValueError("Cube action column is not five-dimensional")
        offsets, lengths = _episode_columns(handle)
        if int(offsets[-1] + lengths[-1]) > int(actions.shape[0]):
            raise ValueError("Cube episode ranges exceed action rows")
        for offset, length in zip(offsets.tolist(), lengths.tolist()):
            if length < action_block:
                continue
            episode = np.asarray(actions[offset: offset + length], dtype=np.float32)
            windows = np.lib.stride_tricks.sliding_window_view(episode, window_shape=action_block, axis=0)
            windows = np.moveaxis(windows, -1, 1)
            valid = windows[np.isfinite(windows).all(axis=(1, 2))]
            if len(valid):
                chunks.append(np.ascontiguousarray(valid))
    if not chunks:
        raise ValueError("Cube source contains no finite full action packet")
    pool = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
    expected_shape = (action_block, RAW_ACTION_DIM)
    if pool.ndim != 3 or tuple(pool.shape[1:]) != expected_shape:
        raise RuntimeError("Cube packet pool lost its time-major action layout")
    return pool, {
        "dataset": str(dataset),
        "dataset_sha256": expected_sha256,
        "packet_count": int(len(pool)),
        "packet_layout": "time-major consecutive raw actions within one source episode",
        "pool_sha256": hashlib.sha256(pool.tobytes()).hexdigest(),
    }


def sample_contiguous_packet_sequences(
    dataset: Path,
    *,
    expected_sha256: str,
    packets_per_sequence: int,
    count: int,
    seed: int,
    action_block: int = ACTION_BLOCK,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Sample source packets without materialising the entire 68-GB HDF5.

    Each returned sequence is one contiguous interval inside a single source
    episode.  Selecting independent valid five-step packets would only match
    an action marginal, while a LeWM history token sequence is trained on a
    causal packet trajectory.  The flattened candidate index is mapped to an
    episode/window pair by cumulative *finite* valid-start counts.  The
    archive has NaNs at some episode boundaries, so nominal episode length is
    not a sampling capacity.  We retain only valid start indices (a few MB for
    this source) rather than materialising the image archive.
    """

    dataset = Path(dataset).resolve()
    if int(packets_per_sequence) < 1 or int(count) < 1 or int(action_block) != ACTION_BLOCK:
        raise ValueError("Cube contiguous packet sampling dimensions are invalid")
    if not dataset.is_file() or _sha256(dataset) != str(expected_sha256):
        raise ValueError("Cube contiguous packet source does not match its SHA-bound receipt")
    needed = int(packets_per_sequence) * int(action_block)
    with h5py.File(dataset, "r", swmr=True) as handle:
        if "action" not in handle:
            raise ValueError("Cube source HDF5 has no action column")
        actions = handle["action"]
        if actions.ndim != 2 or actions.shape[1] != RAW_ACTION_DIM:
            raise ValueError("Cube action column is not five-dimensional")
        offsets, lengths = _episode_columns(handle)
        starts_by_episode: list[np.ndarray] = []
        for offset, length in zip(offsets.tolist(), lengths.tolist()):
            nominal = max(int(length) - needed + 1, 0)
            if nominal == 0:
                starts_by_episode.append(np.empty(0, dtype=np.int64))
                continue
            episode_actions = np.asarray(actions[int(offset):int(offset) + int(length)], dtype=np.float32)
            finite_step = np.isfinite(episode_actions).all(axis=1).astype(np.int16)
            complete = np.convolve(finite_step, np.ones(needed, dtype=np.int16), mode="valid") == needed
            starts_by_episode.append(np.flatnonzero(complete).astype(np.int64, copy=False))
        capacities = np.asarray([len(starts) for starts in starts_by_episode], dtype=np.int64)
        total = int(capacities.sum())
        if total < int(count):
            raise ValueError("Cube source has too few distinct contiguous packet sequences")
        selected = np.sort(np.random.default_rng(np.random.SeedSequence([int(seed), 0xC0BE])).choice(total, size=int(count), replace=False))
        cumulative = np.cumsum(capacities)
        values: list[np.ndarray] = []
        provenance: list[dict[str, int]] = []
        for flat in selected.tolist():
            episode = int(np.searchsorted(cumulative, flat, side="right"))
            prior = int(cumulative[episode - 1]) if episode else 0
            start = int(offsets[episode]) + int(starts_by_episode[episode][int(flat) - prior])
            raw = np.asarray(actions[start:start + needed], dtype=np.float32)
            if raw.shape != (needed, RAW_ACTION_DIM) or not np.isfinite(raw).all():
                raise ValueError("Cube source selected a non-finite or truncated contiguous action sequence")
            values.append(raw.reshape(int(packets_per_sequence), ACTION_BLOCK, RAW_ACTION_DIM))
            provenance.append({"episode_index": episode, "episode_raw_start": start, "packet_count": int(packets_per_sequence)})
    return np.ascontiguousarray(np.stack(values)), provenance


def sample_source_native_contexts(
    dataset: Path,
    *,
    expected_sha256: str,
    history_len: int,
    goal_offset_packets: int,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, int]]]:
    """Return exact model contexts from one source episode, never a reset replay.

    Cube training uses five raw actions for each observation stride and samples
    goal RGB from an episode future.  A live reset plus source actions only
    matches an action marginal; it does *not* recreate the RGB prefix or the
    goal distribution which the frozen checkpoint saw.  This reader preserves
    all three pieces from one HDF5 episode:

    ``[x_t, x_t+5, x_t+10]``, ``[a_t:t+5, ...]``, and a future source goal.

    Privileged fields are returned solely as evaluator provenance.  They are
    intentionally not inputs to LeWM or its goal image.
    """

    dataset = Path(dataset).resolve()
    if min(int(history_len), int(goal_offset_packets), int(count)) < 1:
        raise ValueError("source-native Cube history, goal offset, and count must be positive")
    if not dataset.is_file() or _sha256(dataset) != str(expected_sha256):
        raise ValueError("Cube source-native contexts require the SHA-bound training dataset")
    needed_actions = int(history_len) * ACTION_BLOCK
    # The observed root is after all history packets.  The goal must be a
    # strictly future source image, matching GoalDataset's future branches.
    needed_rows = needed_actions + int(goal_offset_packets) * ACTION_BLOCK + 1
    with h5py.File(dataset, "r", swmr=True) as handle:
        missing = [column for column in SOURCE_NATIVE_COLUMNS if column not in handle]
        if missing:
            raise ValueError(f"Cube source lacks required source-native columns: {missing}")
        actions = handle["action"]
        pixels = handle["pixels"]
        if actions.ndim != 2 or actions.shape[1] != RAW_ACTION_DIM:
            raise ValueError("Cube source action column is not five-dimensional")
        if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
            raise ValueError("Cube source pixels must be uint8 [time,height,width,3]")
        offsets, lengths = _episode_columns(handle)
        capacities = np.maximum(lengths - needed_rows + 1, 0).astype(np.int64)
        total = int(capacities.sum())
        if total < int(count):
            raise ValueError("Cube source has too few complete source-native contexts")
        selected = np.sort(
            np.random.default_rng(np.random.SeedSequence([int(seed), 0xC0BE, 0x51A7])).choice(
                total, size=int(count), replace=False,
            )
        )
        cumulative = np.cumsum(capacities)
        contexts: list[dict[str, Any]] = []
        provenance: list[dict[str, int]] = []
        for flat in selected.tolist():
            episode = int(np.searchsorted(cumulative, flat, side="right"))
            prior = int(cumulative[episode - 1]) if episode else 0
            start = int(offsets[episode]) + (int(flat) - prior)
            root_row = start + needed_actions
            goal_row = root_row + int(goal_offset_packets) * ACTION_BLOCK
            raw_actions = np.asarray(actions[start:root_row], dtype=np.float32)
            if raw_actions.shape != (needed_actions, RAW_ACTION_DIM) or not np.isfinite(raw_actions).all():
                raise ValueError("Cube source-native context has non-finite or truncated actions")
            frame_rows = [start + packet * ACTION_BLOCK for packet in range(int(history_len))]
            frames = np.asarray(pixels[frame_rows], dtype=np.uint8)
            root = np.asarray(pixels[root_row], dtype=np.uint8)
            goal = np.asarray(pixels[goal_row], dtype=np.uint8)
            if frames.shape[0] != int(history_len) or root.shape != goal.shape or root.ndim != 3:
                raise ValueError("Cube source-native RGB rows have incompatible shapes")
            target_rows = slice(start, goal_row + 1)
            target_block = np.asarray(handle["privileged_target_block"][target_rows])
            target_pos = np.asarray(handle["privileged_target_block_pos"][target_rows], dtype=np.float64)
            target_yaw = np.asarray(handle["privileged_target_block_yaw"][target_rows], dtype=np.float64)
            target_task = np.asarray(handle["privileged_target_task"][target_rows])
            if (
                target_block.ndim < 1
                or target_pos.ndim < 2
                or target_yaw.ndim < 2
                or target_task.ndim < 1
                or not np.array_equal(target_block, np.broadcast_to(target_block[0], target_block.shape))
                or not np.allclose(target_pos, target_pos[0], rtol=0.0, atol=1.0e-8)
                or not np.allclose(target_yaw, target_yaw[0], rtol=0.0, atol=1.0e-8)
                or not np.array_equal(target_task, np.broadcast_to(target_task[0], target_task.shape))
            ):
                raise ValueError("Cube source-native target provenance changes inside one context")
            qpos = np.asarray(handle["qpos"][root_row], dtype=np.float64)
            qvel = np.asarray(handle["qvel"][root_row], dtype=np.float64)
            prev_qpos = np.asarray(handle["prev_qpos"][root_row], dtype=np.float64)
            prev_qvel = np.asarray(handle["prev_qvel"][root_row], dtype=np.float64)
            control = np.asarray(handle["control"][root_row], dtype=np.float64)
            time_value = np.asarray(handle["time"][root_row], dtype=np.float64)
            state_values = (qpos, qvel, prev_qpos, prev_qvel, control, time_value)
            if any(value.ndim != 1 or not np.isfinite(value).all() for value in state_values):
                raise ValueError("Cube source-native full simulator snapshot is invalid")
            contexts.append({
                "frames": [frame.copy() for frame in frames],
                "root_frame": root.copy(),
                "goal": goal.copy(),
                "history_raw_packets": raw_actions.reshape(int(history_len), ACTION_BLOCK, RAW_ACTION_DIM),
                "source_rows": {
                    "history_frame_rows": frame_rows,
                    "root_row": root_row,
                    "goal_row": goal_row,
                    "goal_offset_packets": int(goal_offset_packets),
                },
                "evaluator_only_source_state": {
                    "qpos": qpos.tolist(),
                    "qvel": qvel.tolist(),
                    "prev_qpos": prev_qpos.tolist(),
                    "prev_qvel": prev_qvel.tolist(),
                    "control": control.tolist(),
                    "time": time_value.tolist(),
                    "target_block": np.asarray(target_block[0]).tolist(),
                    "target_pos": target_pos[0].tolist(),
                    "target_yaw": target_yaw[0].tolist(),
                    "target_task": target_task[0].decode() if isinstance(target_task[0], bytes) else str(target_task[0]),
                },
            })
            provenance.append({"episode_index": episode, "episode_raw_start": start, "root_row": root_row, "goal_row": goal_row})
    return contexts, provenance


def build(*, dataset: Path, output: Path) -> dict[str, Any]:
    """Write an immutable training-semantics receipt for the official HDF5."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable Cube receipt: {output}")
    if not dataset.is_file():
        raise FileNotFoundError(f"Cube source HDF5 not found: {dataset}")
    dataset = dataset.resolve()
    require_complete_hdf5(dataset)
    source_sha256 = _sha256(dataset)
    with h5py.File(dataset, "r", swmr=True) as handle:
        if "action" not in handle:
            raise ValueError("Cube source HDF5 has no action column")
        actions = handle["action"]
        if actions.ndim != 2 or actions.shape[1] != RAW_ACTION_DIM:
            raise ValueError("Cube source action column must have width five")
        offsets, lengths = _episode_columns(handle)
        if int(offsets[-1] + lengths[-1]) > int(actions.shape[0]):
            raise ValueError("Cube episode ranges exceed action rows")
        mean, std = training_packet_statistics(actions[:])
        action_rows = int(actions.shape[0])
    payload = {
        "schema": SCHEMA,
        "status": "training_semantics_validated",
        "source": {"dataset": str(dataset), "dataset_sha256": source_sha256, "dataset_bytes": dataset.stat().st_size},
        "training_contract": {
            "normalization_order": "get_column_normalizer(action) before HDF5Dataset packs frameskip actions",
            "source_sha256": _training_contract_hashes(),
        },
        "report": {
            "environment": ENVIRONMENT,
            "raw_action_dim": RAW_ACTION_DIM,
            "action_block": ACTION_BLOCK,
            "action_rows": action_rows,
            "episode_count": int(len(offsets)),
            "calibration_action_mean": mean.tolist(),
            "calibration_action_std": std.tolist(),
            "normalization_contract": "raw five-dimensional actions are z-scored before five-step time-major packetisation",
            "all_packet_slots_share_physical_statistics": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_receipt(path: Path, *, verify_local_training_contract: bool = True) -> dict[str, Any]:
    """Fail closed on any receipt not matching the checkpoint action token."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload.get("report", {})
    if payload.get("schema") != SCHEMA or payload.get("status") != "training_semantics_validated":
        raise ValueError("Cube D2P requires a validated training-source receipt")
    contract = payload.get("training_contract", {})
    if contract.get("normalization_order") != "get_column_normalizer(action) before HDF5Dataset packs frameskip actions":
        raise ValueError("Cube source receipt does not bind the trainer's pre-packet normalization order")
    if verify_local_training_contract and contract.get("source_sha256") != _training_contract_hashes():
        raise ValueError("Cube source receipt was made under a different training-contract implementation")
    if row.get("environment") != ENVIRONMENT or row.get("raw_action_dim") != RAW_ACTION_DIM or row.get("action_block") != ACTION_BLOCK:
        raise ValueError("Cube source receipt has an incompatible action contract")
    mean = np.asarray(row.get("calibration_action_mean"), dtype=np.float32)
    std = np.asarray(row.get("calibration_action_std"), dtype=np.float32)
    if mean.shape != (RAW_ACTION_DIM * ACTION_BLOCK,) or std.shape != mean.shape or not np.isfinite(mean).all() or np.any(std <= 0.0):
        raise ValueError("Cube source receipt has invalid action statistics")
    if not np.array_equal(mean.reshape(ACTION_BLOCK, RAW_ACTION_DIM), np.broadcast_to(mean[:RAW_ACTION_DIM], (ACTION_BLOCK, RAW_ACTION_DIM))):
        raise ValueError("Cube source receipt has position-specific action means")
    if not np.array_equal(std.reshape(ACTION_BLOCK, RAW_ACTION_DIM), np.broadcast_to(std[:RAW_ACTION_DIM], (ACTION_BLOCK, RAW_ACTION_DIM))):
        raise ValueError("Cube source receipt has position-specific action standard deviations")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(dataset=args.dataset, output=args.output)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
