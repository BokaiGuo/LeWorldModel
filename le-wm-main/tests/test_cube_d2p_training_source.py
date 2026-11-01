import numpy as np
import pytest
import h5py

from experiments.spectral_reachability.cube_d2p_training_source import (
    ACTION_BLOCK,
    RAW_ACTION_DIM,
    _validate_complete_extent,
    _training_contract_hashes,
    _sha256,
    load_packet_pool,
    sample_contiguous_packet_sequences,
    sample_source_native_contexts,
    require_complete_hdf5,
    training_packet_statistics,
)


def test_cube_stats_are_raw_then_time_major_tiled() -> None:
    actions = np.asarray([
        [-1.0, 0.0, 1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],
    ], dtype=np.float32)
    mean, std = training_packet_statistics(actions)
    assert mean.shape == std.shape == (ACTION_BLOCK * RAW_ACTION_DIM,)
    np.testing.assert_allclose(mean.reshape(ACTION_BLOCK, RAW_ACTION_DIM), np.broadcast_to(mean[:RAW_ACTION_DIM], (ACTION_BLOCK, RAW_ACTION_DIM)))
    np.testing.assert_allclose(std.reshape(ACTION_BLOCK, RAW_ACTION_DIM), np.broadcast_to(std[:RAW_ACTION_DIM], (ACTION_BLOCK, RAW_ACTION_DIM)))


def test_cube_stats_reject_postpacket_or_wrong_width_actions() -> None:
    with pytest.raises(ValueError, match="shape"):
        training_packet_statistics(np.zeros((3, 25), dtype=np.float32))


def test_cube_packet_pool_never_crosses_episode_boundaries(tmp_path) -> None:
    source = tmp_path / "cube.h5"
    actions = np.arange(9 * RAW_ACTION_DIM, dtype=np.float32).reshape(9, RAW_ACTION_DIM)
    with h5py.File(source, "w") as handle:
        handle.create_dataset("action", data=actions)
        handle.create_dataset("ep_offset", data=np.asarray([0, 5], dtype=np.int64))
        handle.create_dataset("ep_len", data=np.asarray([5, 4], dtype=np.int64))
    pool, metadata = load_packet_pool(source, expected_sha256=_sha256(source))
    # The first episode contributes its one full packet; the second is too
    # short.  In particular, the packet must not splice rows 4 and 5.
    assert pool.shape == (1, ACTION_BLOCK, RAW_ACTION_DIM)
    np.testing.assert_array_equal(pool[0], actions[:ACTION_BLOCK])
    assert metadata["packet_count"] == 1


def test_cube_lazy_history_sampling_is_contiguous_and_episode_bound(tmp_path) -> None:
    source = tmp_path / "cube.h5"
    actions = np.arange(20 * RAW_ACTION_DIM, dtype=np.float32).reshape(20, RAW_ACTION_DIM)
    with h5py.File(source, "w") as handle:
        handle.create_dataset("action", data=actions)
        handle.create_dataset("ep_offset", data=np.asarray([0, 10], dtype=np.int64))
        handle.create_dataset("ep_len", data=np.asarray([10, 10], dtype=np.int64))
    values, provenance = sample_contiguous_packet_sequences(
        source, expected_sha256=_sha256(source), packets_per_sequence=2, count=2, seed=7,
    )
    assert values.shape == (2, 2, ACTION_BLOCK, RAW_ACTION_DIM)
    for sequence, receipt in zip(values, provenance):
        start = receipt["episode_raw_start"]
        np.testing.assert_array_equal(sequence.reshape(-1, RAW_ACTION_DIM), actions[start:start + 2 * ACTION_BLOCK])
        assert start // 10 == (start + 2 * ACTION_BLOCK - 1) // 10


def test_cube_lazy_history_sampling_excludes_nonfinite_nominal_windows(tmp_path) -> None:
    source = tmp_path / "cube_nan.h5"
    actions = np.arange(20 * RAW_ACTION_DIM, dtype=np.float32).reshape(20, RAW_ACTION_DIM)
    actions[2, 0] = np.nan
    with h5py.File(source, "w") as handle:
        handle.create_dataset("action", data=actions)
        handle.create_dataset("ep_offset", data=np.asarray([0, 10], dtype=np.int64))
        handle.create_dataset("ep_len", data=np.asarray([10, 10], dtype=np.int64))
    values, provenance = sample_contiguous_packet_sequences(
        source, expected_sha256=_sha256(source), packets_per_sequence=2, count=1, seed=3,
    )
    assert provenance[0]["episode_index"] == 1
    assert np.isfinite(values).all()


def _write_source_native_fixture(path, *, target_shift: bool = False) -> None:
    steps = 25
    with h5py.File(path, "w") as handle:
        handle.create_dataset("action", data=np.arange(steps * RAW_ACTION_DIM, dtype=np.float32).reshape(steps, RAW_ACTION_DIM))
        pixels = np.arange(steps * 2 * 3 * 3, dtype=np.uint8).reshape(steps, 2, 3, 3)
        handle.create_dataset("pixels", data=pixels)
        handle.create_dataset("qpos", data=np.ones((steps, 21), dtype=np.float64))
        handle.create_dataset("qvel", data=np.ones((steps, 20), dtype=np.float64))
        handle.create_dataset("prev_qpos", data=np.ones((steps, 21), dtype=np.float64))
        handle.create_dataset("prev_qvel", data=np.ones((steps, 20), dtype=np.float64))
        handle.create_dataset("control", data=np.ones((steps, 7), dtype=np.float64))
        handle.create_dataset("time", data=np.arange(steps, dtype=np.float64)[:, None])
        handle.create_dataset("privileged_target_block", data=np.zeros(steps, dtype=np.int64))
        positions = np.tile(np.asarray([[0.2, -0.1, 0.02]], dtype=np.float64), (steps, 1))
        if target_shift:
            positions[:, 0] += np.linspace(0.0, 0.01, steps)
        handle.create_dataset("privileged_target_block_pos", data=positions)
        handle.create_dataset("privileged_target_block_yaw", data=np.zeros((steps, 1), dtype=np.float64))
        handle.create_dataset("privileged_target_task", data=np.asarray([b"cube"] * steps))
        handle.create_dataset("ep_offset", data=np.asarray([0], dtype=np.int64))
        handle.create_dataset("ep_len", data=np.asarray([steps], dtype=np.int64))


def test_cube_source_native_context_uses_same_episode_rgb_actions_and_future_goal(tmp_path) -> None:
    source = tmp_path / "cube_source_native.h5"
    _write_source_native_fixture(source)
    contexts, provenance = sample_source_native_contexts(
        source, expected_sha256=_sha256(source), history_len=3, goal_offset_packets=1, count=1, seed=4,
    )
    assert len(contexts) == len(provenance) == 1
    context = contexts[0]
    rows = context["source_rows"]
    assert rows["root_row"] - rows["history_frame_rows"][0] == 3 * ACTION_BLOCK
    assert rows["goal_row"] - rows["root_row"] == ACTION_BLOCK
    assert context["history_raw_packets"].shape == (3, ACTION_BLOCK, RAW_ACTION_DIM)
    assert context["goal"].shape == context["root_frame"].shape == (2, 3, 3)
    assert context["evaluator_only_source_state"]["target_task"] == "cube"
    assert context["evaluator_only_source_state"]["time"] == [float(rows["root_row"])]


def test_cube_source_native_context_rejects_target_drift(tmp_path) -> None:
    source = tmp_path / "cube_target_drift.h5"
    _write_source_native_fixture(source, target_shift=True)
    with pytest.raises(ValueError, match="target provenance changes"):
        sample_source_native_contexts(
            source, expected_sha256=_sha256(source), history_len=3, goal_offset_packets=1, count=1, seed=4,
        )


def test_cube_hdf5_completion_gate_rejects_sparse_logical_placeholder() -> None:
    # A tar member can have its final logical length while its payload remains
    # holes.  The stored HDF5 EOF is the only relevant lower bound.
    with pytest.raises(ValueError, match="sparse"):
        _validate_complete_extent(
            logical_bytes=101_942_558_720,
            allocated_bytes=5_900_000_000,
            stored_eof=101_942_558_720,
        )


def test_cube_training_contract_has_all_prepacket_sources_bound() -> None:
    hashes = _training_contract_hashes()
    assert set(hashes) == {"train_entrypoint", "column_normalizer", "hdf5_dataset"}
    assert all(len(value) == 64 for value in hashes.values())
