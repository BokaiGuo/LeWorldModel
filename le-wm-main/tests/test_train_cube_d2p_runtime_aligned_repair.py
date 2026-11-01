import pytest
import torch
import h5py
import numpy as np
import hashlib
import json

from experiments.spectral_reachability.build_cube_d2p_runtime_repair_pool import (
    replace_candidate_with_uniform_packet,
)

from experiments.spectral_reachability.train_cube_d2p_runtime_aligned_repair import (
    HISTORY_SIZE,
    NUM_STEPS,
    _candidate_only_swap,
    _deployed_inputs,
    _terminal_from_deployed,
    _RuntimePoolClips,
    _validate_runtime_pool,
)


def test_runtime_aligned_cube_window_is_deployed_root_mapping() -> None:
    rgb = torch.zeros((2, NUM_STEPS, 3, 2, 2))
    action = torch.arange(2 * NUM_STEPS * 25, dtype=torch.float32).reshape(2, NUM_STEPS, 25)
    deployed_rgb, deployed_action = _deployed_inputs(rgb, action)
    assert deployed_rgb.shape[1] == HISTORY_SIZE
    assert deployed_rgb.is_contiguous()
    assert deployed_action.is_contiguous()
    torch.testing.assert_close(deployed_action, action[:, 1:4])
    # The deployed terminal is x4, which must not be included as input.
    assert NUM_STEPS == 5


def test_runtime_aligned_negative_swaps_only_candidate_token() -> None:
    action = torch.arange(3 * HISTORY_SIZE * 2, dtype=torch.float32).reshape(3, HISTORY_SIZE, 2)
    negative = _candidate_only_swap(action)
    torch.testing.assert_close(negative[:, :-1], action[:, :-1])
    torch.testing.assert_close(negative[:, -1], action.roll(1, dims=0)[:, -1])
    assert not torch.equal(negative[:, -1], action[:, -1])


def test_runtime_aligned_negative_requires_multiple_candidates() -> None:
    with pytest.raises(ValueError, match="batched three-token"):
        _candidate_only_swap(torch.zeros((1, HISTORY_SIZE, 25)))


def test_runtime_terminal_rejects_unwindowed_source_clip() -> None:
    class _Model:
        def rollout(self, *_args, **_kwargs):  # pragma: no cover - must not run
            raise AssertionError("unwindowed source clip reached model rollout")

    with pytest.raises(ValueError, match="deployed three-token"):
        _terminal_from_deployed(_Model(), torch.zeros((2, NUM_STEPS, 3, 2, 2)), torch.zeros((2, NUM_STEPS, 25)))


def test_runtime_pool_reader_preserves_five_frame_and_packet_layout(tmp_path) -> None:
    path = tmp_path / "pool.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pixels", data=torch.zeros((32, NUM_STEPS, 2, 3, 3), dtype=torch.uint8).numpy())
        handle.create_dataset("action", data=torch.arange(32 * NUM_STEPS * 25, dtype=torch.float32).reshape(32, NUM_STEPS, 25).numpy())
        handle.create_dataset("root_seed", data=list(range(32)))
    pool = _RuntimePoolClips(path)
    item = pool[1]
    assert item["pixels"].shape == (NUM_STEPS, 3, 2, 3)
    assert item["pixels"].dtype == torch.uint8
    torch.testing.assert_close(item["action"], torch.arange(32 * NUM_STEPS * 25, dtype=torch.float32).reshape(32, NUM_STEPS, 25)[1])


def test_uniform_candidate_repair_pool_preserves_history_and_replaces_only_a3() -> None:
    packets = np.arange(NUM_STEPS * 5 * 5, dtype=np.float32).reshape(NUM_STEPS, 5, 5)
    uniform = replace_candidate_with_uniform_packet(packets, generator=np.random.default_rng(9))
    np.testing.assert_array_equal(uniform[:3], packets[:3])
    np.testing.assert_array_equal(uniform[4], packets[4])
    assert not np.array_equal(uniform[3], packets[3])
    assert np.all(uniform[3] >= -1.0) and np.all(uniform[3] <= 1.0)


def test_runtime_pool_validator_accepts_bound_v2_uniform_receipt(tmp_path) -> None:
    source_receipt = tmp_path / "source.json"
    source_receipt.write_text("source receipt\n", encoding="utf-8")
    pool = tmp_path / "uniform.h5"
    with h5py.File(pool, "w") as handle:
        handle.attrs["schema"] = "cube_d2p_live_runtime_repair_pool.v2"
        handle.attrs["source_receipt_sha256"] = hashlib.sha256(source_receipt.read_bytes()).hexdigest()
        handle.attrs["candidate_distribution"] = "uniform"
    receipt = tmp_path / "uniform.h5.receipt.json"
    receipt.write_text(json.dumps({
        "schema": "cube_d2p_live_runtime_repair_pool.v2",
        "status": "pass",
        "source": {"receipt_sha256": hashlib.sha256(source_receipt.read_bytes()).hexdigest()},
        "pool": {
            "path": str(pool.resolve()),
            "sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
            "candidate_distribution": "uniform",
        },
    }), encoding="utf-8")
    validated = _validate_runtime_pool(pool, receipt, source_receipt)
    assert validated["pool"]["candidate_distribution"] == "uniform"
