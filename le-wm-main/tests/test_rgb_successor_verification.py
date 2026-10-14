import numpy as np

from experiments.decision_benchmark.verify_rgb_successor import (
    _render_dataset_matched,
    _static_scene_and_marker,
    _temporal_contract,
)


def test_dataset_matched_renderer_reconstructs_source_marker():
    base = np.zeros((224, 224, 3), dtype=np.uint8)
    base[...] = np.asarray([25, 51, 76], dtype=np.uint8)
    source = base.copy()
    state = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    center = np.rint(np.asarray([27.0, 27.77]) * state[:2]).astype(int)
    source[center[1] - 4 : center[1] + 5, center[0] - 4 : center[0] + 5] = np.asarray([0, 180, 0], dtype=np.uint8)
    images = np.stack([source, base, base, base], axis=0)
    recovered_base, residual, offset = _static_scene_and_marker(images, source, state)
    reconstructed = _render_dataset_matched(recovered_base, residual, state, offset)
    assert np.array_equal(reconstructed, source)


def test_renderer_keeps_scene_outside_moving_marker_unchanged():
    base = np.zeros((224, 224, 3), dtype=np.uint8)
    base[...] = np.asarray([51, 76, 102], dtype=np.uint8)
    source = base.copy()
    state = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    center = np.rint(np.asarray([27.0, 27.77]) * state[:2]).astype(int)
    source[center[1] - 4 : center[1] + 5, center[0] - 4 : center[0] + 5] = np.asarray([0, 180, 0], dtype=np.uint8)
    recovered_base, residual, offset = _static_scene_and_marker(np.stack([source, base, base]), source, state)
    moved = _render_dataset_matched(recovered_base, residual, np.asarray([2.0, 2.0, 0.0, 0.0]), offset)
    mask = np.zeros(base.shape[:2], dtype=bool)
    mask[center[1] - 12 : center[1] + 13, center[0] - 12 : center[0] + 13] = True
    moved_center = np.rint(np.asarray([27.0, 27.77]) * np.asarray([2.0, 2.0])).astype(int)
    mask[moved_center[1] - 12 : moved_center[1] + 13, moved_center[0] - 12 : moved_center[0] + 13] = True
    assert np.array_equal(moved[~mask], base[~mask])


def test_temporal_contract_preserves_legacy_two_frame_bank():
    indices, length = _temporal_contract({"history_windows": 1, "context_length": 2}, 5)
    assert indices == [0, 5]
    assert length == 2


def test_temporal_contract_exposes_full_causal_history_four():
    indices, length = _temporal_contract({"history_windows": 4, "context_length": 4}, 20)
    assert indices == [5, 10, 15, 20]
    assert length == 4


def test_temporal_contract_fails_closed_when_history_is_unavailable():
    try:
        _temporal_contract({"history_windows": 4}, 10)
    except ValueError as error:
        assert "insufficient causal RGB context" in str(error)
    else:
        raise AssertionError("missing causal context must fail closed")
