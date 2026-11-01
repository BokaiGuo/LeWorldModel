import numpy as np
import pytest
import torch

from experiments.spectral_reachability.audit_reacher_d2p_source_render_crossover import crossover_metrics, load_native_clip, pixel_metrics


def test_constant_shared_appearance_shift_preserves_prediction_residual():
    ps, os = torch.tensor([1., 2.]), torch.tensor([2., 4.])
    shift = torch.tensor([.5, -.5])
    result = crossover_metrics(ps, ps + shift, os, os + shift, torch.zeros(2))
    assert result["prediction_residual_shift_l2"] == 0
    assert result["residual_shift_identity_max_error"] == 0
    cells = result["cells"]
    assert cells["source_input_source_observation"]["prediction_error_l2"] == cells["runtime_input_runtime_observation"]["prediction_error_l2"]


def test_input_and_observation_shifts_are_not_conflated():
    z = torch.zeros(2)
    result = crossover_metrics(z, torch.ones(2), z, z, z)
    assert result["input_induced_prediction_shift_l2"] == pytest.approx(2 ** .5)
    assert result["source_vs_runtime_observation_shift_l2"] == 0
    assert result["prediction_residual_shift_l2"] == pytest.approx(2 ** .5)


def dataset():
    return {"ep_len": np.array([30, 30]), "ep_offset": np.array([0, 30]),
        "pixels": np.zeros((60, 4, 4, 3), np.uint8), "action": np.arange(120).reshape(60, 2),
        "qpos": np.zeros((60, 2)), "qvel": np.zeros((60, 2))}


def test_native_clip_causal_packet_and_successor_rows():
    handle = dataset()
    clip = load_native_clip(handle, {"source_episode": 1, "source_start_row": 34})
    assert clip["frame_rows"] == [34, 39, 44, 49, 54]
    np.testing.assert_array_equal(clip["history"].reshape(15, 2), handle["action"][34:49])
    np.testing.assert_array_equal(clip["next_raw_packet"], handle["action"][49:54])


@pytest.mark.parametrize("episode,start", [(0, 10), (0, 29), (1, 29), (2, 60), (-1, 0)])
def test_episode_boundary_cannot_be_crossed_or_dropped(episode, start):
    with pytest.raises(ValueError):
        load_native_clip(dataset(), {"source_episode": episode, "source_start_row": start})


def test_uint8_pixel_difference_does_not_wrap():
    result = pixel_metrics(np.zeros((2, 3), np.uint8), np.full((2, 3), 255, np.uint8))
    assert result == {"mean_absolute_error": 255., "max_absolute_error": 255}
