from io import BytesIO
import zipfile

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import p1_queue_goal_blind_adapter_20261101_codex as a


def history(tmp_path):
    frames = []
    for i in (0, 5, 10):
        pixels = np.full((224, 224, 3), i, np.uint8)
        path = tmp_path / f"{i}.png"
        Image.fromarray(pixels).save(path)
        frames.append(dict(step=i, file=path.name, file_sha256=a.blind.sha(path),
                           pixel_sha256=a.blind.digest(pixels.tobytes())))
    goal = dict(frames[-1])
    return dict(frames=frames, goal_rgb=goal, actions=[[0., 0.]] * 10)


def test_packet_real_frames_and_history(tmp_path):
    h = history(tmp_path)
    result = a.packet(h, tmp_path)
    assert result["rgb_history"].shape == (3, 224, 224, 3)
    assert result["rgb_history"][:, 0, 0, 0].tolist() == [0, 5, 10]
    assert result["raw_action_history"].dtype == np.float32
    assert np.array_equal(result["goal_rgb"], result["rgb_history"][-1])


def test_tamper_and_wrong_history_rejected(tmp_path):
    h = history(tmp_path)
    h["actions"][0] = [1., 0.]
    with pytest.raises(ValueError):
        a.packet(h, tmp_path)
    h["actions"][0] = [0., 0.]
    h["frames"][0]["pixel_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        a.packet(h, tmp_path)


def test_original_loader_rejects_privileged_member(tmp_path):
    path = tmp_path / "extra_blind_inputs.npz"
    np.savez_compressed(path, rgb_history=np.zeros((3, 224, 224, 3), np.uint8),
                        goal_rgb=np.zeros((224, 224, 3), np.uint8),
                        raw_action_history=np.zeros((10, 2), np.float32), physical_state=np.zeros(7))
    with pytest.raises(ValueError):
        a.blind.load_blind(path, a.blind.sha(path))


def test_original_task_export_normalizes_actual_raw_history(tmp_path):
    h = history(tmp_path)
    p = a.packet(h, tmp_path)
    norm = a.blind.native_metadata()
    slots = dict(order=0, source_slot=0, blind_sha256="a" * 64,
                 pixel_sha256=[a.blind.digest(x.tobytes()) for x in
                               [*p["rgb_history"], p["goal_rgb"]]],
                 raw_action_sha256=a.blind.digest(p["raw_action_history"].tobytes()))
    prep = dict(normalization=norm, arguments=dict(whitelist_sha="b" * 64),
                model_bindings=a.blind.MODELS)
    values = [np.full(192, i, np.float32) for i in range(4)]
    export = a.blind.task_export(prep, "c" * 64, slots, p, values)
    tokens = np.asarray(export["task"]["root_context"]["action_history"])
    expected = ((p["raw_action_history"].reshape(2, 10)
                 - np.tile(np.asarray(norm["mean"], np.float32), 5))
                / np.tile(np.asarray(norm["std"], np.float32), 5))
    np.testing.assert_array_equal(tokens, expected)
    assert export["task"]["root_raw_action_history"] == h["actions"]


def test_real_saved_source_and_original_loader(tmp_path):
    path = tmp_path / "dev_blind_inputs.npz"
    result = a.build(path)
    assert result["seed"] == 301000001
    assert result["independent_confirmation"] is False
    assert len(set(result["pixel_sha256"][:3])) == 1
    with zipfile.ZipFile(BytesIO(path.read_bytes())) as z:
        assert set(z.namelist()) == {"rgb_history.npy", "raw_action_history.npy", "goal_rgb.npy"}
    with pytest.raises(ValueError):
        a.build(path)
