import base64
import hashlib
import zlib

import numpy as np
import pytest

from experiments.spectral_reachability.run_reacher_pc_fd_saved_rgb_replay_20261102 import advance_saved, decode_rgb


def test_rgb_digest_and_history_are_bound():
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    payload = rgb.tobytes()
    saved = {"shape": list(rgb.shape), "sha256": hashlib.sha256(payload).hexdigest(),
             "uint8_zlib_base64": base64.b64encode(zlib.compress(payload)).decode()}
    decoded = decode_rgb(saved)
    old = [np.full_like(rgb, x) for x in (1, 2, 3)]
    blocks = [np.full(10, x, dtype=np.float32) for x in (1, 2, 3)]
    context = {"frames": old, "root_frame": np.full_like(rgb, 4), "goal": rgb.copy(),
               "history_action_blocks": blocks}
    action = np.arange(10, dtype=np.float32).reshape(5, 2)
    updated = advance_saved(context, action, decoded)
    assert [int(x[0, 0, 0]) for x in updated["frames"]] == [2, 3, 4]
    np.testing.assert_array_equal(updated["history_action_blocks"][-1], action.ravel())
    np.testing.assert_array_equal(updated["root_frame"], rgb)
    updated["root_frame"][0, 0, 0] = 99
    assert decoded[0, 0, 0] == 0
    saved["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        decode_rgb(saved)


def test_invalid_history_rejected():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    context = {"frames": [image] * 3, "root_frame": image, "goal": image,
               "history_action_blocks": [np.zeros(10)] * 3}
    with pytest.raises(ValueError):
        advance_saved(context, np.zeros((4, 2)), image)
