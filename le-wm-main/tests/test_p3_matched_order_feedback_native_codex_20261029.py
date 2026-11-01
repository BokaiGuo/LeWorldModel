"""Adapter fixtures only; no actual physics, neural model, or GPU."""
import numpy as np
import pytest
from experiments.spectral_reachability import p3_matched_order_feedback_native_codex_20261029 as native
from experiments.spectral_reachability import p3_matched_order_feedback_store_codex_20261029 as store


def test_reject_wrong_root_before_environment_access():
    with pytest.raises(ValueError,match='same root task'):
        native.setup(None,{'root_sha256':'a'},None,{'binding':{'root_sha256':'b'}},None,None,None)


def test_four_slot_capture_goes_beyond_old_32_frame_limit(tmp_path):
    s = store.Store(tmp_path/'run'); bridge = native.CaptureStore(s)
    image = np.zeros((224,224,3),np.uint8)
    for i in range(33):
        bridge.frame(f'slot_frame_{i:03}.png',image)
    assert s.frames == 33 and len(bridge.frames) == 33
    with pytest.raises(ValueError,match='unique'):
        bridge.frame('slot_frame_000.png',image)
