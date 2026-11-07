import numpy as np
import pytest

from experiments.pc_tworoom_spectral_nominal_clip_audit_20261107_codex import action_pair


def test_original_clipped_binding_and_legality():
    raw = np.tile(np.asarray([1.0, -0.8], np.float32), (25, 1))
    clipped = np.clip(raw, -.7, .7)
    pair = action_pair(raw, clipped)
    np.testing.assert_array_equal(pair[0], raw)
    np.testing.assert_array_equal(pair[1], clipped)
    with pytest.raises(ValueError, match="match clipping"):
        action_pair(raw, raw)
    with pytest.raises(ValueError, match="invalid original"):
        action_pair(raw * 2, clipped)
