import numpy as np
import pytest

from experiments.pc_tworoom_history_refinement_20261107_codex import validate_initial


def test_action_packets_and_bounds():
    blocks = np.array([[0.0, 0.2], [0.3, -0.2], [-0.5, 0.6], [1, -1], [0, 0]], dtype=np.float32)
    raw = np.repeat(blocks, 5, axis=0)
    np.testing.assert_array_equal(validate_initial(raw), blocks)
    bad = raw.copy()
    bad[1, 0] = 0.1
    with pytest.raises(ValueError):
        validate_initial(bad)
    bad = raw.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_initial(bad)
    with pytest.raises(ValueError):
        validate_initial(np.full((25, 2), 1.1, dtype=np.float32))
    with pytest.raises(ValueError):
        validate_initial(raw[:-1])
