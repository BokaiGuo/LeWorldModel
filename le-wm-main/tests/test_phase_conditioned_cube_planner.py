import numpy as np

from experiments.run_live_cube_action_adapter_planning import (
    ActionNormalizer,
    _cube_phase_bank,
)


def test_cube_phase_bank_has_frozen_contact_sequence_and_valid_shape():
    normalizer = ActionNormalizer(np.zeros(25, dtype=np.float32), np.ones(25, dtype=np.float32))
    bank = _cube_phase_bank(normalizer, horizon=5, action_block=5)

    assert bank.shape == (4, 5, 25)
    raw = bank.reshape(4, 5, 5, 5)[:, :, 0, :]
    assert np.all(np.abs(raw) <= 1.0)
    np.testing.assert_allclose(raw[:, 1, 2], -1.0)
    np.testing.assert_allclose(raw[:, 2, 2], -1.0)
    np.testing.assert_allclose(raw[:, 3, 4], -1.0)
    np.testing.assert_allclose(raw[:, 4, :2], raw[:, 0, :2])
