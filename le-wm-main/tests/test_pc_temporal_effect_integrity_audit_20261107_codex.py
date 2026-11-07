import numpy as np

from experiments.pc_temporal_effect_integrity_audit_20261107_codex import centered_effect


def test_centering_cancels_reference_and_preserves_probe_identity():
    rng = np.random.default_rng(11)
    costs = rng.normal(size=(8, 9, 5))
    original = centered_effect(costs)
    modified = costs.copy()
    modified[:, 0, :] += np.arange(8)[:, None] * 100
    np.testing.assert_allclose(original, centered_effect(modified), atol=1e-13)
    altered = costs.copy()
    altered[0, 1, 2:] += 2
    assert not np.allclose(original[0], centered_effect(altered)[0])
