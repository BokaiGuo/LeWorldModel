import numpy as np

from experiments.spectral_reachability import codex_p1_multistart_gradient_audit_20261030_main as audit


def test_directions_are_fixed_unit_and_distinct():
    first = audit.direction(0, 0)
    assert first.shape == (1, 2, 5, 2)
    assert np.array_equal(first, audit.direction(0, 0))
    assert np.isclose(np.linalg.norm(first), 1.0)
    assert not np.array_equal(first, audit.direction(0, 1))
