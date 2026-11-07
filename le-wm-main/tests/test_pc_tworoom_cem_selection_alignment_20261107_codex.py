import numpy as np
import pytest

from experiments.pc_tworoom_cem_selection_alignment_20261107_codex import reconstruct, validate_trace


def test_reconstructs_fixed_same_tail_with_reversed_prefix():
    prefix = np.arange(20, dtype=np.float32).reshape(10, 2) / 20
    tails = np.zeros((9, 15, 2), np.float32)
    tails[7] = .3
    row = {"selected": {"descriptor": [0, 2]}, "prefixes": [prefix], "tails": tails}
    a, b = reconstruct(row, "selected"), reconstruct(row, "reversed")
    np.testing.assert_array_equal(a[:10], prefix)
    np.testing.assert_array_equal(b[:10], prefix[::-1])
    np.testing.assert_array_equal(a[10:], b[10:])


def test_trace_check_fails_on_truth_change():
    trace = np.zeros((25, 3))
    validate_trace(trace, trace)
    other = trace.copy()
    other[4, 1] = 1
    with pytest.raises(AssertionError):
        validate_trace(trace, other)
