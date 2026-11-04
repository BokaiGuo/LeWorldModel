"""Numerically safe pairwise ranking for finite candidate score arrays."""

import numpy as np


def ranking(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if (first.ndim != 1 or first.shape != second.shape or first.size < 2
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("finite paired one-dimensional scores required")
    i, j = np.triu_indices(first.size, 1)
    # Compare values directly: multiplication can underflow, subtraction can
    # overflow, even when all input scores are finite.
    a_less, a_greater = first[i] < first[j], first[i] > first[j]
    b_less, b_greater = second[i] < second[j], second[i] > second[j]
    a_tie, b_tie = ~(a_less | a_greater), ~(b_less | b_greater)
    return {
        "comparable": int(np.sum(~a_tie & ~b_tie)),
        "reversed": int(np.sum((a_less & b_greater) | (a_greater & b_less))),
        "first_ties": int(a_tie.sum()),
        "second_ties": int(b_tie.sum()),
    }
