import numpy as np

from experiments.spectral_reachability import (
    p1_dev302_v10_common_tail_physical_20261101_codex as subject,
)


def test_v10_common_tail_is_identical_after_distinct_prefixes():
    value = subject.build()
    tails = [np.asarray(row["actions"], np.float32)[5:] for row in value["roles"]]
    assert all(np.array_equal(tails[0], tail) for tail in tails[1:])
    assert len({row["prefix_sha256"] for row in value["roles"]}) == 4
    assert value["mechanism_intervention"] is True
    assert value["performance_comparison"] is False
    assert value["P1_admission"] is False
