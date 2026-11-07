import numpy as np

from experiments.pc_exact_prefix_quota_coverage_audit_20261107_codex import identities, key


def test_signed_zero_and_duplicate_full_actions():
    assert key(np.array([-0.0])) == key(np.array([0.0]))
    pool = np.zeros((8, 15, 2))
    a = np.zeros((8, 10, 2))
    b = np.zeros((64, 10, 2))
    b[1, 0, 0] = 1
    value = identities(pool, a, b, {"x": list(range(8))})
    assert value["unique_prefixes"] == 1
    assert value["unique_b_tails"] == 2
    assert value["arms"]["x"] == {"repeats_within": 6, "already_common": 7, "unique_novel": 1}
