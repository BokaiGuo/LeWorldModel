import numpy as np
import pytest

from experiments.pc_exact_prefix_quota_proposal_20261107_codex import select


def test_quotas_cover_all_prefixes_and_stable_ties():
    p = np.zeros((8, 64))
    result = select({"descriptor": p, "additive": p, "shuffled": p}, 3)
    assert result["global_descriptor"] == list(range(8))
    for arm in ("quota_descriptor", "quota_additive", "quota_shuffled"):
        assert result[arm] == [i * 64 for i in range(8)]
    assert [i // 64 for i in result["quota_random"]] == list(range(8))
    assert result == select({"descriptor": p, "additive": p, "shuffled": p}, 3)


def test_rejects_nonfinite_predictions():
    p = np.zeros((8, 64))
    p[0, 0] = np.nan
    with pytest.raises(ValueError):
        select({"descriptor": p, "additive": p, "shuffled": p}, 3)
