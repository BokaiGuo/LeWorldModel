import copy

import pytest

from experiments.spectral_reachability import p1_uphill_triplet_summary_20261031_main as s


def test_join_and_reject_cross_run_outcome_mismatch():
    values = [s.base.read(*x) for x in (
        (s.PAIR, s.PAIR_SHA), (s.audit.OPEN_COMPARE, s.audit.OPEN_COMPARE_SHA),
        (s.TRUST_OPEN, s.TRUST_OPEN_SHA), (s.TRUST_REAL, s.TRUST_REAL_SHA))]
    result = s.join(*values)
    assert len(result["groups"]) == 8
    assert all(len(group["prefixes"]) == 3 for group in result["groups"])
    modified = copy.deepcopy(values)
    modified[2]["rows"][1]["open_last_success"] = False
    with pytest.raises(Exception, match="outcome binding"):
        s.join(*modified)
