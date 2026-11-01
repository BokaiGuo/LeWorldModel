import copy

import pytest
from experiments.spectral_reachability import p1_uphill_cp5_initializer_compare_20261030 as compare


def test_identity_and_censoring():
    zero = compare.base.read_checked(compare.ZERO, compare.ZERO_SHA)
    warm = copy.deepcopy(zero)
    warm["rows"][0]["independent_suffix_20step"]["endpoint"] = None
    warm["rows"][0]["independent_suffix_20step"]["status"] = "native_stopped"
    result = compare.compare(zero, warm)
    assert len(result) == 24
    assert result[0]["warm_minus_zero_object_position_error"] is None
    warm["rows"][0]["prefix_cp5_pixel_sha256"] = "mismatch"
    with pytest.raises(ValueError, match="paired identity"):
        compare.compare(zero, warm)
