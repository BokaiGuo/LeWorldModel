"""Pure numeric both-scale common-control capability comparisons."""
import copy

import pytest

from experiments.spectral_reachability.analyze_reacher_d2p_metric_capability import gain_differences


def fits():
    return {str(i):[{"physical_rgb_fd":{"gain_fraction":v}} for v in values]
        for i,values in ((0,(.1,.2)),(2,(.2,.3)),(3,(.4,.1)))}


def test_both_fixed_controls_and_both_scales_are_retained():
    values=fits()
    before=copy.deepcopy(values)
    result=gain_differences(values,{"metric_candidate":3,"zero":0,"greedy":2})
    assert result[0]==pytest.approx([.3,.2])
    assert result[1]==pytest.approx([-.1,-.2])
    assert values==before


def test_greedy_fallback_is_not_a_positive_capability_witness():
    result=gain_differences(fits(),{"metric_candidate":2,"zero":0,"greedy":2})
    assert result[0][1]==result[1][1]==0.
    assert not all(v>1e-10 for scale in result for v in scale)


def test_full_compiler_native_and_common_roles_are_not_interchanged():
    roles = {"native_candidate":2,"common_candidate":3,"zero":0,"greedy":2}
    native = gain_differences(fits(),roles,"native_candidate")
    common = gain_differences(fits(),roles,"common_candidate")
    assert native[0][1] == native[1][1] == 0.
    assert common[0] == pytest.approx([.3,.2])
    assert common[1] == pytest.approx([-.1,-.2])
    with pytest.raises(KeyError): gain_differences(fits(),roles)
