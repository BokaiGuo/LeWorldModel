from copy import deepcopy

import pytest

from experiments.spectral_reachability import audit_zero_signed_MAIN_20261028_v1 as m
from test_agent_mixedmode_zero_signed_comparison_20261028_v1 import fixture


def test_independent_hand_computed_signed_values():
    _, _, z, _, s, _ = fixture()
    result = m.recompute(z['rows'], s['rows'])
    assert len(result) == 4
    for row in result:
        assert row['spaces']['object_xy']['secant_changes'] == pytest.approx([2, 0])
        assert row['spaces']['object_xy']['response_changes'][2] < 0


def test_full_denominator_and_identity_fail_closed():
    _, _, z, _, s, _ = fixture()
    with pytest.raises(ValueError):
        m.recompute(z['rows'][:-1], s['rows'])
    bad = deepcopy(s['rows'])
    bad[5]['goal'][0] += 1
    with pytest.raises(ValueError):
        m.recompute(z['rows'], bad)
