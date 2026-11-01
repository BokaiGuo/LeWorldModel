from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import p1_candidate_short_horizon_alignment_20261031 as subject


@pytest.fixture(scope="module")
def data():
    assert subject.prior.base.sha(subject.MID) == subject.MID_SHA
    assert subject.prior.base.sha(subject.real.INPUT) == subject.real.SHA
    return json.loads(subject.MID.read_bytes()), json.loads(subject.real.INPUT.read_bytes())


def test_fixed_panel_pair_denominator(data):
    result = subject.analyze(*data)
    assert result["five_step_pair_counts"] == {"agree": 111, "reverse": 33}
    assert len(result["pairwise"]) == 24 * 6
    assert len(result["selectors"]) == 24
    assert not result["P1_admission"]


def test_midpoint_pixel_and_candidate_tampering_rejected(data):
    mid, actual = deepcopy(data)
    mid["rows"][0]["midpoint_pixel_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="saved midpoint identity"):
        subject.analyze(mid, actual)
    mid, actual = deepcopy(data)
    actual["rows"][1] = deepcopy(actual["rows"][0])
    with pytest.raises(ValueError, match="candidate identity"):
        subject.analyze(mid, actual)
