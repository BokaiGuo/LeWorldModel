import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_blind_uphill_bank_20261101_exclusive as m


def test_choice_tie_and_no_physical_fallback():
    assert m.choose(1., [0.9, 1., 1.2, 0., 0., 0., 0., 0.],
                    [0., .4, .4, 0., 0., 0., 0., 0.]) == 1
    assert m.choose(1., [0.] * 8, [0.] * 8) is None
    with pytest.raises(ValueError):
        m.choose(1., [np.nan] * 8, [0.] * 8)


def test_sources_reconstruct_both_real_model_contexts():
    value = m.protocol()
    assert [x["slot"] for x in value["inputs"]] == [0, 1]
    assert value["inputs"][0]["root_sha256"] != value["inputs"][1]["root_sha256"]
    assert value["physical_release"] is False and value["P1_admission"] is False
