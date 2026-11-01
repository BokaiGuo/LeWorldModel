import pytest
from experiments.spectral_reachability import codex_p2_candidate_censored_pairs_20261030_v1 as m


def test_actual_censored_common_times():
    x=m.build()
    assert len(x["jobs"])==4 and x["known_roots"]==2 and not x["P2_admission"]
    assert x["jobs"][0]["checkpoints"][-1]["fixed_index_1"] is None
    assert x["jobs"][1]["checkpoints"][-2]["fixed_index_1"] is None
    assert x["jobs"][2]["checkpoints"][-1]["fixed_index_1"] is not None


def test_missing_pair_rejected():
    with pytest.raises(ValueError):
        m.pair([])
