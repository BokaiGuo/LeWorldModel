import numpy as np
from experiments.spectral_reachability import p1_diag32_samebank_tail_20261101_codex as m


def test_schedule():
    branches = m.schedule(np.full((15, 2), .25), np.full((15, 2), .5))
    assert len(branches) == 6
    assert [r["prefix_role"] for r in branches] == ["selected"] * 2 + ["zero"] * 2 + ["greedy"] * 2
    assert branches[0]["suffix"] == branches[2]["suffix"] == branches[4]["suffix"]
    assert branches[1]["suffix"] == branches[3]["suffix"] == branches[5]["suffix"]


def test_tie():
    rows = [dict(candidate=c, iteration=i, q1=1.) for c in range(8) for i in range(33)]
    assert (m.choose_greedy(rows)["candidate"], m.choose_greedy(rows)["iteration"]) == (0, 0)


def test_sources():
    data = m.build()
    assert data["greedy"]["pool_size"] == 264
    assert data["selected"]["candidate"] == 4 and data["selected"]["iteration"] == 13
