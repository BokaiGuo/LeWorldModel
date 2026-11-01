import json

from experiments.spectral_reachability import p1_response_midpoint_online_20261031 as online


def test_fixed_online_roster_is_bound_to_prior_physical_contract():
    path = online.ROOT / "results/p1_response_midpoint_online_20261031_prepare_RUN01.json"
    assert online.sha(path) == "86a81ddb5252ca181fd1f53e7b43a2cdae639256cf9a38c0b83217141a7269a0"
    prepared = json.loads(path.read_text())
    assert prepared == online.build()
    assert len(prepared["physical"]["anchors"]) == 13
    assert sorted(row["index"] for row in prepared["physical"]["excluded"]) == [1, 3, 5]
    assert prepared["fixed_rows"] == 16
    assert prepared["P1_admission"] is False
    assert prepared["independent_confirmation"] is False
