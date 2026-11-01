import json

from experiments.spectral_reachability import p1_response_prefix_h10_20261031 as run


def test_sealed_common_roster_and_denominator():
    prepared = run.ROOT / "results/p1_response_prefix_h10_20261031_prepare_RUN01.json"
    assert run.sha(prepared) == "8577190effc65a61e1b69cee6dcdbd1c2814a3da79b8383d468a76e03f0c05bc"
    data = json.loads(prepared.read_bytes())
    assert data["fixed_rows"] == 16
    assert data["fixed_branches"] == 588
    assert data["censored_before_cp5"] == [1, 5]
    assert len(data["anchors"]) == 14
    assert sorted([a["index"] for a in data["anchors"]] + data["censored_before_cp5"]) == list(range(16))
    for anchor in data["anchors"]:
        bank = data["banks"][anchor["logical_pair"]]
        assert len(bank) == 42
        assert [p["index"] for p in bank] == list(range(42))
    assert data["P1_admission"] is False
