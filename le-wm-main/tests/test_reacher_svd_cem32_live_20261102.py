import pytest

from experiments.spectral_reachability import reacher_svd_cem32_live_20261102 as run


def test_roster_is_derived_from_sealed_model_selection():
    plan = run.prepare()
    assert plan["expected_arms"] == 6 and plan["P1_admission"] is False
    assert [[a["proposal_index"] for a in r["arms"]] for r in plan["roots"]] == [[2, 0, 6], [3, 0, 6]]
    assert all([a["role"] for a in root["arms"]] == ["selected", "neutral", "same_bank_greedy"]
               for root in plan["roots"])


def test_rejects_wrong_authority(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="authority"):
        run.run("irrelevant", "irrelevant", output, "wrong")
    assert not output.exists()
