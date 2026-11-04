import pytest

from experiments.spectral_reachability import reacher_cem32_same_chain_physical_20261102 as run


def test_frozen_scored_roster():
    plan = run.prepare()
    assert plan["expected_branches"] == 6 and plan["P1_admission"] is False
    assert [[b["prefix_index"] for b in r["branches"]] for r in plan["roots"]] == [[2, 0, 6], [3, 0, 6]]
    assert all([b["role"] for b in r["branches"]] == ["selected", "neutral", "same_bank_greedy"]
               for r in plan["roots"])


def test_wrong_authority_does_not_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="authority"):
        run.execute("irrelevant", "irrelevant", output, "wrong")
    assert not output.exists()
