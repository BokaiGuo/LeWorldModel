import pytest

from experiments.spectral_reachability import reacher_cem_live_cp5_20261102 as run


def test_fixed_roster_and_budget():
    plan = run.prepare()
    assert [root["seed"] for root in plan["roots"]] == [20263160, 20263161]
    assert all([arm["proposal_index"] for arm in root["arms"]] == [0, 12] for root in plan["roots"])
    assert plan["cem"] == {"samples": 32, "steps": 3, "elite": 4, "seed_rule": "root+5000"}
    assert plan["P1_admission"] is False and plan["expected_arms"] == 4


def test_rejects_without_authority(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "no_output"
    with pytest.raises(ValueError, match="authority"):
        run.run("none", "none", output, "wrong")
    assert not output.exists()
