import pytest

from experiments.spectral_reachability import reacher_response_svd_cem32_20261102 as run


def test_plan_fixes_search_and_known_root_scope():
    plan = run.make_plan()
    assert (plan["samples"], plan["iterations"], plan["elites"], plan["future_horizon"]) == (32, 3, 4, 2)
    assert "known roots" in plan["scope"]


def test_no_output_without_authority(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    path = tmp_path / "no_output"
    with pytest.raises(ValueError, match="authority"):
        run.run("irrelevant", "irrelevant", path, "wrong")
    assert not path.exists()
