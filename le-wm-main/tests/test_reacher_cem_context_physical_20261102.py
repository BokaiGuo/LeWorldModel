import numpy as np
import pytest

from experiments.spectral_reachability import reacher_cem_context_physical_20261102 as run


def test_complete_pair_roster_and_prefix_binding():
    plan = run.prepare()
    old = __import__("json").loads(run.run.physical.OLD_PLAN.read_text())
    assert plan["expected_branches"] == 48
    assert plan["P1_admission"] is False
    for root, prior in zip(plan["roots"], old["roots"], strict=True):
        assert len(root["branches"]) == 24
        for i in range(0, 24, 2):
            a, b = root["branches"][i:i + 2]
            assert (a["context"], b["context"]) == ("predicted", "observed")
            assert all(a[key] == b[key] for key in ("prefix_index", "budget", "cem_seed"))
            for arm in (a, b):
                np.testing.assert_array_equal(np.asarray(arm["raw_actions"][:5], np.float32),
                                              np.asarray(prior["arms"][arm["prefix_index"]]["raw_prefix"],
                                                         np.float32))


def test_wrong_authority_fails_before_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "no_output"
    with pytest.raises(ValueError, match="authority"):
        run.execute("irrelevant", "irrelevant", output, "wrong")
    assert not output.exists()
