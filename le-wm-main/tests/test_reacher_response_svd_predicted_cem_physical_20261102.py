import json

import numpy as np
import pytest

from experiments.spectral_reachability import reacher_response_svd_predicted_cem_physical_20261102 as runner


def test_plan_reconstructs_and_preserves_recorded_raw_prefix():
    plan = runner.make_plan()
    saved = json.loads((runner.ROOT / "results/reacher_response_svd_predicted_tail_PLAN_20261102_v1.json").read_text())
    old = json.loads(runner.OLD_PLAN.read_text())
    assert plan == saved
    for root, prior in zip(plan["roots"], old["roots"], strict=True):
        assert [b["role"] for b in root["branches"]] == ["selected", "neutral"]
        for branch in root["branches"]:
            original = prior["arms"][branch["prefix_index"]]["raw_prefix"]
            np.testing.assert_array_equal(np.asarray(branch["raw_actions"][:5], np.float32),
                                          np.asarray(original, np.float32))
            assert branch["prefix_cpu_decode_max_abs_error"] <= 1e-6


def test_bad_authority_fails_before_output(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="authority"):
        runner.run("irrelevant", "irrelevant", output, "wrong")
    assert not output.exists()
