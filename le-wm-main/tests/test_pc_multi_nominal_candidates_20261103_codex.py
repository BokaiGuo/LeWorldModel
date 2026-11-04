import numpy as np
import pytest
import json
from pathlib import Path

from experiments.spectral_reachability import pc_multi_nominal_candidates_20261103_codex as module
from experiments import pc_multi_nominal_rgb_dev_20261103_codex as real


def test_multi_nominal_lengths_and_instrumented_calls():
    calls = {"predict": 0, "svd": 0}
    def predict(actions):
        calls["predict"] += 1
        assert np.max(np.abs(actions)) <= 1
        return np.asarray([np.sum(actions[:, 0]**2), np.sum(actions[:, 1])])
    def svd(*args, **kwargs):
        calls["svd"] += 1
        return np.linalg.svd(*args, **kwargs)
    result = module.build_candidates(predict, svd=svd)
    assert calls == {"predict": 192, "svd": 6}
    assert len(result["evidence"]) == 6
    assert len(result["candidates"]) == 8
    assert {x["horizon"] for x in result["candidates"]} == {5, 10}
    assert all(np.max(np.abs(x["prefix"])) <= 1 for x in result["candidates"])


def test_zero_response_and_invalid_policy():
    result = module.build_candidates(lambda x: np.zeros(2), lengths=(5,))
    assert not result["candidates"]
    assert len(result["evidence"]) == 3
    with pytest.raises(ValueError):
        module.build_candidates(lambda x: x, lengths=(3,))


def test_real_rgb_candidate_receipt():
    result = json.loads(real.OUTPUT.read_text())
    sha = real.old.prior.sha
    assert result["runner_sha256"] == sha(Path(real.__file__))
    assert result["generator_sha256"] == sha(Path(module.__file__))
    assert result["protocol_sha256"] == sha(real.PROTOCOL)
    assert result["history_sha256"] == sha(real.old.HISTORY)
    assert len(result["rows"]) == 4
    for row in result["rows"]:
        assert len(row["candidates"]) == 8
        assert len(row["evidence"]) == 6
        assert row["counts"]["prediction_token_samples"] == 318
        assert row["counts"]["prediction_calls"] == 192
        assert row["svd"]["calls"] == 6
