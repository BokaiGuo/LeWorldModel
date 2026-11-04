import numpy as np
import pytest

from experiments.spectral_reachability import p1_second_feedback_replacement_20261102 as subject


def test_tie_keeps_old_and_scores_not_nominal_alone():
    assert subject.select(np.ones((2, 5)))["selected"] == "old"
    assert subject.select([[1, 1, 1, 1, 1], [0, 0, 0, 0, 3]])["selected"] == "old"
    assert subject.select([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])["selected"] == "new"


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError):
        subject.select([[1, 1, 1, 1, 1], [0, 0, 0, 0, float("nan")]])


def test_protocol_has_no_outcome_fields():
    spec = subject.protocol()
    assert len(spec["rows"]) == 8
    assert all(not ({"state", "reward", "physical_evaluation_only"} & set(r)) for r in spec["rows"])
    assert spec["per_arm"]["predict_samples"] == 10


def test_saved_cuda_scores_and_hook_ledger():
    root = subject.ROOT / "results/p1_second_feedback_replacement_20261102_RUN_DEV01"
    report = subject.base.first.fixed(
        root / "report.json", "54d2a2517f5467ec08b286fa2c6a1c89988f32fa6fd196e4b80bf13b48776eaf")
    assert report["model_before"] == report["model_after"]
    assert len(report["rows"]) == 8
    for row in report["rows"]:
        recomputed = subject.select(row["variant_scores"])
        assert row["selected"] == recomputed["selected"]
        np.testing.assert_allclose(row["robust"], recomputed["robust"], rtol=0, atol=1e-12)
        for name in ("action_encoder", "predictor"):
            assert report["hook_ledger"][row["arm"] + ":score:" + name] == {"calls": 1, "samples": 10}
