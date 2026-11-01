import pytest

from experiments.spectral_reachability import p1_response_prefix_censored_consume_20261031 as consume


def test_rejects_partial_batch_before_reading_journal(tmp_path):
    report = {"schema": consume.run.NAME + ".report", "status": "partial_or_failed",
              "fixed_rows": 16, "rows": [{}] * 16, "errors": [],
              "P1_admission": False, "independent_confirmation": False, "new_neural_calls": 0}
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        consume.check(tmp_path, report, {"plans": [{}] * 16})


def test_rejects_missing_branch_before_reading_journal(tmp_path):
    report = {"schema": consume.run.NAME + ".report", "status": "physical_recorded_not_independently_consumed",
              "fixed_rows": 16, "rows": [{}] * 15, "errors": [],
              "P1_admission": False, "independent_confirmation": False, "new_neural_calls": 0}
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        consume.check(tmp_path, report, {"plans": [{}] * 16})
