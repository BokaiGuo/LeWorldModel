"""Saved-only characterization, not a test asserting scientific validity."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_dev305_contact_equal_scale_saved_20261102 as equal
from experiments.spectral_reachability import p1_dev305_contact_rgb_feedback_saved_20261102 as feedback


@pytest.mark.parametrize("consumer", [equal, feedback])
def test_real_censored_rows_remain_labeled_with_unknown_success(monkeypatch, consumer):
    # Source protocol rebuild currently fails on unrelated pinned-source drift.
    # Isolate this consumer's saved-row semantics without claiming closure.
    prep = consumer.base.read(consumer.ROOT / "results" / f"{consumer.source.NAME}_PREP_DEV01.json",
                              consumer.PREP_SHA)
    monkeypatch.setattr(consumer.source, "protocol", lambda: prep)
    result = consumer.analyze()
    stopped = [row for row in result["rows"] if row["status"] in
               ("visibility_guard_stop", "conservative_margin_stop")]
    assert len(stopped) == (5 if consumer is equal else 4)
    assert all(row["steps"] < 5 and row["native_success"] is None for row in stopped)
    assert all(row["native_success"] is False for row in result["rows"]
               if row["status"] == "fixed_horizon_completed")
    assert result["p1_admission"] is False


@pytest.mark.parametrize("consumer", [equal, feedback])
def test_consumer_rejects_inconsistent_failed_report_status_fixture(monkeypatch, consumer):
    """Input mutation tests only; the real saved report itself is recorded/no-error."""
    original_read = consumer.base.read
    target = consumer.ROOT / "results" / f"{consumer.source.NAME}_RUN_DEV01/report.json"
    seen = []

    def altered(path, *args, **kwargs):
        value = original_read(path, *args, **kwargs)
        if Path(path) == target:
            value = deepcopy(value)
            assert value["status"] == "recorded" and value["errors"] == []
            value["status"] = "failed"
            seen.append(True)
        return value

    monkeypatch.setattr(consumer.base, "read", altered)
    with pytest.raises(ValueError, match="report status/errors"):
        consumer.analyze()
    assert seen == [True]


@pytest.mark.parametrize("consumer", [equal, feedback])
def test_consumer_rejects_nonempty_errors_fixture(monkeypatch, consumer):
    original_read = consumer.base.read
    target = consumer.ROOT / "results" / f"{consumer.source.NAME}_RUN_DEV01/report.json"

    def altered(path, *args, **kwargs):
        value = original_read(path, *args, **kwargs)
        if Path(path) == target:
            value = deepcopy(value)
            value["errors"] = [{"message": "injected engineering failure"}]
        return value

    monkeypatch.setattr(consumer.base, "read", altered)
    with pytest.raises(ValueError, match="report status/errors"):
        consumer.analyze()
