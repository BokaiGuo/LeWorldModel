from copy import deepcopy

import pytest

from experiments.spectral_reachability import p1_second_feedback_saved_20261102 as subject


def test_verified_complete_actual_feedback_panel():
    result = subject.analyze()
    assert len(result["rows"]) == 8
    assert result["frames_verified"] == 112
    assert not result["P1_admission"] and not result["independent_confirmation"]
    assert all(not row["new_success"] and not row["old_success"] for row in result["rows"])


@pytest.mark.parametrize("kind", ["roster", "action", "censored", "missing_frame"])
def test_rejects_corrupted_trace(monkeypatch, kind):
    read = subject.physical.previous.read

    def modified(path, expected=None):
        value = read(path, expected)
        if path == subject.RUN / "report.json":
            value = deepcopy(value)
            if kind == "roster":
                value["rows"].pop()
            elif kind == "action":
                value["rows"][0]["transitions"][-1]["action"][0] += .01
            elif kind == "censored":
                value["rows"][0]["transitions"][-1]["terminated"] = True
            else:
                value["rows"][0]["observations"].pop()
        return value

    monkeypatch.setattr(subject.physical.previous, "read", modified)
    with pytest.raises(ValueError):
        subject.analyze()
