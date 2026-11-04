from copy import deepcopy

import pytest

from experiments.spectral_reachability import p1_second_feedback_prediction_saved_20261102 as subject


def test_cuda_report_vectors_and_ledger():
    result = subject.analyze()
    assert len(result["rows"]) == 8
    assert sum(row["ranking"]["ranking"] == "opposite" for row in result["rows"]) == 1
    assert not result["P1_admission"]


@pytest.mark.parametrize("kind", ["vector", "ledger", "roster"])
def test_corrupted_report_rejected(monkeypatch, kind):
    fixed = subject.audit.planner.first.fixed

    def changed(path, sha):
        value = fixed(path, sha)
        if path == subject.RUN / "report.json":
            value = deepcopy(value)
            if kind == "vector":
                value["rows"][0]["vectors"]["pred_old"][0] += 1
            elif kind == "ledger":
                value["hook_ledger"][value["rows"][0]["arm"] + ":predict:predictor"]["calls"] = 1
            else:
                value["rows"].pop()
        return value
    monkeypatch.setattr(subject.audit.planner.first, "fixed", changed)
    with pytest.raises(ValueError):
        subject.analyze()
