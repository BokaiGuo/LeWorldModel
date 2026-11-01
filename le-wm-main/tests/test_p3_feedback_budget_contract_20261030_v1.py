from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import p3_feedback_budget_contract_20261030_v1 as m


def slot():
    metric = m.metrics([1, 2, 3, 4, 0, 5, 6], [0] * 7)
    return {"root_sha256": "root", "goal": [0] * 7, "cp0_identity": [{"state": [0]}, "pixel"],
            "attempted": deepcopy(m.EXPECTED), "completed": deepcopy(m.EXPECTED),
            "shape_counts": {"shape": 2}, "checkpoints": {str(i): metric for i in (0, 5, 10, 15, 20)}}


def test_pair_measured_and_shape_separate():
    a, b = slot(), slot()
    assert m.compare_pair(a, b)["measured_vector_match"] is True
    b["shape_counts"] = {"different": 2}
    assert m.compare_pair(a, b)["measured_vector_match"] is True
    assert m.compare_pair(a, b)["hook_shape_multiset_match"] is False


@pytest.mark.parametrize("field", ["attempted", "completed"])
def test_changed_or_missing_axis_rejected(field):
    a, b = slot(), slot()
    b[field].pop("jacobian_API")
    with pytest.raises(ValueError, match="budget"):
        m.compare_pair(a, b)


@pytest.mark.parametrize("field", ["root_sha256", "goal", "cp0_identity"])
def test_pair_identity_rejected(field):
    a, b = slot(), slot()
    b[field] = "other"
    with pytest.raises(ValueError, match="identity"):
        m.compare_pair(a, b)


def test_incomplete_is_unknown_not_zero():
    assert m.compare_pair(slot(), None) == {
        "measured_vector_match": None, "hook_shape_multiset_match": None, "checkpoints": None}


def test_native_metric_includes_velocity():
    r = m.metrics([3, 4, 0, 0, 0, 6, 8], [0] * 7)
    assert r["native7_distance"] == pytest.approx(125 ** .5)
    assert r["native_success"] is True
    with pytest.raises(ValueError):
        m.metrics([float("nan")] * 7, [0] * 7)


def test_shape_includes_status_and_directions():
    x = {"resource": "grad", "shape": [1, 3], "directions": 3, "status": "returned"}
    y = dict(x, directions=4)
    assert m.shapes([x]) != m.shapes([y])


def test_pinned_inputs_exclusive_and_duplicate_json(tmp_path):
    p = tmp_path / "saved.json"
    h = m.save_new(p, {"a": 1})
    assert m.read_pinned(p, h) == {"a": 1}
    with pytest.raises(FileExistsError):
        m.save_new(p, {"a": 2})
    with pytest.raises(ValueError):
        m.read_pinned(p, "0" * 64)
    q = tmp_path / "duplicate.json"
    q.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match="duplicate"):
        m.read_pinned(q, m.sha(q))


def test_contract_is_fixed_and_never_scientific():
    c = m.contract()
    assert c["numerical_attempted_and_completed"] == m.EXPECTED
    assert c["scientific_pass"] is False
    assert c["fresh_roots"] == 0
    assert json.loads(json.dumps(c)) == c


def test_partial_receipt_preserves_denominator(tmp_path, monkeypatch):
    prepared = tmp_path / "prepare.json"
    prepared.write_text("{}")
    monkeypatch.setattr(m, "PREPARE_SHA", m.sha(prepared))
    contract = tmp_path / "contract.json"
    contract_sha = m.save_new(contract, m.contract())
    run = tmp_path / "run"
    run.mkdir()
    report = {"prepare_sha256": m.PREPARE_SHA, "scientific_pass": False,
              "rows": [{"slot": s, "root": r, "policy": p, "status": "not_started"}
                       for s, r, p in m.ROSTER]}
    report_sha = m.save_new(run / "report.json", report)
    completion_sha = m.save_new(run / "completion.json",
                                {"report_sha256": report_sha, "scientific_pass": False})
    receipt = tmp_path / "receipt.json"
    receipt_sha = m.save_new(receipt, {"report_sha256": report_sha, "completion_sha256": completion_sha,
                                      "scientific_pass": False, "full_saved_strict": False})
    result = m.inspect(run, receipt, receipt_sha, prepared, contract, contract_sha)
    assert result["slots"] == [None] * 4
    assert len(result["slot_statuses"]) == 4
    assert result["pairs"][0]["measured_vector_match"] is None
    assert result["status"] == "partial_unmatched_unknown"
    with pytest.raises(ValueError):
        m.inspect(run, receipt, "0" * 64, prepared, contract, contract_sha)
