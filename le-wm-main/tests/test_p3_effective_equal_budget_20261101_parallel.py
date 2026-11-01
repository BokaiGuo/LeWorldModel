import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p3_effective_equal_budget_20261101_parallel as panel


@pytest.fixture(scope="module")
def actual():
    primary = panel.load_json(panel.PRIMARY_CONSUME, panel.PRIMARY_CONSUME_SHA256)
    primary = panel._certify_returned(primary)
    report = json.loads((panel.PRIMARY_RUN / "report.json").read_bytes())
    events = panel.journal.read(
        panel.PRIMARY_RUN / "events.jsonl", report["journal_tail"], panel.host.CAP
    )
    resolved = panel.load_json(panel.RESOLVED_CURVE, panel.RESOLVED_CURVE_SHA256)
    return primary, report, events, resolved


def assemble(actual, primary=None, report=None, events=None, resolved=None):
    p, r, e, w = actual
    return panel.assemble(
        copy.deepcopy(primary if primary is not None else p),
        copy.deepcopy(events if events is not None else e),
        copy.deepcopy(report if report is not None else r),
        copy.deepcopy(resolved if resolved is not None else w),
        source_hashes={"test": "fixture"},
    )


def test_fixed_strategies_are_different_and_measured_axes_equal(actual):
    result = assemble(actual)
    curve = result["primary_online_cuda_contract"]["curve"]
    assert [row["budget_point"] for row in curve] == [20, 30]
    assert curve[0]["signal"]["allocation"] == [15, 5]
    assert curve[0]["reverse"]["allocation"] == [5, 15]
    assert curve[1]["signal"]["allocation"] == [22, 8]
    assert curve[1]["reverse"]["allocation"] == [8, 22]
    assert all(row["equal_measured_axes"] for row in curve)
    assert result["budget_accounting"]["full_total_compute_equal"] is False
    assert result["scientific_boundary"]["P3_admission"] is False
    assert result["scientific_boundary"]["physical_outcome"] is False


@pytest.mark.parametrize("field", ["allocation", "weights", "attempted", "returned", "winner"])
def test_primary_tamper_is_rejected(actual, field):
    primary = copy.deepcopy(actual[0])
    row = next(
        item for item in primary["rows"] if item["point"] == 20 and item["arm"] == "signal"
    )
    if field == "allocation":
        row["allocation"] = [10, 10]
    elif field == "weights":
        row["weights"] = [1.0, 1.0]
    elif field == "attempted":
        row["attempted"]["metric"] -= 1
    elif field == "returned":
        row["returned"]["metric"] -= 1
    else:
        row["winners"][0]["cost"] = float("nan")
    with pytest.raises(ValueError):
        assemble(actual, primary=primary)


def test_resolved_wall_is_separate_and_tamper_rejected(actual):
    result = assemble(actual)
    auxiliary = result["auxiliary_resolved_wall_contract"]
    assert "different execution contract" in auxiliary["warning"]
    assert len(auxiliary["rows"]) == 4
    resolved = copy.deepcopy(actual[3])
    resolved["points"][0]["arms"][1]["wall_seconds"] = -1
    with pytest.raises(ValueError, match="resolved consumed timing row"):
        assemble(actual, resolved=resolved)


def test_incomplete_arm_is_not_imputed(actual):
    resolved = copy.deepcopy(actual[3])
    resolved["points"][0]["arms"][1]["status"] = "missing_or_failed"
    resolved["points"][0]["arms"][1]["axis_attempted"] = None
    with pytest.raises(ValueError, match="resolved consumed timing row"):
        assemble(actual, resolved=resolved)


def test_journal_chain_tamper_is_rejected(tmp_path):
    report = json.loads((panel.PRIMARY_RUN / "report.json").read_bytes())
    lines = (panel.PRIMARY_RUN / "events.jsonl").read_bytes().splitlines()
    row = json.loads(lines[10])
    row["event"]["kind"] = "tampered"
    lines[10] = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(ValueError, match="journal hash chain"):
        panel.journal.read(path, report["journal_tail"], panel.host.CAP)


def test_pinned_source_hash_tamper_is_rejected(tmp_path):
    path = tmp_path / "value.json"
    path.write_text('{"value":1}')
    with pytest.raises(ValueError, match="sha256 mismatch"):
        panel.load_json(path, "0" * 64)


def test_actual_saved_reconstruction_and_gate_boundary():
    result = panel.build_saved()
    assert result["status"].endswith("not_scientifically_admitted")
    assert result["primary_online_cuda_contract"]["model_before"] == result[
        "primary_online_cuda_contract"
    ]["model_after"]
    assert result["scientific_boundary"] == {
        "known_roots": True,
        "physical_outcome": False,
        "independent_confirmation": False,
        "model_cost_only": True,
        "P3_admission": False,
        "C3_admission": False,
        "gate_relaxed": False,
    }
