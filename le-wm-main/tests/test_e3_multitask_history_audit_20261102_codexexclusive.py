import json

import pytest
import numpy as np

from experiments.spectral_reachability import e3_multitask_history_audit_20261102_codexexclusive as audit


def test_failure_is_retained_not_promoted(monkeypatch):
    def fail(index, spec):
        raise ValueError(f"bad cell {index}")
    monkeypatch.setattr(audit, "reconstruct", fail)
    result = audit.build()
    assert not result["complete"]
    assert all(row["state"] == "audit_failed" and row["nodes"] is None for row in result["rows"])
    assert result["fixed_denominator"] == 16
    assert 12 in result["uninspected"]
    assert result["scientific_admission"] is False


def test_exclusive_output_cannot_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(audit.cross, "RESULTS", tmp_path)
    monkeypatch.setattr(audit, "build", lambda: {"complete": True})
    out = tmp_path / "e3_multitask_history_audit_20261102_codexexclusive_test.json"
    out.write_text("preserve")
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(out)])
    with pytest.raises(FileExistsError):
        audit.main()
    assert out.read_text() == "preserve"


def test_output_scope_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(audit.cross, "RESULTS", tmp_path)
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(tmp_path / "wrong.json")])
    with pytest.raises(ValueError, match="exclusive output"):
        audit.main()


def test_result_json_is_strict():
    with pytest.raises(ValueError):
        json.dumps({"cost": float("nan")}, allow_nan=False)


def test_packet_rejects_changed_actions_and_stop():
    actions = np.arange(20, dtype="<f4").reshape(10, 2)
    controls = [dict(step=i+1, raw_bits=row.tobytes().hex(), terminated=False, truncated=False)
                for i, row in enumerate(actions)]
    assert audit.check_packet(controls, 10, actions[5:]).hex() == actions[5:].tobytes().hex()
    changed = actions[5:].copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match="executed action"):
        audit.check_packet(controls, 10, changed)
    controls[6]["truncated"] = True
    with pytest.raises(ValueError, match="stopped"):
        audit.check_packet(controls, 10, actions[5:])
