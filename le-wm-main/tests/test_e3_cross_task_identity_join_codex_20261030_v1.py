"""Offline join tests; no actual model, CUDA, or physical run."""
from copy import deepcopy

import pytest

from experiments.spectral_reachability import e3_cross_task_identity_join_codex_20261030_v1 as m


def good(index):
    return dict(binding_chunks_complete=True, base=dict(cell=index, status="completed20"),
                full_saved_strict=False, scientific_admission=False,
                identity=dict(pending=None, binding=dict(model_sha256=m.resolver.MODEL_SHA,
                    prepare_sha256=m.PREPARE_SHA, device="cuda:0"),
                    encodes=[dict(boundary=10, outcome="return"), dict(boundary=15, outcome="return")]))


def test_join_preserves_fixed_denominator_and_uninspected_cells(monkeypatch):
    calls = []
    def inspect(index, spec):
        calls.append(index)
        return m.check_inspection(index, good(index))
    monkeypatch.setattr(m, "inspect_cell", inspect)
    result = m.build()
    assert calls == [0, 1, 8, 9]
    assert result["joined_cells"] == calls and result["four_cell_saved_join_complete"]
    assert len(result["rows"]) == 16
    assert all(r["identity"] is None for r in result["rows"] if r["index"] not in calls)
    assert result["same_model_object_across_processes"] == "unknown"
    assert not result["scientific_admission"] and not result["full_saved_strict"]


@pytest.mark.parametrize("defect", ["partial", "publication_only", "wrong_cell", "model", "prepare", "missing_encode",
                                    "reverse", "failed_encode", "pending", "missing_binding", "false_gate"])
def test_bad_identity_is_rejected(defect):
    value = good(8)
    if defect == "partial":
        value["base"]["status"] = "partial"
    elif defect == "publication_only":
        value["base"]["status"] = "completed"
    elif defect == "wrong_cell":
        value["base"]["cell"] = 9
    elif defect == "model":
        value["identity"]["binding"]["model_sha256"] = "0"*64
    elif defect == "prepare":
        value["identity"]["binding"]["prepare_sha256"] = "0"*64
    elif defect == "missing_encode":
        value["identity"]["encodes"].pop()
    elif defect == "reverse":
        value["identity"]["encodes"].reverse()
    elif defect == "failed_encode":
        value["identity"]["encodes"][1]["outcome"] = "failure"
    elif defect == "pending":
        value["identity"]["pending"] = {}
    elif defect == "missing_binding":
        value["binding_chunks_complete"] = False
    else:
        value["scientific_admission"] = True
    with pytest.raises(ValueError):
        m.check_inspection(8, value)


def test_source_drift_does_not_become_multitask_success(monkeypatch):
    def inspect(index, spec):
        if index == 8:
            raise ValueError("loaded source drift")
        return m.check_inspection(index, good(index))
    monkeypatch.setattr(m, "inspect_cell", inspect)
    result = m.build()
    assert result["joined_cells"] == [0, 1, 9]
    assert result["rows"][8]["state"] == "identity_join_failed"
    assert "drift" in result["rows"][8]["error"]
    assert not result["four_cell_saved_join_complete"]


def test_reused_saved_inspector_receives_fixed_external_pins(monkeypatch):
    calls = []
    def inspect(*args):
        calls.append(args)
        return deepcopy(good(8))
    monkeypatch.setattr(m.host, "inspect", inspect)
    result = m.inspect_cell(8, m.SPECS[8])
    assert result["source_guard_revalidated"] and calls[0][1:3] == m.SPECS[8][1:3]
    assert calls[0][4] == m.SPECS[8][4]


def test_cli_fails_closed_on_incomplete_join(monkeypatch, capsys):
    monkeypatch.setattr(m.sys, "argv", ["saved-audit"])
    monkeypatch.setattr(m, "build", lambda: {"four_cell_saved_join_complete": False})
    assert m.main() == 1
    assert '"four_cell_saved_join_complete": false' in capsys.readouterr().out


def test_denominator_names_cannot_drift(monkeypatch):
    rows = deepcopy(m.resolver.roster())
    rows[2]["cell_id"] = rows[1]["cell_id"]
    monkeypatch.setattr(m.resolver, "roster", lambda: rows)
    with pytest.raises(ValueError, match="denominator"):
        m.build()
