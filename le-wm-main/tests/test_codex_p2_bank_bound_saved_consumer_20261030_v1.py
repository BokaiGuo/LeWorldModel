import json

import pytest

from experiments.spectral_reachability import codex_p2_bank_bound_saved_consumer_20261030_v1 as c
from test_p2_composition_predict_frontier_20261027_v1 import base


def test_saved_bank_and_plan_binding_and_tamper(tmp_path, monkeypatch):
    ready = {"root_sha256": c.frontier.ROOTS[0], "root_index": 0, "root": {}, "action_domain": {}}
    events = []
    def provenance(protocol, prepared, ready, bank, arm):
        return {"candidate_bank_sha256": c.frontier.digest(
            {"bank": bank, "arm": arm, "contract": c.frontier.contract()})}
    monkeypatch.setattr(c.frontier, "provenance", provenance)
    def run(arm, bank, prov):
        raw = [[0., 0.] for _ in range(10)]
        return {"arm": arm, "complete": True, "actions": raw,
                "candidate_id": arm, "witness": {"receipt": {"actions": raw}}}
    paired = c.bound.bind_then_compare(
        directory=tmp_path, ready=ready, compile_bank=base,
        make_provenance=lambda bank, arm: provenance({}, {}, ready, bank, arm),
        run_arm=run, verify_arm=lambda *a: None, emit=events.append)
    monkeypatch.setattr(c.frontier, "validate_actual", lambda entries: None)
    monkeypatch.setattr(c.frontier, "validate_operations", lambda ops, results: None)
    monkeypatch.setattr(c.frontier, "validate_result", lambda result, **kw: result)
    source = tmp_path / "source.py"
    source.write_text("source")
    report = {"schema": "codex_p2_bank_bound_frontier_v1", "status": "exploratory_complete",
              "fixture": False, "binding": {"root_sha256": ready["root_sha256"],
              "protocol_sha256": c.frontier.digest({}), "prepared_sha256": c.frontier.digest({})},
              "bank_binding_sha256": paired["binding_sha256"], "bank_file_sha256": paired["bank_file_sha256"],
              "provenances": paired["provenances"], "results": paired["results"],
              "P1_admission": False, "P2_admission": False, "equal_total_compute": False,
              "physical_authorized": False}
    (tmp_path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    report["events_sha256"] = c.frontier.file_sha(tmp_path / "events.jsonl")
    c.bound._write(tmp_path / "report.json", report)
    kwargs = dict(directory=tmp_path, expected_report_sha256=c.frontier.file_sha(tmp_path / "report.json"),
                  ready=ready, compiled={}, protocol={}, prepared={},
                  source_bindings={str(source): c.frontier.file_sha(source)})
    result = c.consume(**kwargs)
    assert len(result["plans"]) == 2 and result["physical_authorized"] is False
    (tmp_path / "bank.json").write_text("{}")
    with pytest.raises(ValueError, match="bank"):
        c.consume(**kwargs)
