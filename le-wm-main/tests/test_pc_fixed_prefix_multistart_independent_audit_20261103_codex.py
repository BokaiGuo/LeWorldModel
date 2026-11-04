import json

import pytest

from experiments import pc_fixed_prefix_multistart_feedback_independent_audit_20261103_codex as audit


def test_replay_does_not_overwrite_sealed_audit():
    before = audit.exp.AUDIT.read_bytes()
    audit.verify()
    assert audit.exp.AUDIT.read_bytes() == before


def test_detects_changed_audit_and_restores_base_globals(tmp_path, monkeypatch):
    base = audit.exp.base
    original = base.PLANS, base.AUDIT, base.dependencies
    fake = tmp_path / "audit.json"
    altered = json.loads(audit.exp.AUDIT.read_text())
    altered["rows"][0]["tokens"][0] += 1
    fake.write_text(json.dumps(altered))
    monkeypatch.setattr(audit.exp, "AUDIT", fake)
    # exp.dependencies includes the original audit path, so preserve the
    # verified dependency snapshot while redirecting only the comparison.
    hashes = json.loads(audit.exp.PLANS.read_text())["dependencies"]
    monkeypatch.setattr(audit.exp, "dependencies", lambda: hashes)
    with pytest.raises(ValueError, match="differs"):
        audit.verify()
    assert (base.PLANS, base.AUDIT, base.dependencies) == original
