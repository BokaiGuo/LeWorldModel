import json

import pytest

from experiments.spectral_reachability import pc_atlas_depth3_root_saved_audit_20261103 as audit


def test_saved_program_replay_and_tampered_selection():
    report = json.loads(audit.RESULT.read_text())
    assert audit.verify(report)["replayed_programs"] == 288
    report["rows"][0]["selected_index"] = (report["rows"][0]["selected_index"] + 1) % 4
    with pytest.raises(ValueError, match="selected program"):
        audit.verify(report)


def test_tampered_program_action():
    report = json.loads(audit.RESULT.read_text())
    report["rows"][0]["programs"][0]["actions"][0][0] = 0.1
    with pytest.raises(ValueError, match="root action binding"):
        audit.verify(report)
