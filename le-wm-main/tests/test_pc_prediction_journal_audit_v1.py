import json

import numpy as np
import pytest

from experiments.pc_prediction_journal_audit_v1 import audit_journal
from experiments.pc_prediction_journal_v1 import PredictionJournal


def test_external_audit_preserves_success_and_injected_failure(tmp_path):
    def predictor(actions):
        if actions[0, 0, 0] > 0:
            raise RuntimeError("injected")
        return actions.sum(axis=1), np.zeros(len(actions))

    journal = PredictionJournal(tmp_path / "calls", predictor, action_dim=2, packet_length=5, token_cap=5)
    journal(np.zeros((2, 5, 2), np.float32))
    with pytest.raises(RuntimeError, match="injected"):
        journal(np.ones((1, 5, 2), np.float32))
    report = audit_journal(journal.directory)
    assert report["tokens"] == 3
    assert [item["status"] for item in report["calls"]] == ["returned", "failed"]


def test_detects_incomplete_and_corrupted_attempts(tmp_path):
    journal = PredictionJournal(tmp_path / "calls", lambda a: (a.sum(axis=1), np.zeros(len(a))),
                                action_dim=2, packet_length=5, token_cap=5)
    journal(np.zeros((1, 5, 2), np.float32))
    returned = journal.directory / "000000_returned.json"
    returned.unlink()
    assert audit_journal(journal.directory)["calls"][0]["status"] == "incomplete"
    attempt_path = journal.directory / "000000_attempt.json"
    attempt = json.loads(attempt_path.read_text())
    attempt["cost"] = 0
    attempt_path.write_text(json.dumps(attempt))
    with pytest.raises(ValueError, match="cost"):
        audit_journal(journal.directory)
