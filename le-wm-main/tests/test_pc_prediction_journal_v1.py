import json

import numpy as np
import pytest

from experiments.pc_prediction_journal_v1 import PredictionJournal, digest


def test_failed_call_retains_attempt_and_previous_results(tmp_path):
    def predict(actions):
        if actions[0, 0, 0] > 0:
            raise RuntimeError("injected failure")
        return actions.sum(axis=1), np.zeros(len(actions))

    journal = PredictionJournal(tmp_path / "trace", predict, action_dim=2, packet_length=5, token_cap=10)
    journal(np.zeros((2, 5, 2), np.float32))
    actions = np.full((1, 5, 2), .5, np.float32)
    with pytest.raises(RuntimeError, match="injected failure"):
        journal(actions)
    assert (journal.calls, journal.tokens) == (2, 3)
    attempt = json.loads((journal.directory / "000001_attempt.json").read_text())
    assert attempt["cost"] == 1
    assert attempt["sha256"] == digest(journal.directory / attempt["actions"])
    with np.load(journal.directory / attempt["actions"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["actions"], actions)
    assert json.loads((journal.directory / "000001_failed.json").read_text())["type"] == "RuntimeError"
    assert (journal.directory / "000000_returned.json").exists()
    assert not (journal.directory / "000001_returned.json").exists()


def test_out_of_budget_does_not_call_predictor(tmp_path):
    seen = []
    journal = PredictionJournal(tmp_path / "trace", lambda a: seen.append(a),
                                action_dim=2, packet_length=5, token_cap=1)
    with pytest.raises(RuntimeError):
        journal(np.zeros((2, 5, 2), np.float32))
    assert not seen and journal.calls == 0 and journal.tokens == 0
