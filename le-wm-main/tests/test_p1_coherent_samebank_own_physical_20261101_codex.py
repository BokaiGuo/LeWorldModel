"""CPU-only selection and fail-closed tests; no physical or neural execution."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_coherent_samebank_own_physical_20261101_codex as module


def fixture():
    bank = np.broadcast_to(np.arange(1, 9, dtype=np.float32)[:, None, None, None] / 10,
                           (8, 4, 5, 2)).copy()
    scores = np.ones((8, 8), dtype=float)
    np.fill_diagonal(scores, 0)
    selection = [1.] * 9
    selection[2] = 0.
    chosen = bank[2, :3].tolist()
    events = [
        dict(kind="optimized_candidates_before_exchange", payload=dict(raw=bank.tolist())),
        dict(kind="suffix_exchange", payload=dict(scores=scores.ravel().tolist(), tail_indices=list(range(8)))),
        dict(kind="selected_full_plan", payload=dict(index=2, scores=selection, raw=chosen)),
    ]
    row = dict(scores=selection, selected_index=2, selected_suffix=chosen)
    return events, row


def test_selection_reconstructs_exchange_and_rejects_tampering():
    events, row = fixture()
    assert np.asarray(module.validate_selection(events, row)).shape == (3, 5, 2)
    corrupt = deepcopy(events)
    corrupt[1]["payload"]["tail_indices"][2] = 3
    with pytest.raises(ValueError, match="exchange argmin"):
        module.validate_selection(corrupt, row)
    corrupt = deepcopy(row)
    corrupt["selected_suffix"][0][0][0] += 0.1
    with pytest.raises(ValueError, match="selected suffix"):
        module.validate_selection(events, corrupt)
    with pytest.raises(ValueError, match="unique selected_full_plan"):
        module.validate_selection(events + [events[-1]], row)


def test_hold_wins_exact_selection_tie():
    events, row = fixture()
    scores = [0.] * 9
    chosen = np.zeros((3, 5, 2), np.float32).tolist()
    events[2]["payload"] = dict(index=8, scores=scores, raw=chosen)
    row = dict(scores=scores, selected_index=8, selected_suffix=chosen)
    assert not module.validate_selection(events, row).any()


def test_report_hash_checked_before_other_inputs(tmp_path):
    (tmp_path / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="report SHA"):
        module.collect(tmp_path, "0" * 64)
