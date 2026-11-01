import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p3_rgb_fallback_h3_branch_20261031 as branch


@pytest.mark.parametrize("admitted", [True, False])
def test_selection_uses_rgb_and_journals_before_return(monkeypatch, admitted):
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    actions = np.zeros((20, 2), dtype=np.float32)
    fallback = actions.copy()
    fallback[15:] = 0.5
    monkeypatch.setattr(branch.edge, "preflight", lambda frame: {"admitted": admitted})
    events = []
    row = {}
    store = SimpleNamespace(emit=lambda kind, **kw: events.append((kind, kw)))
    result = branch.choose_suffix(frame, row, SimpleNamespace(where={"branch": 0}),
                                  store, actions, fallback)
    expected = actions[15:] if admitted else fallback[15:]
    assert result.tobytes() == expected.tobytes()
    assert [x[0] for x in events] == ["rgb_edge_preflight", "fallback_selection"]
    assert row["chosen_suffix_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    assert row["suffix_selection"] == ("planned" if admitted else "cached_fallback")


def test_rejects_wrong_prefix_and_journal_failure(monkeypatch):
    actions = np.zeros((20, 2), dtype=np.float32)
    other = actions.copy()
    other[0, 0] = 0.1
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    ops = SimpleNamespace(where={})
    with pytest.raises(ValueError, match="prefix"):
        branch.choose_suffix(frame, {}, ops, SimpleNamespace(emit=lambda *a, **k: None),
                             actions, other)
    monkeypatch.setattr(branch.edge, "preflight", lambda frame: {"admitted": False})
    def fail(kind, **kwargs):
        if kind == "fallback_selection":
            raise OSError("journal failed")
    with pytest.raises(OSError, match="journal failed"):
        branch.choose_suffix(frame, {}, ops, SimpleNamespace(emit=fail), actions, actions)
