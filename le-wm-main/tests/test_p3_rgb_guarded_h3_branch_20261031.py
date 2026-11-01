from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p3_rgb_guarded_h3_branch_20261031 as branch
from experiments.spectral_reachability import p3_rgb_guarded_h3_batch_20261031 as batch


@pytest.mark.parametrize("admitted", [True, False])
def test_guard_records_exact_decision_and_pixels(monkeypatch, admitted):
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    decision = {"admitted": admitted, "reason": "fixture", "bounds_xyxy": None}
    monkeypatch.setattr(branch.edge, "preflight", lambda image: decision)
    events = []
    store = SimpleNamespace(emit=lambda kind, **payload: events.append((kind, payload)))
    row = {}
    assert branch.guard_at_15(frame, row, SimpleNamespace(where={"branch": 2}), store) is admitted
    assert row["rgb_edge_preflight"] == decision
    assert events == [("rgb_edge_preflight", {"branch": 2, "step": 15,
                         "pixel_sha256": branch.kernel.moving.image_sha(frame), "decision": decision})]


def test_guard_does_not_dispatch_if_journal_fails(monkeypatch):
    monkeypatch.setattr(branch.edge, "preflight", lambda image: {"admitted": True})
    def fail(*args, **kwargs):
        raise OSError("journal failed")
    with pytest.raises(OSError, match="journal failed"):
        branch.guard_at_15(np.zeros((224, 224, 3), dtype=np.uint8), {},
                           SimpleNamespace(where={}), SimpleNamespace(emit=fail))


def test_batch_rejects_fake_abstention_and_unapproved_continuation():
    row = {"status": "rgb_preflight_abstain", "own_returned": 15, "own_attempted": 15,
           "endpoint_scope": "abstained_at_15", "rgb_edge_preflight": {"admitted": False},
           "terminated": False, "truncated": False}
    batch.validate_guard_status(row)
    with pytest.raises(ValueError, match="abstention identity"):
        batch.validate_guard_status({**row, "own_returned": 16})
    with pytest.raises(ValueError, match="continuation requires RGB admission"):
        batch.validate_guard_status({**row, "status": "horizon_completed", "own_returned": 20})
    batch.validate_guard_status({**row, "status": "horizon_completed", "own_returned": 20,
                                 "rgb_edge_preflight": {"admitted": True}})
