import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_adam_terminal_rgb_cuda_20261031 as subject


def test_full_saved_panel():
    entries, stopped, sources = subject.collect()
    assert (len(entries), len(stopped)) == (64, 8)
    assert len(sources) >= 150


def test_rejects_changed_dispatch_and_endpoint():
    raw = np.zeros((2, 5, 2), dtype="<f4")
    import hashlib
    digest = hashlib.sha256(raw.tobytes()).hexdigest()
    identity = {"index": 0, "role": "zero", "condition": "off", "root_slot": 0}
    prepared = {**identity, "raw10_float32_hex": raw.tobytes().hex(), "raw10_sha256": digest}
    consumed = {**identity, "status": "probe_completed", "completed_endpoint": [1]}
    observed = {**identity, "budgets_status": "probe_completed"}
    shard = {"status": "recorded_not_consumed"}
    rec = {"status": "probe_completed", "responses": [{"state": [1]}]}
    subject.verify_binding(0, raw, observed, shard, prepared, consumed, rec)
    altered = raw.copy()
    altered[0, 0, 0] = 1
    with pytest.raises(Exception):
        subject.verify_binding(0, altered, observed, shard, prepared, consumed, rec)
    changed = copy.deepcopy(rec)
    changed["responses"][-1]["state"] = [2]
    with pytest.raises(Exception):
        subject.verify_binding(0, raw, observed, shard, prepared, consumed, changed)
    with pytest.raises(Exception):
        subject.verify_binding(0, raw, {**observed, "role": "selected"}, shard, prepared, consumed, rec)
