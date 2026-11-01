import copy
import json

import pytest

from experiments.spectral_reachability import p3_h3_real_context_physical_20261031 as subject


def test_fixed_batch_roster_and_action_binding():
    manifest = json.loads((subject.ROOT / "results/p3_h3_real_context_physical_20261031_prepare_RUN01.json").read_text())
    assert manifest["fixed_pairs"] == 52
    assert manifest["padding_pairs"] == 2
    assert len(manifest["pairs"]) == 54
    assert [i for i, pair in enumerate(manifest["pairs"]) if pair["padding"]] == [52, 53]
    for i in range(18):
        p = subject.prepare_batch(manifest, i)
        assert len(p["plans"]) == len(p["guards"]) == len(p["identities"]) == 6
        assert subject.batch.validate_plans(p["plans"])
    changed = copy.deepcopy(subject.prepare_batch(manifest, 0))
    changed["plans"][1]["commands"][15][0] += 0.125
    with pytest.raises(ValueError):
        subject.batch.validate_plans(changed["plans"])


def test_cp15_mismatch_rejects_before_suffix(monkeypatch, tmp_path):
    manifest = json.loads((subject.ROOT / "results/p3_h3_real_context_physical_20261031_prepare_RUN01.json").read_text())
    p = subject.prepare_batch(manifest, 0)
    dispatched = []

    class Store:
        def __init__(self, path):
            pass

        def put(self, *args):
            pass

    def fake_capture(env, obs, phase, step, row, ops, store, index):
        row["checkpoints"].append({"state": [], "snapshot": {}, "pixel_sha256": "wrong"})
        return None, None

    def fake_execute(p, data, store, factory):
        subject.batch.branch.capture(None, None, "plan", 15, {"checkpoints": []},
                                     None, store, 0)
        dispatched.append(16)

    monkeypatch.setattr(subject.host.helper, "Store", Store)
    monkeypatch.setattr(subject.batch.branch, "capture", fake_capture)
    monkeypatch.setattr(subject.batch, "execute", fake_execute)
    with pytest.raises(ValueError, match="cp15 mismatch"):
        subject.execute_guarded(p, {}, tmp_path / "run")
    assert dispatched == []
