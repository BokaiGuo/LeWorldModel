import json

import pytest

from experiments.spectral_reachability import p2_replay_reserved_runtime_20261101 as runtime
from experiments.spectral_reachability import p2_replay_source_seal_20261101 as seal


def test_provenance_factory_is_bound_to_actual_bank():
    bank = {"actual": 1}
    value = runtime.resolve_provenance(
        lambda actual: {arm: {"bank": actual} for arm in runtime.base.ARMS}, bank)
    assert all(value[arm]["bank"] is bank for arm in runtime.base.ARMS)
    with pytest.raises(ValueError, match="both bank-bound"):
        runtime.resolve_provenance(lambda _: {}, bank)


def test_inventory_verification_requires_external_hash_and_detects_drift(tmp_path, monkeypatch):
    data = {"prepared": "/prepared", "prepared_sha256": "x", "files": {"a": "b"}}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(seal, "inventory", lambda *_: data.copy())
    assert seal.verify(path, seal.sha(path)) == data
    with pytest.raises(ValueError, match="external"):
        seal.verify(path, "wrong")
    monkeypatch.setattr(seal, "inventory", lambda *_: dict(data, files={"a": "changed"}))
    with pytest.raises(ValueError, match="drift"):
        seal.verify(path, seal.sha(path))
