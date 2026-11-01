from copy import deepcopy

import pytest

from experiments.spectral_reachability import e3_cp10_dls_sourcefix_20261031 as fix


def test_only_structured_source_changes_and_preflight_runs(monkeypatch):
    contract = {"arms": {name: {"initializer_sha256": "a" * 64}
                         for name in ("structured", "native", "random", "immediate_progress")}}
    previous = deepcopy(contract)
    seen = []
    monkeypatch.setattr(fix.control.spec, "digest", lambda value: "digest")
    monkeypatch.setattr(fix.control.spec, "preflight", lambda value, digest: seen.append((value, digest)))
    result = fix.bind_initializer(contract, "b" * 64)
    assert result["arms"]["structured"]["initializer_sha256"] == "b" * 64
    for arm in ("native", "random", "immediate_progress"):
        assert result["arms"][arm] == previous["arms"][arm]
    assert seen == [(result, "digest")]


def test_invalid_source_hash_rejected_before_mutation():
    contract = {"arms": {"structured": {"initializer_sha256": "a" * 64}}}
    previous = deepcopy(contract)
    with pytest.raises(ValueError):
        fix.bind_initializer(contract, "not-a-hash")
    assert contract == previous
