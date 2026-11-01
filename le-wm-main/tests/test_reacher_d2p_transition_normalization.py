"""Metadata admission tests only; no CPU neural execution."""
from types import SimpleNamespace
import hashlib
from pathlib import Path

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_transition_normalization import digest, select_training_rows


def test_selects_only_training_and_excludes_known_evaluation_episodes():
    base = SimpleNamespace(clip_indices=[(ep, start) for ep in range(15) for start in range(3)])
    groups = select_training_rows(base, {0, 2})
    assert list(groups) == [1, 3, 4, 5]
    assert all(len(v) == 2 for v in groups.values())
    assert all(base.clip_indices[i][0] == ep for ep, group in groups.items() for i in group)


def test_too_few_admitted_episodes_fails_closed():
    with pytest.raises(ValueError, match="four admitted"):
        select_training_rows(SimpleNamespace(clip_indices=[(ep, t) for ep in (8, 9, 10, 11, 12) for t in range(2)]), set())


def test_source_hash_uses_bounded_reads(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    payload = b"source" * 400000
    source.write_bytes(payload)

    def forbidden_read_bytes(*args, **kwargs):
        raise AssertionError("unbounded source read forbidden")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    assert digest(source) == hashlib.sha256(payload).hexdigest()
