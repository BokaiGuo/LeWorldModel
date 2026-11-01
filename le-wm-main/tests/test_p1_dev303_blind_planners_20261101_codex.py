"""Strict development export adapter checks; no neural or physical execution."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_dev303_blind_planners_20261101_codex as m


@pytest.fixture
def export():
    path = m.base.ROOT / "results/task_p1_dev303_goal_blind_20261101_codex.json.gz"
    if not path.is_file():
        pytest.skip("sealed dev303 development export unavailable")
    assert m.base.sha(path) == m.TASK_SHA
    return m.base.read(path)


def test_exact_development_export_is_accepted(export):
    m.validate_export(export)
    with m.patched():
        actual = m.base.load_export(
            m.base.ROOT / "results/task_p1_dev303_goal_blind_20261101_codex.json.gz", m.TASK_SHA
        )
    assert actual["task_id"] == export["task_id"]


@pytest.mark.parametrize("field,value", [
    ("development_seed", 301000004),
    ("development_seed", True),
    ("global_freshness", "certified"),
    ("P1_admission", True),
    ("blind_source_sha256", "0" * 64),
    ("unexpected", 1),
])
def test_rejects_identity_and_scope_drift(export, field, value):
    changed = deepcopy(export)
    changed[field] = value
    with pytest.raises(ValueError):
        m.validate_export(changed)


def test_rejects_manifest_or_encoding_prepare_drift(export):
    for key in ("whitelist_sha", "prepare_sha256"):
        changed = deepcopy(export)
        changed["task_contract"][key] = "0" * 64
        with pytest.raises(ValueError):
            m.validate_export(changed)


def test_patching_restores_original_validator():
    original = m.base.validate_export_envelope
    with m.patched():
        assert m.base.validate_export_envelope is m.validate_export
    assert m.base.validate_export_envelope is original


def test_real_prepare_reconstructs_and_binds_adapter(export):
    task = m.base.ROOT / "results/task_p1_dev303_goal_blind_20261101_codex.json.gz"
    with m.patched():
        contract = m.base.build([str(task)], [m.TASK_SHA])
    assert contract["classification"] == "development_only_not_independent"
    assert contract["contract"]["P1_admission"] is False
    assert contract["bindings_sha256"][str(Path(m.__file__).resolve())] == m.base.sha(m.__file__)
    assert contract["slots"][0]["task_id"] == export["task_id"]
