"""CPU-only launcher and consumer guards; no CUDA or real-result assertions."""
from copy import deepcopy

import pytest

from experiments.spectral_reachability import p3_live_pair_host_20261102_codexexclusive as host
from experiments.spectral_reachability import p3_live_pair_saved_20261102_codexexclusive as saved


def test_saved_empty_and_unreturned_events_fail_closed():
    with pytest.raises(ValueError):
        saved.verify_events([], [1, 2], [[{}] * 20, [{}] * 20])
    events = [{"kind": "resource_attempt", "scope": [20, "signal", "signal", 0],
               "entry": {"index": 0, "scope": [20, "signal", "signal", 0],
                         "resource": "predict", "completed": False}}]
    with pytest.raises(ValueError, match="incomplete"):
        saved.verify_events(events, [1, 2], [[{}] * 20, [{}] * 20])


def test_saved_mutated_resource_return_is_rejected():
    attempt = {"index": 0, "scope": [20, "signal", "signal", 0],
               "resource": "predict", "completed": False, "started_seconds": 1}
    returned = dict(attempt, completed=True, ended_seconds=2, resource="svd")
    events = [{"kind": "resource_attempt", "scope": attempt["scope"], "entry": attempt},
              {"kind": "resource_return", "scope": attempt["scope"], "entry": returned}]
    with pytest.raises(ValueError, match="resource return"):
        saved.verify_events(events, [1, 2], [[{}] * 20, [{}] * 20])


def test_output_paths_are_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(host, "ROOT", tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    path = results / (host.NAME + "_PREP_01.json")
    assert host.output(path, "_PREP_") == path
    path.write_text("{}")
    with pytest.raises(ValueError, match="exclusive"):
        host.output(path, "_PREP_")


def test_prepare_is_bound_to_existing_task_and_bank(monkeypatch):
    plan = {"task_sha256": ["wrong", "wrong"], "bank_sha256": ["wrong", "wrong"]}
    monkeypatch.setattr(host.host, "material", lambda: (deepcopy(plan), [1, 2], [[], []]))
    monkeypatch.setattr(host.live.kernel, "banks", lambda _: [[1], [2]])
    with pytest.raises(ValueError, match="task/bank"):
        host.build()


def test_run_needs_explicit_authority_before_loading():
    with pytest.raises(ValueError, match="authority"):
        host.run("ignored", "ignored", "ignored", "wrong")
