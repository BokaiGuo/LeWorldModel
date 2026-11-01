import json

import pytest

from experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation import bind_exclusions


def report(tmp_path, *, passed=True):
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"status": "completed", "gates": {"ok": passed}, "rows": [{"seed": 10, "source_episode": 5}],
        "sealed_protocol": {"excluded_source_episodes": [1, 2]}}))
    return path


def test_exclusions_include_transitive_prior_episodes(tmp_path):
    excluded, bindings = bind_exclusions([report(tmp_path)], [20])
    assert excluded == {1, 2, 5}
    assert len(bindings) == 1 and len(bindings[0]["sha256"]) == 64


def test_no_overlap_of_source_seeds(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        bind_exclusions([report(tmp_path)], [10])


def test_failed_reference_cannot_admit_new_confirmation(tmp_path):
    with pytest.raises(ValueError, match="integrity"):
        bind_exclusions([report(tmp_path, passed=False)], [20])
