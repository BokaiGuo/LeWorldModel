from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import codex_p1_common_structured_cp10_host_20261101_exclusive as m


def test_target_binding_precedes_protocol_and_runtime_rebinding():
    m.configure()
    prepared = m.host.build()
    identity = prepared["cp10_identity"]
    assert identity["goal_pixel_sha256"] == "a571f9bf915090927ce3a868e283b75983d54f475035ab9f8a3e87f473fa1218"
    assert identity["goal_latent_sha256"] == "9c8f39b40b6e4d4124881ea635802310e45892f07a84f1a69f9ed24c9bfe5124"
    assert identity["root_sha256"] == "84e0afc040e5cb3e5067e7e2965f3ba483438c850116577e04f39d3b91ca3757"
    base = prepared["common_node_protocol"]["base_protocol"]
    assert base["identity"]["goal"] == identity["goal_latent_sha256"]
    assert base["identity"]["root"] == identity["root_sha256"]
    runtime_copy = deepcopy(prepared)
    references = m.host.p3.source_data(prepared["source"])
    assert m.cp10.rebind(runtime_copy, references) == identity
    assert prepared["source"]["tasks"][0]["binding"]["root_sha256"] == identity["root_sha256"]
    assert prepared["common_node_protocol"]["caps"]["replay_calls"] == 193


def test_cem_report_drift_fails_closed(tmp_path, monkeypatch):
    bogus = tmp_path / "report.json"
    bogus.write_text(json.dumps({"cp10_goal_binding": {}}))
    monkeypatch.setattr(m, "CEM_REPORT", bogus)
    with pytest.raises(ValueError, match="CEM report SHA drift"):
        m.checked_target({})
