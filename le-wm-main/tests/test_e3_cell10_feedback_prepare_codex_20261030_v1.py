from copy import deepcopy

import pytest

from experiments.spectral_reachability import e3_cell10_feedback_prepare_codex_20261030_v1 as m


def sample():
    prior = dict(schema="fixed", cell=0, resolver={"path": "resolver", "sha256": "x"},
                 control5={"prepare": {"path": "prepare", "sha256": "y"}},
                 additional_sources={"/a": "a", "/b": "b"})
    fresh = dict(schema="fixed", cell=10, resolver=deepcopy(prior["resolver"]),
                 control5=m.references(prior), additional_sources={"/a": "a"})
    return prior, fresh


def test_cell10_references_and_no_mutation():
    prior, fresh = sample()
    old = deepcopy(prior)
    value = m.combine(prior, fresh)
    assert value["cell"] == 10 and value["control5"] == fresh["control5"]
    assert value["additional_sources"] == prior["additional_sources"] and prior == old
    assert value["control5"]["report"]["path"].endswith("_cell10_MAIN01/report.json")


@pytest.mark.parametrize("defect", ["cell", "base_cell", "schema", "resolver", "prepare", "receipt",
                                    "source", "extra"])
def test_reject_wrong_upstream_or_source(defect):
    prior, fresh = sample()
    if defect == "cell":
        fresh["cell"] = 11
    elif defect == "base_cell":
        prior["cell"] = 1
    elif defect == "schema":
        fresh["schema"] = "other"
    elif defect == "resolver":
        fresh["resolver"]["sha256"] = "changed"
    elif defect == "prepare":
        fresh["control5"]["prepare"]["sha256"] = "changed"
    elif defect == "receipt":
        fresh["control5"]["receipt"]["sha256"] = "changed"
    elif defect == "source":
        fresh["additional_sources"]["/a"] = "changed"
    else:
        fresh["additional_sources"]["/other"] = "new"
    with pytest.raises(ValueError):
        m.combine(prior, fresh)


def test_exclusive_output_rejects_existing_or_foreign_path(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    existing = tmp_path / "results" / (m.NAME + "_existing.json")
    existing.write_bytes(b"original")
    with pytest.raises(ValueError, match="exclusive"):
        m.prepare(existing)
    with pytest.raises(ValueError, match="exclusive"):
        m.prepare(tmp_path / "wrong.json")
    assert existing.read_bytes() == b"original"


def test_changed_roster_stops_before_source_loading(monkeypatch):
    roster = m.resolver.roster()
    roster[10]["cell_id"] = "wrong"
    monkeypatch.setattr(m.resolver, "roster", lambda: roster)
    monkeypatch.setattr(m.host, "authority", lambda *a: pytest.fail("should not load"))
    with pytest.raises(ValueError, match="fixed cell10"):
        m.build()


@pytest.mark.parametrize("step", [5, 4])
def test_verify_connects_native_saved_setup(monkeypatch, step):
    old = m.host.launch.episode.old
    body = {"horizon_packets": 3, "domain": {"name": "Dxy"}}
    calls = []
    monkeypatch.setattr(m, "build", lambda: {"authority": "fixed"})
    monkeypatch.setattr(m.host.launch.host, "read", lambda *args: {"authority": "fixed"})
    monkeypatch.setattr(m.host.launch.host, "sha", lambda path: m.BASE_SHA if path == m.BASE else "sha")
    monkeypatch.setattr(m.host, "authority", lambda *args: {
        "cell": 10, "args": ("joined",), "resolver": {"path": "resolver", "sha256": "pin"}})
    monkeypatch.setattr(old.authority, "load", lambda *args: {"cell": 10, "body": body})
    def resolve(path, sha, cell, actual_body):
        calls.append((path, sha, cell, actual_body))
        return {"expected_cp5": {"step": step}}
    monkeypatch.setattr(old.setup_authority, "resolve", resolve)
    if step == 5:
        assert m.verify("authority", "sha")["native_saved_setup_resolved"] is True
    else:
        with pytest.raises(ValueError, match="native setup"):
            m.verify("authority", "sha")
    assert calls == [("resolver", "pin", 10, body)]
