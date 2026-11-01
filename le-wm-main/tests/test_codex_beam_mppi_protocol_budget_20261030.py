import hashlib
import sys

import pytest

from experiments.spectral_reachability import codex_beam_mppi_protocol_budget_20261030 as m


def test_real_roster_and_phase_decomposition():
    out = m.audit(m.read_sources())
    assert len(out["rows"]) == 2
    assert out["fair_total_compute"] is False
    for row in out["rows"]:
        assert row["beam"]["schedule_search_predict"] == 140
        assert row["beam"]["schedule_replay_predict"] == 4
        assert row["mppi"]["schedule_search_predict"] == 144
        assert row["mppi"]["schedule_replay_predict"] == 0
        assert row["total_flops"] is None


@pytest.mark.parametrize("mutate", [
    lambda d: d["beam"]["rows"][0].__setitem__("task_id", "wrong"),
    lambda d: d["sampling"]["contract"].__setitem__("cap", 0),
    lambda d: d["beam"]["contract"]["schedule"].__setitem__("fresh_whole_predict", 3),
    lambda d: d["comparison"]["rows"][0]["identity"].__setitem__("task_id", "wrong"),
    lambda d: d["sampling"]["rows"][1]["resources"]["predict_API"].__setitem__("completed", 143),
])
def test_mismatch_fails_closed(mutate):
    data = m.read_sources()
    mutate(data)
    with pytest.raises(ValueError):
        m.audit(data)


def test_source_sha_mismatch(tmp_path, monkeypatch):
    (tmp_path / "source.json").write_text("{}")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "SOURCES", {"x": ("source.json", "0" * 64)})
    with pytest.raises(ValueError, match="SHA mismatch"):
        m.read_sources()


def test_exclusive_write_and_verify(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "read_sources", lambda: {})
    monkeypatch.setattr(m, "audit", lambda _: {"test": True})
    path = tmp_path / "results" / (m.NAME + "_test.json")
    monkeypatch.setattr(sys, "argv", ["audit", "build", "--output", str(path)])
    m.main()
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", ["audit", "verify", "--output", str(path), "--sha256", sha])
    m.main()
    monkeypatch.setattr(sys, "argv", ["audit", "build", "--output", str(path)])
    with pytest.raises(FileExistsError):
        m.main()
    path.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["audit", "verify", "--output", str(path), "--sha256", sha])
    with pytest.raises(ValueError):
        m.main()
