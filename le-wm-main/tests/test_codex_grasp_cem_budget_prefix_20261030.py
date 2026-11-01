import hashlib
import math
import sys

import pytest

from experiments.spectral_reachability import codex_grasp_cem_budget_prefix_20261030 as m


def test_real_roots_and_counts():
    out = m.analyze(m.inputs())
    assert [x["root"] for x in out["results"]] == [0, 1]
    assert not out["fair_ranking"]
    for row in out["results"]:
        assert row["remaining_predictor_samples_to_grasp"] == 102
        assert all(b["below"] + b["equal"] + b["above"] == 300 for b in row["cem_batches"])
        assert row["grasp_resources"]["total_FLOPs"] is None


def test_exact_ties_and_invalid_values():
    scores = [1.0] * 300
    assert m.census(scores, 1)["equal"] == 300
    scores[0] = math.nan
    with pytest.raises(ValueError):
        m.census(scores, 1)
    with pytest.raises(ValueError):
        m.census([1] * 299, 1)


@pytest.mark.parametrize("mutate", [
    lambda d: d["prepare"]["slots"][0]["goal_emb"].__setitem__(0, 999),
    lambda d: d["cem"]["rows"][0].__setitem__("task_id", "wrong"),
    lambda d: d["cem"]["rows"][0]["planning_scores"][1].__setitem__("iteration", 0),
    lambda d: d["grasp"]["resources"][2]["completed"].__setitem__("predict_samples", 1901),
])
def test_reject_wrong_binding_or_account(mutate):
    data = m.inputs()
    mutate(data)
    with pytest.raises(ValueError):
        m.analyze(data)


def test_sha_mismatch(tmp_path, monkeypatch):
    (tmp_path / "input.json").write_text("{}")
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "SOURCES", {"x": ("input.json", "0" * 64)})
    with pytest.raises(ValueError, match="SHA mismatch"):
        m.inputs()


def test_exclusive_output_and_reconstruction(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "inputs", lambda: {})
    monkeypatch.setattr(m, "analyze", lambda _: {"value": 1})
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
