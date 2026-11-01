"""RAM/saved-only tests for the JSON-safe strong-baseline summary."""
from copy import deepcopy
import json
import os
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p2_strong_baseline_saved_summary_20261028_v2 as m


def test_real_saved_inputs_are_sealed_and_protocols_are_separate():
    value = m.build()
    assert value["status"].startswith("saved_")
    assert set(value["local"]) == {"kinodynamic", "no_composition", "sampling"}
    assert set(value["comparisons"]) == {"kinodynamic__vs__no_composition", "kinodynamic__vs__sampling", "no_composition__vs__sampling"}
    assert all(item["status"] == "protocol_mismatch_no_direct_comparison" for item in value["comparisons"].values())
    assert value["author"]["status"] == "unavailable"
    assert value["P1_admission"] is False


def test_real_denominators_and_local_values_are_not_paper_values():
    value = m.build()
    assert value["denominators"] == {"kinodynamic": 2, "no_composition": 2, "sampling": 4}
    assert all("local_values" in value["local"][name] for name in value["local"])
    assert value["author"]["values"] is None


@pytest.mark.parametrize("damage", ["status", "schema", "protocol", "errors", "admission", "row"])
def test_saved_summary_mutation_rejected(tmp_path, damage):
    value = m.build()
    if damage == "status":
        value["status"] = "bad"
    elif damage == "schema":
        value["schema"] = "other"
    elif damage == "protocol":
        value["local"]["sampling"]["protocol"]["history"] = "fake"
    elif damage == "errors":
        value["local"]["kinodynamic"]["raw_keys"].append("errors")
    elif damage == "admission":
        value["P1_admission"] = True
    else:
        value["local"]["no_composition"]["local_values"].pop()
    path = tmp_path / "summary.json"
    m.write(path, value)
    with pytest.raises(ValueError):
        m.verify(path, m.sha(path))


def test_author_values_are_separate_and_optional(tmp_path):
    author = tmp_path / "author.json"
    author.write_text(json.dumps({"kinodynamic": {"reported": 0.1}}))
    value = m.build(author)
    assert value["author"]["values"]["kinodynamic"]["reported"] == 0.1
    assert all("author" not in value["local"][name] for name in value["local"])


def test_protocol_mismatch_never_returns_delta():
    value = m.compare({"protocol": {"horizon": 2}}, {"protocol": {"horizon": 10}})
    assert value["status"].endswith("no_direct_comparison")
    assert value["delta"] is None


def test_report_errors_and_unknown_values_fail_closed(tmp_path, monkeypatch):
    report = deepcopy(m.load_local()["sampling"]["report"])
    report["errors"] = ["unknown"]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    monkeypatch.setitem(m.LOCAL, "sampling", (path, m.sha(path)))
    with pytest.raises(ValueError):
        m.load_local()


def test_cold_import_does_not_import_neural_runtime():
    code = """import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','gym','gymnasium','pymunk','stable_worldmodel'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from experiments.spectral_reachability import agent_p2_strong_baseline_saved_summary_20261028_v2 as module
assert module.compare({'protocol':{}},{'protocol':{}})['status'].endswith('allowed')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=m.ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
