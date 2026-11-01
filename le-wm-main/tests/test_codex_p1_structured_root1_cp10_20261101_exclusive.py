from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import codex_p1_structured_root1_cp10_20261101_exclusive as m


def test_pins_reject_drift(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_bytes(b"changed")
    monkeypatch.setattr(m, "PINS", {path: "0" * 64})
    with pytest.raises(ValueError, match="pinned source drift"):
        m.check_pins()


def test_closure_rejects_changed_file(tmp_path):
    path = tmp_path / "changed"
    path.write_bytes(b"actual")
    with pytest.raises(ValueError, match="source closure drift"):
        m.check_closure({"sources": {str(path): "0" * 64}})


def test_real_binding_and_index_isolation_in_fresh_process():
    code = """
from copy import deepcopy
from experiments.spectral_reachability import codex_p1_structured_root1_cp10_20261101_exclusive as m
m.configure()
p = m.host.build()
d = m.host.p3.source_data(p["source"])
c = deepcopy(p)
c["source"]["tasks"][1] = deepcopy(p["original_root1_task"])
identity, _ = m.bind(c, d)
assert identity == p["cp10_identity"]
assert p["source"]["tasks"][0] == p["source"]["tasks"][1] == c["source"]["tasks"][1]
assert p["common_node_protocol"]["base_protocol"]["identity"]["root"] == identity["root_sha256"]
assert p["common_node_protocol"]["base_protocol"]["identity"]["goal"] == identity["goal_latent_sha256"]
assert m.host.protocol()["root_slot"] == 1
rows = m.host.roster()
assert len(rows) == 1 and rows[0]["source_slot"] == 1 and rows[0]["arm"] == "structured"
assert rows[0]["episode_id"] == "s1_structured"
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr


def test_report_pin_is_specific():
    assert m.REPORT.name == "report.json"
    assert Path(m.REPORT).parent.name.endswith("root1_RUN_MAIN02")
    assert len(m.PINS[m.REPORT]) == 64
