from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p2_knownroot_physical_join_20261030_v1 as m


def test_join_rejects_wrong_run_sha_before_validation(tmp_path):
    (tmp_path / "run_report.json").write_text("{}")
    with pytest.raises(ValueError, match="run SHA"):
        m.join(run_dir=tmp_path, run_sha="0" * 64, prepared_dir=tmp_path,
               prepared_sha="0" * 64, reference_dir=tmp_path)


def test_join_rejects_physical_root_census_before_consumption(tmp_path, monkeypatch):
    report = {"status": "exploratory_complete", "prepare": {},
              "physical_authorized": False, "P1_admission": False,
              "P2_admission": False, "roots": [{}, {}, {}, {}]}
    path = tmp_path / "run_report.json"
    import json
    path.write_text(json.dumps(report))
    monkeypatch.setattr(m.host, "validate_prepare", lambda *a: ({}, {}, {}))
    refs = {"root_seed": 202753010, "root_oversample_factor": 4, "packets": [], "references": {}}
    with pytest.raises(ValueError, match="physical root census"):
        m.join(run_dir=tmp_path, run_sha=m.host.frontier.file_sha(path), prepared_dir=tmp_path,
               prepared_sha="0" * 64, reference_dir=tmp_path,
               source_validator=lambda *a: refs,
               consumer=lambda **kw: pytest.fail("must not consume"))
