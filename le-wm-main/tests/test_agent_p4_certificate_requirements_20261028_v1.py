"""Saved metadata/RAM only; no root, NPZ, model, physics, or network work."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p4_certificate_requirements_20261028_v1 as m


def test_real_requirements_and_no_approval():
    value = m.build()
    assert value["denominators"] == {"historical_sources": 8, "inherited_refs": 16, "known_dispatches": 10, "unresolved_occurrences": 975}
    assert value["source_run_authorized"] is False
    assert value["certificate_created"] is False and value["approval_created"] is False
    assert value["reserved_roots_opened"] is False and value["history_certificate"] is None
    assert value["new_neural_calls"] == value["new_physics_calls"] == 0
    assert len(value["unresolved_sources"]) == 975
    assert all(item["can_be_satisfied_by_new_execution"] is False for item in value["requirements"])


def test_authority_request_is_compatible_without_reading_origin_assets():
    value = m.build()
    from experiments.spectral_reachability import agent_freshroot_authority_20261028_v1 as authority
    request = value["authority_request"]
    assert request == authority.requests(m.load_inputs()["coverage"])
    response = {"schema": authority.NAME + ".response", "request_sha256": value["authority_request_sha256"], "sources": [], "historical_scope": None}
    result = authority.validate(response, request, authority.Reader(provider=lambda _: pytest.fail("must not follow report paths")))
    assert result["missing_sources"] == list("ABCDEFGH")
    assert result["reset_blockers_cleared"] == 0


def test_all_B_launches_and_original_loaded_roles_preserved():
    rows = m.build()["historical_dispatch_requirements"]
    b = [row for row in rows if row["source_id"] == "B"]
    assert len(b) == 3 and len({row["call_id"] for row in b}) == 3
    assert all(row["attempts"] is None for row in rows)
    assert all("shell_launcher" in row["route_options"][1]["loaded_roles"] for row in b)
    assert all("p0_dependency" in row["route_options"][1]["loaded_roles"] for row in rows if row["source_id"] in "DEFGH")


def test_absent_certificate_field_is_not_called_explicit_null():
    needs = {row["id"]: row for row in m.build()["requirements"]}
    assert needs["root_manifest.history_certificate"]["field_present"] is False
    assert needs["consumer_manifest.history_certificate"]["field_present"] is False
    assert needs["coverage_approval"]["observed"]["external_file"] is None


def test_identity_and_family_gaps_not_confused_with_run_denominator():
    value = m.build()
    assert len(value["identity_inventory"]["records"]) == 618
    assert value["identity_inventory"]["summary"]["inherited_refs_without_direct_identity"] == 614
    assert value["identity_inventory"]["closure_summary"]["inherited_refs_preserved"] == 960
    families = {item["family"]: item for item in value["family_requirements"]}
    assert families["original_four"]["inventory_paths"] == []
    assert families["original_four"]["status"] == "blocked_inventory_authority_not_received"
    assert len(families["overlay"]["inventory_paths"]) == 5
    assert len(families["feedback"]["inventory_paths"]) == 1


def test_build_does_not_follow_any_embedded_path(monkeypatch):
    original = m.file_bytes
    allowed = {str(m.ROOT / relative) for relative, _ in m.PINS.values()} | {
        str(Path(m.__file__).resolve()), str(m.ROOT / "tests" / ("test_" + m.NAME + ".py")), str(m.ROOT / "innerdocs" / (m.NAME + ".md"))}
    reads = []
    def guarded(path):
        assert str(path) in allowed
        assert Path(path).suffix != ".npz"
        reads.append(str(path))
        return original(path)
    monkeypatch.setattr(m, "file_bytes", guarded)
    m.build()
    assert len(reads) == len(m.PINS) + 3


@pytest.mark.parametrize("key", list(m.PINS))
def test_each_upstream_pin_drift_rejected(monkeypatch, key):
    relative, _ = m.PINS[key]
    monkeypatch.setitem(m.PINS, key, (relative, "0" * 64))
    with pytest.raises(ValueError, match="pinned metadata drift"):
        m.build()


@pytest.mark.parametrize("field", ["source_run_authorized", "certificate_created", "reserved_roots_opened", "history_certificate", "unresolved_sources", "authority_request"])
def test_saved_request_mutations_rejected(tmp_path, field):
    value = m.build()
    value[field] = True if field.endswith(("authorized", "created", "opened")) else []
    path = tmp_path / (m.NAME + "_mutation.json")
    m.write(path, value)
    with pytest.raises(ValueError, match="reconstruction"):
        m.consume(path, m.digest(m.canonical(value)))


def test_saved_roundtrip_remains_blocked_and_exclusive(tmp_path):
    value = m.build()
    path = tmp_path / (m.NAME + "_request.json")
    m.write(path, value)
    result = m.consume(path, m.digest(m.canonical(value)))
    assert result["status"] == "blocked" and result["source_run_authorized"] is False
    with pytest.raises(FileExistsError):
        m.write(path, value)
    with pytest.raises(ValueError, match="request SHA"):
        m.consume(path, "0" * 64)


def test_duplicate_keys_nonfinite_and_foreign_path_refused(tmp_path):
    with pytest.raises(ValueError):
        m.parse(b'{"x":0,"x":1}')
    with pytest.raises(ValueError):
        m.parse(b'{"x":NaN}')
    with pytest.raises(ValueError, match="own request"):
        m.consume(tmp_path / "reserved.npz", "0" * 64)


def test_cold_import_no_neural_physics_numpy_or_audit_execution():
    code = """import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','numpy','gym','gymnasium','pymunk','stable_worldmodel'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from experiments.spectral_reachability import agent_p4_certificate_requirements_20261028_v1 as m
assert m.build()['source_run_authorized'] is False
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=m.ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_request_and_consume_roundtrip(tmp_path):
    path = tmp_path / (m.NAME + "_cli.json")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    module = "experiments.spectral_reachability." + m.NAME
    create = subprocess.run([sys.executable, "-B", "-m", module, "request", "--output", str(path)], cwd=m.ROOT, env=env, capture_output=True, text=True)
    assert create.returncode == 0, create.stderr
    receipt = m.parse(create.stdout.encode())
    validate = subprocess.run([sys.executable, "-B", "-m", module, "consume", "--request", str(path), "--sha", receipt["sha256"]], cwd=m.ROOT, env=env, capture_output=True, text=True)
    assert validate.returncode == 0, validate.stderr
    assert m.parse(validate.stdout.encode())["status"] == "blocked"
