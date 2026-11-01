from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import p5_forward_launch_readiness_20261101 as m


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(m.supervisor, "RESULTS", tmp_path)
    monkeypatch.setattr(m.supervisor, "paths", lambda suffix: (tmp_path / ("sup_" + suffix),
                                                               tmp_path / ("run_" + suffix)))
    monkeypatch.setattr(m.supervisor, "preflight", lambda: ({}, {}))
    monkeypatch.setattr(m.host, "inspect", lambda floor: {"blockers": ["gpu_unverified"], "floor": floor})
    return tmp_path


def run():
    return m.check("NEW", 200, 40, 50, 80, 20)


def test_read_only_and_always_blocked(isolated):
    result = run()
    assert result["status"] == "blocked_not_authorization"
    assert result["errors"] == ["gpu_unverified"] or "user_manager_bus_or_uid_unavailable" in result["errors"]
    assert result["gpu_queried"] is False and result["cuda_initialized"] is False
    assert not list(isolated.iterdir())


@pytest.mark.parametrize("which", ["sup_NEW", "run_NEW",
    m.supervisor.NAME + "_accept_NEW.json", m.host.NAME + "_NEW.json"])
def test_existing_or_dangling_symlink_blocks_without_overwrite(isolated, which):
    (isolated / which).symlink_to(isolated / "missing")
    assert "output_path_already_exists" in run()["errors"]
    assert (isolated / which).is_symlink()


@pytest.mark.parametrize("runner,hard,service,overhead", [(50, 50, 80, 20), (40, 50, 69, 20),
                                                         (40, 50, 80, 0), (3601, 4000, 5000, 20)])
def test_deadline_hierarchy(isolated, runner, hard, service, overhead):
    assert "deadline_hierarchy_or_overhead_invalid" in m.check(
        "NEW", 200, runner, hard, service, overhead)["errors"]


def test_source_failure_is_reported(isolated, monkeypatch):
    def fail():
        raise ValueError("changed")
    monkeypatch.setattr(m.supervisor, "preflight", fail)
    assert "probe_contract_or_source_preflight_failed" in run()["errors"]


def test_supervisor_existing_directory_refuses_before_gpu_query(isolated, monkeypatch):
    sdir = isolated / "sup_NEW"
    sdir.mkdir()
    marker = sdir / "preserved"
    marker.write_bytes(b"old evidence")
    monkeypatch.setattr(m.supervisor, "static_binding", lambda args: ({}, {}, 0, {}))
    monkeypatch.setattr(m.supervisor.sup, "query_gpu", lambda *args: pytest.fail("GPU queried"))
    args = SimpleNamespace(authorization=m.supervisor.AUTH,
        gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc", run_suffix="NEW",
        min_free_mib=100, min_total_mib=200, hard_seconds=50, runner_seconds=40,
        max_age_seconds=300, watchdog_max_seconds=80)
    with pytest.raises(FileExistsError):
        m.supervisor.launch(args)
    assert marker.read_bytes() == b"old evidence"


@pytest.mark.parametrize("suffix", ["../x", "", "x/y", "x" * 57])
def test_rejects_unsafe_suffix(isolated, suffix):
    with pytest.raises(ValueError):
        m.check(suffix, 200, 40, 50, 80, 20)
