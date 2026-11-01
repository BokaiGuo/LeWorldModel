import json

import pytest

from experiments.spectral_reachability import e3_cell11_cooperative_run_20261031_main as runner


def test_refuses_wrong_confirmation():
    with pytest.raises(ValueError, match="confirmation"):
        runner.run("not authorized")


def test_marker_exclusive_and_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    marker = tmp_path / "marker"
    runner.exclusive_write(marker, {"cell": 11})
    assert json.loads(marker.read_text()) == {"cell": 11}
    with pytest.raises(FileExistsError):
        runner.exclusive_write(marker, {"cell": 12})
    assert json.loads(marker.read_text()) == {"cell": 11}


def test_precheck_binds_exact_cell11_authority():
    command = runner.precheck_command()
    assert command[command.index("--authority-sha256") + 1] == runner.AUTH_SHA
    assert command[command.index("--closure-sha256") + 1] == runner.CLOSURE_SHA
    assert "e3_cell11" in command[command.index("--source") + 1]
