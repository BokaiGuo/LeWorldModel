from pathlib import Path

from experiments.spectral_reachability import p2_b0_isolated_physical_consume_20261102_exclusive as m


def test_command_is_readonly_and_calls_only_saved_audit():
    cmd = m.command(Path("/tmp/history.py"))
    assert cmd.count("--ro-bind") == 2
    assert "--bind" not in cmd
    assert "CUDA_VISIBLE_DEVICES=" in cmd
    assert "verify_contract" in cmd[-1] and "audit_output" in cmd[-1]
    assert "physical.run" not in cmd[-1]
