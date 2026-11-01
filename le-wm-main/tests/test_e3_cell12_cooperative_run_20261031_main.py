import pytest

from experiments.spectral_reachability import e3_cell12_cooperative_run_20261031_main as runner


def test_confirmation_and_cell12_pins():
    with pytest.raises(ValueError, match="confirmation"):
        runner.run("invalid")
    command = runner.precheck_command()
    assert "cell12" in command[command.index("--source") + 1]
    assert command[command.index("--source-sha256") + 1] == runner.AUTH_SHA
    assert command[command.index("--authority-sha256") + 1] == runner.AUTH_SHA
    assert command[command.index("--closure-sha256") + 1] == runner.CLOSURE_SHA
    assert "cell12" in str(runner.OUTPUT)
