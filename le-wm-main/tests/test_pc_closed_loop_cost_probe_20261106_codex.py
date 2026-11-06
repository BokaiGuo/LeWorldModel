import pytest

from experiments.pc_closed_loop_cost_probe_20261106_codex import instrument, records


def test_instrument_preserves_result_and_accumulates_calls():
    class Subject:
        def run(self, value):
            return value + 2

    records.clear()
    instrument(Subject, "run", "test")
    assert Subject().run(3) == 5
    assert Subject().run(6) == 8
    assert records["test"]["calls"] == 2
    assert records["test"]["wall_s"] >= 0


def test_instrument_preserves_exception():
    class Subject:
        def run(self):
            raise ValueError("sentinel")

    records.clear()
    instrument(Subject, "run", "failure")
    with pytest.raises(ValueError, match="sentinel"):
        Subject().run()
    assert records["failure"]["calls"] == 1
