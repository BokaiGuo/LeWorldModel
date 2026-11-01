import pytest

from experiments.spectral_reachability import e3_uncovered_fixed_inputs_20261101_exclusive as subject


@pytest.mark.parametrize("cell", sorted(subject.PINS))
def test_actual_fixed_sources_build(cell):
    result = subject.build(cell)
    assert result["cell"] == cell
    assert result["control5"]["receipt"]["sha256"] == subject.PINS[cell][3]


def test_reject_unlisted_cell():
    with pytest.raises(Exception):
        subject.build(10)


def test_reject_bad_output():
    with pytest.raises(Exception):
        subject.prepare(5, "/tmp/not-an-exclusive-authority.json")


def test_reject_source_pin_drift(monkeypatch):
    pins = dict(subject.PINS)
    row = list(pins[5])
    row[1] = "0" * 64
    pins[5] = tuple(row)
    monkeypatch.setattr(subject, "PINS", pins)
    with pytest.raises(Exception):
        subject.build(5)
