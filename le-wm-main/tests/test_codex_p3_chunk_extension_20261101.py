import pytest

from experiments.spectral_reachability import codex_p3_chunk_extension_20261101 as p3


def costs(**kwargs):
    return {key: kwargs.get(key, 0) for key in p3.AXES}


def config():
    return p3.Config(5, 5, 3, 0.2, 0.1)


def test_dynamic_stops_on_new_signal_and_charges_rejection():
    meter = p3.Meter(costs(predict=3, predictor_samples=6))
    signals = iter([p3.Signal(2, 1, 0.05), p3.Signal(1, 0.9, 0.05)])
    calls = []
    def acquire(start, end):
        calls.append((start, end))
        return next(signals)
    result = p3.schedule(config(), "dynamic", meter,
                         costs(predict=1, predictor_samples=2), acquire)
    assert result["chunk_steps"] == 10
    assert calls == [(5, 10), (10, 15)]
    assert result["attempted"]["predict"] == result["returned"]["predict"] == 2
    assert not result["P3_admission"]


def test_fixed_has_no_signal_or_shadow_cost():
    meter = p3.Meter(costs(predict=3))
    result = p3.schedule(config(), "fixed", meter, costs(), lambda *_: 1 / 0)
    assert result["chunk_steps"] == 5
    assert not any(result["attempted"].values())


def test_budget_prevents_dispatch_and_failure_charges_attempt():
    meter = p3.Meter(costs(predict=1))
    def fail(*_):
        raise RuntimeError("model failure")
    with pytest.raises(RuntimeError):
        p3.schedule(config(), "dynamic", meter, costs(predict=1), fail)
    assert meter.attempted["predict"] == 1
    assert meter.returned["predict"] == 0
    with pytest.raises(ValueError):
        meter.call("retry", costs(), lambda: None)
    meter = p3.Meter(costs())
    with pytest.raises(ValueError):
        p3.schedule(config(), "dynamic", meter, costs(predict=1), fail)
    assert not meter.events


@pytest.mark.parametrize("signal", [
    p3.Signal(float("nan"), 0, 0), p3.Signal(1, 0, -1),
    p3.Signal(1, 0, 0.2), p3.Signal(1, 1, 0),
])
def test_invalid_or_unsafe_model_signal_does_not_extend(signal):
    meter = p3.Meter(costs(predict=1))
    result = p3.schedule(config(), "dynamic", meter, costs(predict=1), lambda *_: signal)
    assert result["chunk_steps"] == 5
    assert result["attempted"]["predict"] == 1


def test_max_extensions_and_complete_vectors():
    meter = p3.Meter(costs(predict=3))
    result = p3.schedule(config(), "dynamic", meter, costs(predict=1),
                         lambda *_: p3.Signal(3, 2, 0))
    assert result["chunk_steps"] == 20
    assert len(result["decisions"]) == 3
    with pytest.raises(ValueError):
        p3.Meter({"predict": 1})
    with pytest.raises(ValueError):
        p3.schedule(config(), "dynamic", p3.Meter(costs()), costs(), lambda *_: None)
