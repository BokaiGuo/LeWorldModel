import pytest

from experiments.spectral_reachability import (
    p3_c3_model_boundary_20261102_exclusive as subject,
)


class Tensor:
    def __init__(self, batch):
        self.shape = (batch, 2, 3)


class Model:
    def __init__(self):
        self.calls = []
        self.failure = None

    def action_encoder(self, actions):
        self.calls.append("encoder")
        if self.failure == "encoder":
            raise RuntimeError("encoder failure")
        return actions

    def predict(self, context, encoded):
        self.calls.append("predict")
        if self.failure == "predict":
            raise RuntimeError("predict failure")
        return encoded


def adapter(model, caps=None, sync=lambda: None, arm="signal"):
    return subject.BoundaryAdapter(
        model, point=20, arm=arm,
        caps=caps or {"action_encoder": 10, "predict": 10, "predictor_sample": 20},
        sync=sync,
    )


def test_real_boundary_dispatch_and_sample_batch_count():
    model = Model()
    meter = adapter(model)
    assert meter.candidate(0, 0, lambda m: m.predict(Tensor(3), "act", phase="signal")) == "act"
    saved = meter.snapshot()
    assert model.calls == ["encoder", "predict"]
    assert saved["attempted"] == saved["returned"] == {
        "action_encoder": 1, "predict": 1, "predictor_sample": 3
    }
    assert saved["by_phase"]["signal"]["attempted"] == saved["attempted"]
    assert saved["P3_admission"] is saved["C3_admission"] is False


@pytest.mark.parametrize("failure,expected", [
    ("encoder", (1, 0, 0, 0)),
    ("predict", (1, 1, 1, 0)),
])
def test_exceptions_at_real_boundaries_fail_closed(failure, expected):
    model = Model()
    model.failure = failure
    meter = adapter(model)
    with pytest.raises(RuntimeError):
        meter.candidate(0, 0, lambda m: m.predict(Tensor(2), "a", phase="search"))
    s = meter.snapshot()
    assert (s["attempted"]["action_encoder"], s["returned"]["action_encoder"],
            s["attempted"]["predict"], s["returned"]["predict"]) == expected
    assert s["failed"] and s["candidate_returned"] == [0, 0]
    with pytest.raises(ValueError):
        meter.candidate(0, 1, lambda m: None)


def test_over_cap_rejected_before_predict_and_candidate_quota_enforced():
    model = Model()
    meter = adapter(model, {"action_encoder": 1, "predict": 1, "predictor_sample": 1})
    with pytest.raises(ValueError, match="cap exceeded"):
        meter.candidate(0, 0, lambda m: m.predict(Tensor(2), "a", phase="search"))
    assert model.calls == ["encoder"]
    assert meter.snapshot()["attempted"]["predict"] == 0
    meter = adapter(Model(), arm="reverse")
    for i in range(5):
        meter.candidate(0, i, lambda m: None)
    with pytest.raises(ValueError, match="quota/order"):
        meter.candidate(0, 5, lambda m: None)


def test_sync_failure_is_attempted_not_returned():
    model = Model()
    def sync():
        raise RuntimeError("asynchronous failure")
    meter = adapter(model, sync=sync)
    with pytest.raises(RuntimeError):
        meter.candidate(0, 0, lambda m: m.predict(Tensor(1), "a", phase="search"))
    assert model.calls == ["encoder"]
    assert meter.snapshot()["attempted"]["action_encoder"] == 1
    assert meter.snapshot()["returned"]["action_encoder"] == 0


def test_invalid_phase_rejected_before_any_model_call():
    model = Model()
    meter = adapter(model)
    with pytest.raises(ValueError):
        meter.candidate(0, 0, lambda m: m.predict(Tensor(1), "a", phase="other"))
    assert model.calls == []
