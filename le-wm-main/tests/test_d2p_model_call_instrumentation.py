"""Metadata-only fakes; no torch import, inference or device allocation."""
from types import SimpleNamespace
import json

import pytest

from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation


class Input:
    def __init__(self, batch=32, context=3, device="cuda", index=0):
        self.shape = (batch, context, 192)
        self.device = SimpleNamespace(type=device, index=index)


class Model:
    def predict(self, emb, act):
        return emb


def test_actual_calls_samples_context_and_declared_horizon_are_independent():
    model = Model()
    with PredictInstrumentation(model) as counter:
        with counter.phase("planning", method="cem", horizon_tokens=4) as receipt:
            for _ in range(4):
                value = Input()
                assert model.predict(value, Input()) is value
        with counter.phase("evaluation", method="cem", horizon_tokens=1):
            model.predict(emb=Input(1), act=Input(1))
    assert receipt["predict_calls"] == 4
    assert receipt["predict_samples"] == 128
    assert receipt["context_lengths"] == {3: 4}
    assert receipt["horizon_tokens"] == 4
    assert counter.records[1]["phase"] == "evaluation"
    assert counter.records[1]["predict_samples"] == 1
    json.dumps(counter.records)
    assert "predict" not in vars(model)


@pytest.mark.parametrize("error_source", ["body", "predict"])
def test_restore_exact_instance_override_on_exception(error_source):
    model = Model()
    def original(emb, act):
        if error_source == "predict":
            raise LookupError("model failure")
        return emb
    model.predict = original
    with pytest.raises(LookupError):
        with PredictInstrumentation(model) as counter:
            with counter.phase("planning", method="gn", horizon_tokens=2) as receipt:
                model.predict(Input(1), Input(1))
                raise LookupError("body failure")
    assert model.predict is original
    assert receipt["predict_calls"] == 1
    assert receipt["failed_predict_calls"] == (error_source == "predict")
    assert receipt["completed"] is False


@pytest.mark.parametrize("inputs", [
    (Input(device="cpu"), Input()), (Input(), Input(device="cpu")),
    (Input(index=0), Input(index=1)), (Input(2), Input(3)),
    (Input(0), Input(0)), (Input(context=0), Input(context=0)),
])
def test_invalid_inputs_never_reach_predict(inputs):
    model = Model()
    model.predict = lambda emb, act: pytest.fail("invalid input reached model")
    with PredictInstrumentation(model) as counter:
        with counter.phase("planning", method="x", horizon_tokens=1) as receipt:
            with pytest.raises(ValueError):
                model.predict(*inputs)
    assert receipt["predict_calls"] == 0


def test_unscoped_calls_and_nested_instrumentation_fail_closed():
    model = Model()
    with PredictInstrumentation(model) as counter:
        with pytest.raises(RuntimeError, match="phase"):
            model.predict(Input(), Input())
        with pytest.raises(RuntimeError):
            with PredictInstrumentation(model):
                pass
        with counter.phase("planning", method="a", horizon_tokens=1):
            with pytest.raises(RuntimeError):
                with counter.phase("evaluation", method="a", horizon_tokens=1):
                    pass
            model.predict(Input(1), Input(1))


@pytest.mark.parametrize("phase,method,horizon", [
    ("feedback", "a", 1), ("planning", "", 1), ("planning", "a", True),
    ("planning", "a", 0), ("planning", "a", 1.5),
])
def test_invalid_scopes_rejected(phase, method, horizon):
    with PredictInstrumentation(Model()) as counter:
        with pytest.raises(ValueError):
            with counter.phase(phase, method=method, horizon_tokens=horizon):
                pass


def test_lifecycle_and_record_copies():
    model = Model()
    counter = PredictInstrumentation(model)
    with pytest.raises(RuntimeError):
        with counter.phase("planning", method="a", horizon_tokens=1):
            pass
    with counter:
        with counter.phase("planning", method="a", horizon_tokens=1):
            model.predict(Input(1), Input(1))
    records = counter.records
    records[0]["predict_calls"] = 900
    records[0]["context_lengths"][3] = 900
    assert counter.records[0]["predict_calls"] == 1
    assert counter.records[0]["context_lengths"] == {3: 1}
    with pytest.raises(RuntimeError):
        with counter:
            pass


def test_caught_model_failure_cannot_produce_completed_receipt():
    model = Model()
    def fail(emb, act):
        raise ArithmeticError("failed")
    model.predict = fail
    with PredictInstrumentation(model) as counter:
        with counter.phase("evaluation", method="x", horizon_tokens=1) as receipt:
            with pytest.raises(ArithmeticError):
                model.predict(Input(1), Input(1))
    assert receipt["completed"] is False
    assert receipt["failed_predict_calls"] == 1
    assert model.predict is fail


@pytest.mark.parametrize("shape", [(1,), (1, 3), (1, 3, 2, 2), (True, 3, 192), (1, -1, 192)])
def test_bad_shapes_rejected(shape):
    value = Input()
    value.shape = shape
    with PredictInstrumentation(Model()) as counter:
        with counter.phase("planning", method="x", horizon_tokens=1):
            with pytest.raises(ValueError):
                counter.model.predict(value, Input())


def test_predict_signature_errors_do_not_count_as_invocations():
    model = Model()
    with PredictInstrumentation(model) as counter:
        with counter.phase("planning", method="x", horizon_tokens=1) as receipt:
            with pytest.raises(TypeError):
                model.predict(Input())
            with pytest.raises(TypeError):
                model.predict(Input(), Input(), unexpected=True)
    assert receipt["predict_calls"] == 0


def test_multiple_methods_and_horizons_are_not_aggregated():
    model = Model()
    with PredictInstrumentation(model) as counter:
        for method, horizon in (("gn", 2), ("cem", 3), ("gn", 1)):
            with counter.phase("planning", method=method, horizon_tokens=horizon):
                model.predict(Input(1), Input(1))
    assert [r["method"] for r in counter.records] == ["gn", "cem", "gn"]
    assert [r["horizon_tokens"] for r in counter.records] == [2, 3, 1]
