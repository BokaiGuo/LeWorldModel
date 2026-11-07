import pytest
import torch

from experiments.pc1_layer_audit import dino_repair_run as pipeline


def test_parity_rejects_outlier_even_with_small_mean_error():
    reference = torch.ones(100000)
    actual = reference.clone()
    actual[0] += .001
    record = pipeline.error_record(actual, reference)
    assert record["relative_l2"] < pipeline.RELATIVE
    assert record["violations"] == 1 and not record["passed"]
    with pytest.raises(ValueError, match="outside frozen tolerance"):
        pipeline.require_parity(actual, reference, "test")


@pytest.mark.parametrize("actual", [torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                                   torch.zeros(2)])
def test_parity_rejects_nonfinite_and_mismatched_shape(actual):
    with pytest.raises(ValueError, match="nonfinite or unmatched"):
        pipeline.error_record(actual, torch.ones(1))


def test_evaluation_requires_every_prediction_before_loading_model(monkeypatch):
    calls = []
    def record(name):
        calls.append(name)
        if len(calls) == 2:
            raise FileNotFoundError("unsealed root")
        return {"model_fingerprint": "test"}
    monkeypatch.setattr(pipeline, "read_record", record)
    def disallowed():
        pytest.fail("model must not load before all roots are sealed")
    monkeypatch.setattr(pipeline, "models", disallowed)
    with pytest.raises(FileNotFoundError, match="unsealed"):
        pipeline.evaluate()
    assert calls == ["predict_r1832.json", "predict_r1833.json"]


def test_repaired_checkpoint_cannot_be_loaded_as_rgb():
    from experiments.pc1_layer_audit.dino_source_loader import load_model
    with pytest.raises(ValueError, match="RGB-only is forbidden"):
        load_model(track="primary_rgb")
