import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_dev303_goal_tangent_builder_20261102_exclusive as builder


def image():
    rgb = np.full((224, 224, 3), 255, np.uint8)
    rgb[50:65, 60:75] = [100, 110, 120]
    rgb[100:110, 100:110] = [65, 105, 225]
    return rgb


def test_builder_uses_fixed_perturbations_and_central_differences(monkeypatch):
    calls = []
    jacobians = []
    construct = builder.tangent.construct
    def capture(jac):
        jacobians.append(jac.copy())
        return construct(jac)
    monkeypatch.setattr(builder.tangent, "construct", capture)
    fake_preflight = SimpleNamespace(image_transform=lambda: object())
    name = "experiments.spectral_reachability.preflight_pusht_d2p_p0"
    monkeypatch.setitem(sys.modules, name, fake_preflight)
    monkeypatch.setattr(builder, "_encode_rgb", lambda model, rgb, device, transform, preflight:
                        calls.append(rgb.copy()) or np.eye(192, dtype=np.float64)[len(calls) // 2] * (1 if len(calls) % 2 == 0 else -1))
    sqrt, meta = builder.build(SimpleNamespace(training=False), image(), "cuda:0")
    assert len(calls) == meta["encoder_calls"] == 11
    assert np.array_equal(jacobians[0], 2 * np.eye(192, dtype=np.float64)[:, 1:6])
    assert sqrt.shape == (192, 192) and sqrt.dtype == np.float64
    assert len(meta["singular_values"]) == 5 and len(meta["goal_latent"]) == 192
    assert meta["goal_pixel_sha256"] == hashlib.sha256(image().tobytes()).hexdigest()
    assert [(x["axis"], x["amount"]) for x in meta["perturbations"]] == [
        (axis, amount) for axis in builder.tangent.AXES for amount in (0.5, -0.5)]
    for record, pixels in zip(meta["perturbations"], calls[1:]):
        assert record["pixel_sha256"] == hashlib.sha256(pixels.tobytes()).hexdigest()
    assert meta["P1_admission"] is False


def test_visibility_rejected_before_model_or_preflight(monkeypatch):
    monkeypatch.setattr(builder, "_encode_rgb", lambda *args: pytest.fail("unexpected encode"))
    with pytest.raises(ValueError, match="visibility"):
        builder.build_goal_tangent_metric(SimpleNamespace(training=False), np.full((224, 224, 3), 255, np.uint8), "cuda:0")


@pytest.mark.parametrize("device,training", [("cpu", False), ("cuda", False), ("cuda:0", True)])
def test_rejects_invalid_device_or_training(device, training):
    with pytest.raises(ValueError):
        builder.build_goal_tangent_metric(SimpleNamespace(training=training), image(), device)


def test_bad_input_rejected():
    with pytest.raises(ValueError, match="uint8 RGB"):
        builder.build_goal_tangent_metric(SimpleNamespace(training=False), image().astype(float), "cuda:0")
