import numpy as np
from PIL import Image

from experiments.spectral_reachability import p3_rgb_edge_preflight_20261031 as gate


def image():
    return np.zeros((224, 224, 3), dtype=np.uint8)


def test_central_marker_is_allowed_and_edge_marker_is_rejected():
    rgb = image()
    rgb[50:55, 50:55] = gate.features.BLUE
    assert gate.preflight(rgb)["admitted"] is True
    rgb = image()
    rgb[50:55, 7:12] = gate.features.BLUE
    assert gate.preflight(rgb)["admitted"] is False
    rgb = image()
    rgb[50:55, 8:13] = gate.features.BLUE
    assert gate.preflight(rgb)["admitted"] is True


def test_missing_clipped_fragmented_and_invalid_images_are_rejected():
    assert not gate.preflight(image())["admitted"]
    rgb = image()
    rgb[40:45, 0:5] = gate.features.BLUE
    assert not gate.preflight(rgb)["admitted"]
    rgb = image()
    rgb[40:44, 40:44] = gate.features.BLUE
    rgb[60:64, 60:64] = gate.features.BLUE
    assert not gate.preflight(rgb)["admitted"]
    assert not gate.preflight(np.zeros((8, 8, 3), dtype=np.uint8))["admitted"]


def test_saved_incident_cp15_is_rejected_using_only_rgb():
    path = gate.ROOT / "results/p3_h3_real_context_physical_20261031_batch16_RUN01/b1_plan_15.png"
    with Image.open(path) as im:
        rgb = np.asarray(im).copy()
    result = gate.preflight(rgb)
    assert result["admitted"] is False
    assert result["reason"] == "mask absent, too small, or clipped"
