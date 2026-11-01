from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import codex_p1_rgb_visual_diagnostic_20261030_v1 as diag

ROOT = Path(__file__).resolve().parents[1]


def image():
    a = np.full((224, 224, 3), 255, dtype=np.uint8)
    y, x = np.ogrid[:224, :224]
    a[(x-80)**2+(y-80)**2 <= 5**2] = (65, 105, 225)
    a[100:120, 100:120] = (119, 136, 153)
    return a


def test_separate_visible_marker_is_only_tentative():
    r = diag.assess(image())
    assert r["decision"] == "tentative_visual_marker"
    assert r["centroid_xy"] == [80., 80.]
    assert r["occlusion_free_certified"] is False


def test_multiple_clipped_and_near_object_are_rejected():
    a = image()
    a[35:38, 35:38] = (65, 105, 225)
    assert diag.assess(a)["reason"] == "blue marker absent or multiple components"
    a = image()
    a[75:86, 85:90] = (119, 136, 153)
    assert diag.assess(a)["reason"] == "object proximity or unavailable context"
    a = image()
    a[:, :100] = 255
    a[70:80, 0:10] = (65, 105, 225)
    assert diag.assess(a)["reason"] == "marker clipped"
    a = image()
    a[70:90, 70:90] = 255
    a[75:85, 75:85] = (65, 105, 225)
    assert diag.assess(a)["reason"] == "marker shape ambiguous"
    with pytest.raises(ValueError):
        diag.assess(np.zeros((224, 224, 3), dtype=np.float32))


def test_real_rejected_cp10_has_fragmented_exact_color():
    path = (ROOT/"results/codex_p1_rgb_feedback_launcher_20261030_v1_run_CODEX01/"
            "row_00/physical/b0_probe_05_0.png")
    result = diag.assess(np.asarray(Image.open(path).convert("RGB")))
    assert [c["count"] for c in result["exact_components"]] == [30, 8]
    assert result["decision"] == "rejected"
    assert result["reason"] == "object proximity or unavailable context"
