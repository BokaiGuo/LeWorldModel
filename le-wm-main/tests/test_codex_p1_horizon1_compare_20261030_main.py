import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import codex_p1_horizon1_compare_20261030_main as module


def test_midpoint_validates_pixels_and_rejects_tampered_image(tmp_path):
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    image_path = tmp_path / "mid.png"
    Image.fromarray(pixels).save(image_path)
    frame = {"phase": "probe", "step": 5, "condition": "off", "overlay_index": 0,
             "png": image_path.name, "file_sha256": module.h1.physical.old.sha(image_path),
             "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"row": {"trajectory": [frame]}}))
    report_sha = module.h1.physical.old.sha(report_path)
    assert module.midpoint(tmp_path, report_sha, "off") == frame["pixel_sha256"]
    Image.fromarray(np.ones_like(pixels)).save(image_path)
    with pytest.raises(ValueError):
        module.midpoint(tmp_path, report_sha, "off")


def test_midpoint_rejects_missing_condition(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"row": {"trajectory": []}}))
    with pytest.raises(ValueError):
        module.midpoint(tmp_path, module.h1.physical.old.sha(report), "on")
