import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.spectral_reachability import codex_p1_horizon1_prediction_gap_20261030_main as gap


def test_frame_rejects_mutated_pixel_file(tmp_path):
    data = np.zeros((224, 224, 3), dtype=np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(data).save(path)
    frame = {"phase": "probe", "step": 10, "condition": "off", "overlay_index": 0,
             "png": path.name, "file_sha256": gap.base.sha(path),
             "pixel_sha256": hashlib.sha256(data.tobytes()).hexdigest()}
    report = {"row": {"trajectory": [frame]}}
    assert gap.frame(tmp_path, report, "off", 10)[1] == frame["pixel_sha256"]
    Image.fromarray(np.ones_like(data)).save(path)
    with pytest.raises(ValueError):
        gap.frame(tmp_path, report, "off", 10)


def test_saved_latent_vectors_recompute_prediction_mse():
    path = Path(__file__).resolve().parents[1] / "results" / (
        "codex_p1_horizon1_prediction_gap_20261030_main_RUN01.json")
    assert gap.base.sha(path) == "f12715da81230f074b153f9ff11ff9aec0567167f6c813d8e2967ff55eaa87d1"
    result = json.loads(path.read_text())
    assert result["status"] == "complete" and len(result["rows"]) == 24
    for i, row in enumerate(result["rows"]):
        assert row["identity"]["index"] == i
        for arm in ("one_packet", "two_packet"):
            value = row[arm]
            pred = np.asarray(value["predicted_latent"], dtype=np.float64)
            real = np.asarray(value["actual_latent"], dtype=np.float64)
            assert pred.shape == real.shape == (192,)
            assert np.mean((pred-real)**2) == pytest.approx(value["prediction_error_mse"], abs=1e-7)
