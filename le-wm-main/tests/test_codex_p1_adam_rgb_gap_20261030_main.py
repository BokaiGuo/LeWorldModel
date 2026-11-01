import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import codex_p1_adam_rgb_gap_20261030_main as gap


def test_saved_uncensored_replay_vectors():
    entries = gap.inputs()
    path = gap.base.ROOT / "results/codex_p1_adam_rgb_gap_20261030_main_RUN01.json"
    data = json.loads(path.read_text())
    assert len(entries) == len(data["rows"]) == 19
    assert data["omitted_native_stopped"] == 5
    assert data["model_before"] == data["model_after"] == gap.base.MODEL_HASH
    for entry, row in zip(entries, data["rows"], strict=True):
        assert entry[0] == row["index"]
        assert entry[2]["raw10_sha256"] == row["action_sha256"]
        assert entry[6]["pixel_sha256"] == row["terminal_pixel_sha256"]
        predicted = np.array(row["predicted_latent"], dtype=np.float32)
        actual = np.array(row["actual_latent"], dtype=np.float32)
        assert predicted.shape == actual.shape == (192,)
        assert np.isclose(np.mean((predicted - actual) ** 2), row["prediction_error_mse"], rtol=1e-6)
