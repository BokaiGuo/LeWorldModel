import json

import numpy as np

from experiments.spectral_reachability import codex_p1_adam_feedback_rank_20261030_main as rank


def test_inputs_vectors_and_preserved_denominator():
    inputs = rank.inputs()
    saved = json.loads((rank.base.ROOT / f"results/{rank.NAME}_RUN01.json").read_text())
    assert len(inputs) == len(saved["rows"]) == 19
    assert saved["censored"] == 5
    assert saved["model_before"] == saved["model_after"] == rank.base.MODEL_HASH
    for entry, row in zip(inputs, saved["rows"], strict=True):
        assert entry[0]["index"] == row["index"]
        assert row["midpoint_pixel_sha256"] == entry[4]["pixel_sha256"]
        for candidate in (row["open"], row["feedback"]):
            pred = np.asarray(candidate["predicted_latent"], dtype=np.float32)
            actual = np.asarray(candidate["actual_latent"], dtype=np.float32)
            assert pred.shape == actual.shape == (192,)
            assert np.isclose(np.mean((pred-actual)**2), candidate["prediction_error_mse"], rtol=1e-6)
