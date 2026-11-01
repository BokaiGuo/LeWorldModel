import json

import numpy as np

from experiments.spectral_reachability import codex_p1_normalizer_sensitivity_20261030_main as audit


def test_saved_predictions_and_fixed_action_identity():
    entries = audit.source.inputs()
    old, _ = audit.source.prior.gap.read(
        audit.base.ROOT / "results/codex_p1_adam_rgb_gap_20261030_main_RUN01.json",
        "264d1aaf155896b7ead155624d1ffb074d829be6cee13da80e116500700c48e8")
    result = json.loads((audit.base.ROOT / f"results/{audit.NAME}_RUN01.json").read_text())
    assert len(entries) == len(old["rows"]) == len(result["rows"]) == 19
    assert result["censored_omitted"] == 5
    assert result["model_before"] == result["model_after"] == audit.base.MODEL_HASH
    for entry, previous, row in zip(entries, old["rows"], result["rows"], strict=True):
        assert entry[0] == previous["index"] == row["index"]
        assert entry[2]["raw10_sha256"] == row["action_sha256"]
        pred = np.asarray(row["alternate_predicted_latent"], dtype=np.float32)
        actual = np.asarray(previous["actual_latent"], dtype=np.float32)
        assert pred.shape == actual.shape == (192,)
        assert np.isclose(np.mean((pred - actual) ** 2), row["alternate_prediction_error_mse"], rtol=1e-6)
