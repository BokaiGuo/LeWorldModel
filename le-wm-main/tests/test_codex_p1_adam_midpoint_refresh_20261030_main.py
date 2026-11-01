import json

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_adam_midpoint_refresh_20261030_main as refresh


def test_bound_vectors_and_fixed_denominator():
    entries = refresh.inputs()
    path = refresh.base.ROOT / "results/codex_p1_adam_midpoint_refresh_20261030_main_RUN01.json"
    saved = json.loads(path.read_text())
    assert len(entries) == len(saved["rows"]) == 19
    assert saved["omitted_native_stopped"] == 5
    assert saved["model_before"] == saved["model_after"] == refresh.base.MODEL_HASH
    for entry, row in zip(entries, saved["rows"], strict=True):
        assert entry[0][0] == row["index"]
        assert entry[3]["pixel_sha256"] == row["mid_pixel_sha256"]
        pred = np.asarray(row["refreshed_latent"], dtype=np.float32)
        actual = np.asarray(row["actual_latent"], dtype=np.float32)
        assert pred.shape == actual.shape == (192,)
        assert np.isclose(np.mean((pred-actual)**2), row["refresh_error_mse"], rtol=1e-6)


def test_altered_executed_action_is_rejected(monkeypatch):
    original = refresh.trace.transitions

    def corrupt(*args):
        events = original(*args)
        events[0] = {**events[0], "actual_bits": "0000000000000000"}
        return events

    monkeypatch.setattr(refresh.trace, "transitions", corrupt)
    with pytest.raises(ValueError, match="executed action mismatch"):
        refresh.inputs()
