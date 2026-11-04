"""Audit integrity and reproduce all gates from saved arrays."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/pc_pointmaze_shadow_renderer_audit_v4_20261103_CODEX.json"
PRIOR = ROOT / "results/pc_pointmaze_rgb_forward_clock_differences_20261103_CODEX.npz"


def test_saved_shadow_replay_gates():
    report = json.loads(RESULT.read_text())
    arrays = RESULT.with_suffix(".npz")
    assert hashlib.sha256(arrays.read_bytes()).hexdigest() == report["arrays_sha256"]
    assert report["passed"] and all(report["checks"].values())
    with np.load(arrays, allow_pickle=False) as data, np.load(PRIOR, allow_pickle=False) as old:
        assert data["states"].shape == (64, 86, 5)
        np.testing.assert_array_equal(data["states"], old["states"][0])
        np.testing.assert_array_equal(data["raw_hashes"], data["sealed_hashes"])
        np.testing.assert_array_equal(data["shadow_hashes"], old["image_hashes"][1])
        assert data["invariants"].shape == (64, 18, 3)
        assert data["invariants"].all()
        assert np.isfinite(data["residuals"]).all()
        assert (data["residuals"] < 1e-12).all()
        assert np.count_nonzero(data["raw_hashes"] != data["shadow_hashes"]) == 1088
