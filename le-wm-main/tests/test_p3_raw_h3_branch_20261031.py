import numpy as np
import pytest
import json
from pathlib import Path

from experiments.spectral_reachability import p3_raw_h3_branch_20261031 as branch


def test_raw20_requires_exact_bounded_float32_and_full_length():
    good = np.zeros((20, 2), dtype=np.float32).astype(float).tolist()
    assert branch.raw20(good).shape == (20, 2)
    for bad in (good[:10], [[1.1, 0.0]] * 20, [[float("nan"), 0.0]] * 20,
                [[0.1, 0.0]] * 20):
        with pytest.raises(ValueError):
            branch.raw20(bad)


def test_sealed_roster_allows_exact_20_actions():
    root = Path(__file__).resolve().parents[1]
    prepared = json.loads((root / "results/p3_raw_h3_physical_prepare_20261031_RUN01.json").read_bytes())
    assert len(prepared["plans"]) == 72
    for plan in prepared["plans"]:
        assert branch.raw20(plan["commands"]).shape == (20, 2)


def test_capture_adds_15_and_20_without_duplicating_earlier_checkpoints(monkeypatch):
    def fake_capture(env, obs, phase, step, row, ops, store, index):
        row["last_observation"] = {"step": step}
        if phase == "plan" and step in (0, 5, 10):
            row["checkpoints"].append({"step": step})
        return None, None
    monkeypatch.setattr(branch.kernel, "capture", fake_capture)
    row = {"checkpoints": []}
    for step in (0, 5, 10, 15, 20):
        branch.capture(None, None, "plan", step, row, None, None, 0)
    assert [x["step"] for x in row["checkpoints"]] == [0, 5, 10, 15, 20]
