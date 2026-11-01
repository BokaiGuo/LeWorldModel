import hashlib
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_pusher_projection_diagnostic_20261030_main as mod


def test_saved_projection_diagnostic_has_fixed_denominator_and_consistent_energy():
    path = mod.base.ROOT / "results/codex_p1_pusher_projection_diagnostic_20261030_main_RUN01.json"
    if not path.is_file():
        pytest.skip("host CUDA result unavailable")
    data = json.loads(path.read_bytes())
    assert data["status"] == "complete"
    assert len(data["rows"]) == 22 and len(data["excluded"]) == 2
    assert {r["index"] for r in data["excluded"]} == {2, 5}
    assert data["encode_calls"] == 20
    assert data["sources"][str(Path(mod.__file__).resolve())] == hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
    for row in data["rows"]:
        assert min(row["singular_values"]) > 1e-6
        for label in ("actual", "predicted"):
            metric = row[label]
            assert 0 <= metric["fraction_energy"] <= 1 + 1e-5
            assert metric["pusher_subspace_mse"] == pytest.approx(
                metric["goal_mse"] * metric["fraction_energy"], abs=1e-6)
