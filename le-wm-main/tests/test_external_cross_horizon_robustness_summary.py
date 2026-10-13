import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/pusht/lewm/external_cross_horizon_readout_20261004/cross_horizon_normalization_density_robustness_20261004.json"


def test_cross_horizon_robustness_matrix_is_complete_and_cuda_backed():
    report = json.loads(ARTIFACT.read_text())
    assert report["status"] == "complete"
    rows = report["rows"]
    assert len(rows) == 9
    assert {(row["normalization"], row["windows_per_episode"]) for row in rows} == {
        (mode, windows)
        for mode in ("public", "val_zscore", "raw_div100")
        for windows in (2, 4, 8)
    }
    assert all(row["device"] == "cuda" for row in rows)
    assert all(row["episodes"] == 21 for row in rows)
    assert all(row["bootstrap_repeats"] == 2000 for row in rows)
    for row in rows:
        for value in row["contrasts"].values():
            assert len(value["ci95"]) == 2
            assert all(isinstance(item, float) for item in value["ci95"])
