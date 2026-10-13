import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/pusht/lewm/external_cross_horizon_source_split_robustness_20261004/source_split_cross_horizon_robustness_20261004.json"


def test_source_split_cross_horizon_matrix_is_complete_and_directional():
    report = json.loads(ARTIFACT.read_text())
    assert report["status"] == "complete"
    rows = report["rows"]
    assert len(rows) == 12
    assert {
        (row["source_split"], row["normalization"], row["windows_per_episode"])
        for row in rows
    } == {
        (split, mode, windows)
        for split in ("20261006", "20261007")
        for mode in ("public", "val_zscore", "raw_div100")
        for windows in (2, 8)
    }
    assert all(row["device"] == "cuda" for row in rows)
    for row in rows:
        for key in (
            "h1_to_h4/dynamics_minus_random",
            "h1_to_h8/dynamics_minus_random",
            "h4_to_h8/dynamics_minus_random",
            "h4_to_h8/dynamics_minus_covariance",
            "h4_to_h8/residual_minus_dynamics",
        ):
            value = row["contrasts"][key]
            assert value["ci95"][0] > 0.0
