from pathlib import Path

from experiments.theory_spectral_transfer.audit_dino_action_alignment import audit_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_historical_future_only_artifacts_are_flagged_as_source_aligned_noops():
    paths = [
        ROOT / "results/theory_operator_alignment_iclr_oral_20260829/hardened_ablation_20260901/metrics/dino_pusht_future_zero.json",
        ROOT / "results/theory_operator_alignment_iclr_oral_20260829/hardened_ablation_20260901/metrics/dino_pointmaze_future_zero.json",
        ROOT / "results/theory_operator_alignment_iclr_oral_20260829/hardened_ablation_20260901/metrics/dino_wall_future_zero.json",
    ]
    report = audit_artifacts(paths)
    assert len(report["rows"]) == 3
    assert all(row["historical_h1_source_preserved"] for row in report["rows"])
    assert all(row["corrected_h1_source_intervened"] for row in report["rows"])
