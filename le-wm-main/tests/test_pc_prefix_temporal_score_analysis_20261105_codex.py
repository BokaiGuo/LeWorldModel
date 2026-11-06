import pytest

from experiments.pc_prefix_temporal_score_analysis_20261105_codex import summarize


def row(kind, hit, predicted, final):
    return {"index": 0, "chain": kind, "first_success_step": hit, "predicted_final_cost": predicted,
            "steps": [{"step": t, "actual_encoded_cost": final + (25 - t),
                       "success_this_step": (hit is not None and t == 25)} for t in (0, 5, 10, 15, 20, 25)]}


def test_paired_summary_and_missing_pair_fails():
    records = [row("novel", 13, 1, 1), row("common", None, 2, 2)]
    result = summarize(records)
    assert result["ever_success"] == result["success_at_25"] == 1
    assert result["predicted_picks_success"] == result["actual_final_encoded_picks_success"] == 1
    with pytest.raises(ValueError):
        summarize(records[:1])
