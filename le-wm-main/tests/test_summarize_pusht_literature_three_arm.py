from experiments.summarize_pusht_literature_three_arm import summarize


def _row(value: float) -> dict[str, float | int]:
    return {"estimate": value, "ci95_low": value - 0.1, "ci95_high": value + 0.1, "n": 4}


def _report() -> dict:
    def arm(pose: float, xy: float, angle: float, loss: float, effect: float) -> dict:
        metrics = {"pose_success": _row(pose), "final_xy": _row(xy),
                   "final_angle": _row(angle), "contact_loss_rate": _row(loss),
                   "predicted_observed_effect_error": _row(effect)}
        return {"metrics": metrics}
    return {"protocol": {"resolved_device": "cuda"}, "closed_loop": {
        "mode_aware_lookahead": arm(.25, 4.0, .8, 0.0, .2),
        "safe_fallback": arm(.25, 5.0, 1.0, 0.0, .3),
        "adaptive_online": arm(.50, 3.0, .7, 0.0, .1),
    }}


def test_summary_keeps_predictor_and_repair_claims_separate() -> None:
    summary = summarize(_report())
    assert summary["decision"]["state_chart_predictor_independent_gain_supported"] is False
    assert summary["decision"]["adaptive_local_effect_gain_supported"] is True
    assert summary["source_device"] == "cuda"
