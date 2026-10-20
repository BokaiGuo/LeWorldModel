from experiments.summarize_pusht_phase_repair import summarize


def _arm(values):
    return {"metrics": {
        "pose_success": {"estimate": sum(v[0] for v in values) / len(values)},
        "final_xy": {"estimate": sum(v[1] for v in values) / len(values)},
        "final_angle": {"estimate": sum(v[2] for v in values) / len(values)},
        "contact_loss_rate": {"estimate": 0.0},
        "predicted_observed_effect_error": {"estimate": 0.2},
    }, "rows": [{"pose_success": bool(v[0]), "final_xy": v[1], "final_angle": v[2]} for v in values]}


def test_phase_repair_summary_is_paired() -> None:
    baseline = [(False, 10.0, 1.0), (False, 12.0, 1.2)]
    repair = [(True, 5.0, 0.8), (False, 11.0, 1.1)]
    result = summarize({"protocol": {"resolved_device": "cuda"}, "closed_loop": {
        "safe_fallback": _arm(baseline),
        "orientation_pair_reverse_phase_gate_shielded_two_pulse_macro": _arm(repair),
    }})
    assert result["decision"]["phase_repair_positive_point_estimate"] is True
    assert result["paired"]["final_xy"]["repair_wins"] == 2
