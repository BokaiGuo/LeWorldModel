from experiments.spectral_reachability import codex_p1_predicted_midpoint_10x2_20261030_main as variant


def test_variant_is_bound_and_keeps_ten_step_optimizer():
    variant.configure()
    result = variant.launcher.build()
    assert result["optimizer"]["steps"] == 10
    assert result["stage_caps"]["backward_calls"] == 10
    assert result["extra_prediction_calls_per_second_stage"] == {"action_encoder": 1, "predict": 1}
    assert result["variant"] == variant.NAME
    assert str(variant.Path(variant.__file__).resolve()) in result["sources"]
    assert not result["P1_admission"]
