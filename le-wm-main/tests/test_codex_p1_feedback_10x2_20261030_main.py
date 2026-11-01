from experiments.spectral_reachability import codex_p1_feedback_10x2_20261030_main as variant


def test_variant_config_and_original_prepare_source_binding():
    variant.configure()
    result = variant.launcher.build()
    assert result["optimizer"]["steps"] == 10
    assert result["stage_caps"]["backward_calls"] == 10
    assert result["stage_caps"]["predict_calls"] == 22
    assert result["rows"] == 24
    assert result["variant"] == variant.NAME
    assert str(variant.Path(variant.__file__).resolve()) in result["sources"]
    assert not result["equal_budget_certified"]
    assert not result["P1_admission"]
