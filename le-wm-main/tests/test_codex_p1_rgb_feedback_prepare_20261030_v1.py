from experiments.spectral_reachability import codex_p1_rgb_feedback_prepare_20261030_v1 as prepare


def test_fixed_roster_and_no_outcome_fields():
    value = prepare.build()
    assert value["fixed_rows"] == 24
    assert [r["index"] for r in value["roster"]] == list(range(24))
    assert all("outcome" not in r and "endpoint" not in r and "raw10_float32_hex" not in r
               for r in value["roster"])
    assert value["rgb_evaluations_cap_each_row"] == 1
    assert value["physical_execution_authorized"] is False
    assert value["P1_admission"] is False
