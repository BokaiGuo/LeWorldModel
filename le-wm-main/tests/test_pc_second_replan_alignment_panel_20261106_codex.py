from experiments.pc_second_replan_alignment_panel_20261106_codex import representatives


def test_representatives_include_success_only_when_available():
    records = [
        {"occurrences": [[0, 0]], "costs": [1.0], "first_success_step": None},
        {"occurrences": [[1, 0]], "costs": [3.0], "first_success_step": 20},
        {"occurrences": [[19, 0]], "costs": [2.0], "first_success_step": None},
    ]
    picked = representatives(records)
    assert picked["best_model"] is records[0]
    assert picked["best_success"] is records[1]
    assert picked["selected_mean"] is records[2]
    assert "best_success" not in representatives([records[0], records[2]])
