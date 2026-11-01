from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import (
    p1_unified_cp5_feedback_physical_saved_20261101_codex as subject,
)


PREPARED = subject.ROOT / "results/p1_unified_cp5_feedback_physical_20261101_codex_PREP_MAIN01.json"
PREPARED_SHA = "4b94f2bb476a367a7befe7a53fa9aaf0c2966a5084dab84713dd9336e99e61df"
DIRECTORY = subject.ROOT / "results/p1_unified_cp5_feedback_physical_20261101_codex_run_MAIN01"
REPORT_SHA = "8ebcbc1abc31fa9c59dff63f67d6bfa9f120d4f3f6bdb4c379cc81c3b437df82"


def test_actual_saved_run_is_fully_verified_but_not_admitted():
    result = subject.verify(PREPARED, PREPARED_SHA, DIRECTORY, REPORT_SHA)
    assert result["verified_rows"] == 6
    assert result["interpretation"] == {
        "actual_cp5_rgb_feedback_verified": True,
        "planned_executed_actions_exact": True,
        "root1_selected_repaired_to_success": True,
        "root1_selected_beats_zero_by_steps": 1,
        "selected_beats_same_bank_greedy_on_any_root": False,
        "known_development_roots": 2,
        "fair_total_cost": False,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    assert result["by_root"]["0"]["same_bank_greedy"]["native_success"] is True
    assert result["by_root"]["1"]["same_bank_greedy"]["returned_steps"] == 4


def test_report_tamper_is_rejected_before_interpretation(tmp_path):
    copied = tmp_path / DIRECTORY.name
    copied.mkdir()
    for path in DIRECTORY.iterdir():
        if path.is_file():
            (copied / path.name).write_bytes(path.read_bytes())
    report_path = copied / "report.json"
    report = json.loads(report_path.read_bytes())
    damaged = deepcopy(report)
    damaged["rows"][0]["executed_raw"][0][0] += 0.25
    report_path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="external report SHA"):
        subject.verify(PREPARED, PREPARED_SHA, copied, REPORT_SHA)
