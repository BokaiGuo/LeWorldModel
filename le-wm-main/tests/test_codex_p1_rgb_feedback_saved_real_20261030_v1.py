"""Saved-run regression on the actual CODEX01 artifact; no new execution."""
import copy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_rgb_feedback_saved_20261030_v1 as saved

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT/"results/codex_p1_rgb_feedback_launcher_20261030_v1_prepare_CODEX01.json"
RUN = ROOT/"results/codex_p1_rgb_feedback_launcher_20261030_v1_run_CODEX01"
PREP_SHA = "c9d8f1dd4a27b51b68409fe6cd2e5bcfc5b923a5412cff0fbfe32f9675870316"
RUN_SHA = "a1c55ba949850607f979452dc36852b96cdf293cd9b994c6c2f92f24c981e536"


def test_complete_real_run_consumes_all_and_keeps_rejections():
    result = saved.consume(RUN, RUN_SHA, PREP, PREP_SHA)
    assert result["status"] == "saved_checks_passed"
    assert [r["index"] for r in result["rows"]] == list(range(24))
    rejected = [r for r in result["rows"] if r["status"] == "rgb_rejected"]
    assert [r["index"] for r in rejected] == [0, 18, 21]
    assert all(r["accepted_suffix_steps"] == 5 and r["endpoint_metrics"] is None
               and len(r["stage_resources"]) == 2 for r in rejected)


def test_goal_source_tampering_fails_on_actual_journal():
    run = saved.run
    prepared = run.verify(PREP, PREP_SHA)
    anchors = run.h10.native.read(run.h10.INPUT, cap=64*1024**2)["anchors"]
    blind = json.loads(run.base.BUNDLE.read_bytes())["rows"]
    report = json.loads((RUN/"run_report.json").read_bytes())
    source_report = run.h10.native.read(run.h10.old.prep.DIRECTORY/"report.json", cap=64*1024**2)
    _, data = run.h10.runner.physical.verify(source_report["execution_binding"]["prepared"],
                                              run.h10.old.prep.PREP_SHA)
    bad = copy.deepcopy(prepared["roster"][1])
    bad["goal_rgb_pixel_sha256"] = "0"*64
    with pytest.raises(ValueError, match="fixed goal source"):
        saved.check_row(RUN/"row_01", anchors[1], blind[1], report["shards"][1],
                        prepared["sources"], data, bad)
