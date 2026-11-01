"""Read-only real-run regression; tampering happens only in pytest temp copies."""
import json
import shutil
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_two_stage_saved_20261030_v1 as saved

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"results/codex_p1_two_stage_launcher_20261030_v1_run_CODEX01"
PREP = ROOT/"results/codex_p1_two_stage_launcher_20261030_v1_prepare_CODEX01.json"


@pytest.mark.parametrize("target", ["physical/journal.jsonl", "stage_1/report.json",
                                    "stage_2/planning/events.jsonl", "physical/b0_probe_05_0.png"])
def test_saved_rejects_tampered_real_row(tmp_path, target):
    if not (RUN/"run_report.json").is_file():
        pytest.skip("real exploration not present")
    path = tmp_path/"row_00"
    shutil.copytree(RUN/"row_00", path)
    target_path = path/target
    with target_path.open("ab") as stream:
        stream.write(b"tamper")
    preparation = json.loads(PREP.read_bytes())
    report = json.loads((RUN/"run_report.json").read_bytes())
    anchors = saved.run.h10.native.read(saved.run.h10.INPUT, cap=64*1024**2)["anchors"]
    blind = json.loads(saved.run.base.BUNDLE.read_bytes())["rows"][0]
    with pytest.raises((ValueError, KeyError)):
        saved.check_row(path, anchors[0], blind, report["shards"][0], preparation["sources"])
