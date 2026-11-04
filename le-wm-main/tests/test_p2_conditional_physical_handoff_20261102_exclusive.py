from copy import deepcopy
import json
import subprocess

import pytest

from experiments.spectral_reachability import p2_conditional_physical_handoff_20261102_exclusive as handoff
from experiments.spectral_reachability import p2_conditional_native_recipe_20261102_exclusive as recipe

PLAN = handoff.RESULTS / (handoff.NAME + "_PLAN_01.json")
PLAN_SHA = "df6c41689dc75340b16e297bc12a3267bb554e1dae82c0854468d2c1b69fa035"


def inputs():
    report = json.loads((handoff.MODEL_RUN / "report.json").read_bytes())
    events = [json.loads(line) for line in (handoff.MODEL_RUN / "journal.jsonl").read_bytes().splitlines()]
    return report, events


def test_real_plan_binds_two_fixed_model_winners():
    plan = handoff.verify(PLAN, PLAN_SHA)
    assert plan["status"] == "prepared_execution_blocked"
    assert [x["arm"] for x in plan["arms"]] == list(handoff.ARMS)
    assert all(len(bytes.fromhex(row["full25_raw_hex"])) == 200 for row in plan["arms"])
    assert bytes.fromhex(plan["arms"][0]["full25_raw_hex"])[:120] == bytes.fromhex(
        plan["arms"][1]["full25_raw_hex"])[:120]
    assert not plan["independent_confirmation"] and not plan["equal_total_compute"]


def test_mutations_to_winner_and_journal_fail():
    report, events = inputs()
    changed = deepcopy(report)
    changed["result"]["rows"][0]["arms"][0]["raw_float32_hex"] = "00" * 80
    with pytest.raises(ValueError, match="winner"):
        handoff.winners(changed, events)
    changed_events = deepcopy(events)
    first = next(x for x in changed_events if x["record"]["kind"] == "optimizer_returned")
    first["record"]["payload"]["score"] = float("nan")
    with pytest.raises(ValueError, match="score"):
        handoff.winners(report, changed_events)
    reordered = deepcopy(report)
    reordered["result"]["rows"][0], reordered["result"]["rows"][1] = (
        reordered["result"]["rows"][1], reordered["result"]["rows"][0])
    with pytest.raises(ValueError, match="order"):
        handoff.winners(reordered, events)


def test_reject_bad_plan_sha_and_path():
    with pytest.raises(ValueError, match="SHA drift"):
        handoff.verify(PLAN, "0" * 64)
    with pytest.raises(ValueError, match="absolute"):
        handoff.verify("not-absolute.json", PLAN_SHA)


def test_run_stops_without_physical_dispatch():
    command = [str(handoff.ROOT / ".venv/bin/python"), "-B", "-m",
               "experiments.spectral_reachability." + handoff.NAME, "run",
               "--plan", str(PLAN), "--plan-sha256", PLAN_SHA]
    run = subprocess.run(command, cwd=handoff.ROOT, capture_output=True, text=True, check=False, timeout=30)
    assert run.returncode != 0
    assert "physical execution blocked" in run.stderr


def test_capture_pixels_equal_saved_prefix_events():
    bundle, _, _, _ = handoff.old.launch.wire.load_capture(
        handoff.old.launch.CAPTURE / "capture.json", handoff.old.launch.CAPTURE_SHA)
    prepared = json.loads(handoff.old.PREP.read_bytes())
    references = prepared["prefix_reference"]
    for frame in bundle["frames"]:
        matches = [row["receipt"]["pixel_sha256"] for row in references
                   if row["kind"] == "fresh_observation"
                   and row["receipt"]["phase"] == "plan"
                   and row["receipt"]["step"] == frame["step"]
                   and row["receipt"]["overlay_index"] == 0]
        assert matches == [frame["pixel_sha256"]]


def test_recipe_distinguishes_new_witness_and_old_runner_rejects():
    value = recipe.build()
    original = json.loads(handoff.old.PREP.read_bytes())
    plan = handoff.verify(PLAN, PLAN_SHA)
    assert not value["execution_authorized"]
    for row, entry in zip(value["plans"], plan["arms"], strict=True):
        assert row["raw_float32_hex"] == entry["full25_raw_hex"]
        assert row["new_model_plan"]["model_report_sha256"] == handoff.REPORT_SHA
        assert row["new_model_plan"] != original["plans"][0]["new_model_plan"]
        assert row["selection_digest"] == handoff.old.physical.digest(
            {key: val for key, val in row.items() if key != "selection_digest"})
    modified = deepcopy(original)
    modified["plans"] = value["plans"]
    with pytest.raises(ValueError):
        handoff.old.physical.validate_execution(modified)
