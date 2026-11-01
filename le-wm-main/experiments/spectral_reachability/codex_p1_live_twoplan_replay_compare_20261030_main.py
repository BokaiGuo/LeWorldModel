"""Compare fixed-input offline and online P1 plans; no new neural inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_live_twoplan_pair_20261030_main as pair
from . import codex_p1_live_best_pair_20261030_main as previous

ROOT = pair.ROOT
NAME = "codex_p1_live_twoplan_replay_compare_20261030_main"
PLAN = ROOT / "results/codex_p1_adam_cp5_20261030_main_RUN01.json"
PLAN_SHA = "455c93a37f8d1200ffe75290e30e939378ed536b9d81ede802da221f7b6c4253"
need = pair.need


def compare_row(offline, live1, live2, reference1, reference2):
    need(offline["index"] == live1["index"] == live2["index"] ==
         reference1["index"] == reference2["index"], "index")
    plan = np.asarray(offline["selected_actions"], dtype="<f4")
    need(plan.shape == (2, 5, 2) and hashlib.sha256(plan.tobytes()).hexdigest() ==
         offline["selected_action_sha256"], "offline action hash")
    new = bytes.fromhex(live1["result"]["raw_horizon_f32_hex"])
    old_first = bytes.fromhex(reference1["result"]["raw_horizon_f32_hex"])[:40]
    new_second = bytes.fromhex(live2["result"]["raw_horizon_f32_hex"])[:40]
    old_second = bytes.fromhex(reference2["result"]["raw_horizon_f32_hex"])[:40]
    need(len(new) == 80 and len(old_first) == len(new_second) == len(old_second) == 40, "action size")
    return {"index": offline["index"],
            "same_initial_hash": offline["initial_f32_sha256"] == live1["initial_sha256"],
            "same_scores": offline["candidate_scores"] == live1["candidate_scores"],
            "same_selected_index": offline["selected_index"] == live1["selected_index"],
            "same_ten_step_plan": plan.tobytes() == new,
            "same_reference_first_packet": new[:40] == old_first,
            "same_midpoint_pixels": live2["pixel_sha256"] == reference2["pixel_sha256"],
            "same_stage2_history": live2["executed_history_sha256"] == reference2["executed_history_sha256"],
            "same_stage2_scores": live2["scores"] == reference2["scores"],
            "same_stage2_selected_step": live2["selected_step"] == reference2["selected_step"],
            "same_stage2_packet": new_second == old_second}


def build():
    plan = pair.read(PLAN, PLAN_SHA)
    live_saved = pair.read(pair.SAVED, pair.SAVED_SHA)
    need(pair.consumer.consume(pair.RUN, pair.RUN_SHA, pair.PREP, pair.PREP_SHA) == live_saved,
         "live saved reconstruction")
    previous_saved = pair.read(previous.SAVED, previous.SAVED_SHA)
    need(previous.saved.consume(previous.LIVE, previous.REPORT_SHA, previous.PREP, previous.PREP_SHA)
         == previous_saved, "reference saved reconstruction")
    need(len(plan["rows"]) == len(live_saved["rows"]) == len(previous_saved["rows"]) == 24,
         "denominator")
    rows = []
    for i, (offline, live, ref) in enumerate(zip(plan["rows"], live_saved["rows"],
                                                   previous_saved["rows"], strict=True)):
        need(offline["index"] == live["index"] == ref["index"] == i
             and all(live[k] == ref[k] for k in ("logical_pair", "root_slot", "condition", "arm", "role")),
             "row identity")
        need(live["accepted_suffix_steps"] > 5 and ref["accepted_suffix_steps"] > 5, "stage2 present")
        reports = []
        for directory in (pair.RUN, previous.LIVE):
            reports.append([json.loads((directory / f"row_{i:02d}/stage_{stage}/report.json").read_bytes())
                            for stage in (1, 2)])
        row = compare_row(offline, *reports[0], *reports[1])
        row.update(root_slot=live["root_slot"], condition=live["condition"], role=live["role"],
                   live_status=live["status"], reference_status=ref["status"],
                   same_endpoint=live["endpoint_metrics"] == ref["endpoint_metrics"],
                   same_last_observed=live["last_observed_metrics"] == ref["last_observed_metrics"])
        rows.append(row)
    keys = [k for k, value in rows[0].items() if k.startswith("same_") and isinstance(value, bool)]
    return {"schema": NAME, "status": "descriptive", "rows": rows,
            "summary": {key: sum(row[key] for row in rows) for key in keys},
            "plan_sha256": PLAN_SHA, "live_report_sha256": pair.RUN_SHA,
            "reference_report_sha256": previous.REPORT_SHA,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "P1_admission": False, "independent_confirmation": False,
            "limitations": ["same known roots and overlapping history", "no new neural or physical experiment",
                            "hash equality checks do not independently recompute latent",
                            "source dependency closure and total budget not certified"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_"),
         "output scope")
    result = build()
    with out.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"summary": result["summary"], "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
