"""Compare fixed live best-iterate execution with the sealed offline replay."""
import json
from pathlib import Path

from . import codex_p1_adam_best_iterate_compare_20261030_main as prior
from . import codex_p1_live_best_saved_20261030_main as saved

ROOT = prior.ROOT
NAME = "codex_p1_live_best_pair_20261030_main"
LIVE = ROOT / "results/codex_p1_live_best_20261030_main_run_RUN01"
REPORT_SHA = "888ac514840add480296378c7ca3c28fc0310f12516744c5819b5e1730688949"
SAVED = ROOT / "results/codex_p1_live_best_saved_20261030_main_RUN01.json"
SAVED_SHA = "99500ba96104bde727e6e76b56486ffb1078999198f8be9ba329cc8002390f2b"
PREP = ROOT / "results/codex_p1_live_best_20261030_main_prepare_RUN01.json"
PREP_SHA = "420d7d3d09169cae91ac00c431de8976ea737db6b33a1d3b3c129267a6c79991"


def build():
    read = prior.compare.plan.gap.prior.gap.read
    offline = prior.build()
    live, _ = read(SAVED, SAVED_SHA)
    checked = saved.consume(LIVE, REPORT_SHA, PREP, PREP_SHA)
    if checked != live:
        raise ValueError("live saved reconstruction")
    live_report, _ = read(LIVE / "run_report.json", REPORT_SHA)
    offline_report, _ = read(ROOT / f"results/{prior.physical.NAME}_RUN01/run_report.json", prior.REPORT_SHA)
    plan, _ = read(prior.physical.PLAN, prior.physical.SHA)
    source = prior.compare.plan.inputs()
    rows = []
    for i, (a, b, ls, os, p, entry) in enumerate(zip(offline["rows"], live["rows"], live_report["shards"],
                                                    offline_report["shards"], plan["rows"], source, strict=True)):
        if any(x["index"] != i for x in (a, b, ls, os, p)) or any(
                a[k] != b[k] for k in ("root_slot", "condition", "role")):
            raise ValueError("identity")
        live_folder = LIVE / f"row_{i:02d}/physical"
        off_folder = ROOT / f"results/{prior.physical.NAME}_RUN01/row_{i:02d}"
        lt = prior.compare.trace.transitions(live_folder, ls["report_sha256"])
        ot = prior.compare.trace.transitions(off_folder, os["report_sha256"])
        if len(lt) < 5 or len(ot) < 5:
            raise ValueError("first five incomplete")
        fields = ("step", "actual_bits", "state", "snapshot", "terminated", "truncated")
        if any(any(lt[j][key] != ot[j][key] for key in fields) for j in range(5)):
            raise ValueError("first packet physical mismatch")
        stage = json.loads((LIVE / f"row_{i:02d}/stage_2/report.json").read_bytes())
        if stage["pixel_sha256"] != entry[4]["pixel_sha256"]:
            raise ValueError("actual midpoint pixel mismatch")
        same_packet = stage["result"]["raw_horizon_f32_hex"][:80] == p["second_packet_hex"]
        same_trajectory = len(lt) == len(ot) and all(all(x[k] == y[k] for k in fields)
                                                      for x, y in zip(lt, ot, strict=True))
        rows.append({"index": i, "root_slot": a["root_slot"], "condition": a["condition"], "role": a["role"],
                     "live_status": b["status"], "offline_status": a["new_status"],
                     "live_steps": b["accepted_suffix_steps"], "offline_steps": a["new_steps"],
                     "live_success": b["last_observed_metrics"]["native_success"],
                     "offline_success": a["new_last_success"], "same_midpoint": True,
                     "same_second_packet": same_packet, "same_all_transitions": same_trajectory,
                     "live_endpoint": b["endpoint_metrics"], "offline_endpoint": a["new_endpoint"]})
    return {"schema": NAME, "rows": rows, "status": "complete",
            "summary": {"rows": len(rows), "same_midpoint": sum(r["same_midpoint"] for r in rows),
                        "same_second_packet": sum(r["same_second_packet"] for r in rows),
                        "same_all_transitions": sum(r["same_all_transitions"] for r in rows),
                        "live_success_records": sum(r["live_success"] for r in rows),
                        "offline_success_records": sum(r["offline_success"] for r in rows)},
            "code_sha256": prior.compare.base.sha(Path(__file__).resolve()),
            "live_report_sha256": REPORT_SHA, "live_saved_sha256": SAVED_SHA,
            "offline_report_sha256": prior.REPORT_SHA, "P1_admission": False,
            "independent_confirmation": False,
            "limitations": "same known roots and fixed first packets; no independent samples or equal compute claim"}


if __name__ == "__main__":
    out = ROOT / f"results/{NAME}_RUN01.json"
    result = build()
    print(json.dumps({"sha256": prior.compare.base.write(out, result), "summary": result["summary"]}))
