"""Build planner-only real-cp5 inputs; retain censored rows without imputation."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_response_prefix_censored_20261031 as physical
from . import codex_p1_three_prefix_blind_inputs_20261030_v1 as blind

ROOT = physical.ROOT
NAME = "p1_response_prefix_cp5_bundle_20261031"
RUN = ROOT / "results/p1_response_prefix_censored_20261031_RUN01"
REPORT_SHA = "b7eed664e4b34f2aea3c7713862cb59180355301e1d32da4358233e81490a6a9"
RECEIPT = ROOT / "results/p1_response_prefix_censored_20261031_consume_RUN01.json"
RECEIPT_SHA = "7087e555b266a78553361001e4f9127c84f74de89ae77d00ae650081db33cb50"
PROTOCOL = ROOT.parent / "docs/D2P_P1_RESPONSE_PREFIX_CP5_PLANNER_PROTOCOL_20261031.md"
need, sha = physical.need, physical.sha


def build():
    read = physical.prior.native.read
    need(sha(RUN / "report.json") == REPORT_SHA and sha(RECEIPT) == RECEIPT_SHA, "physical hashes")
    report, receipt = read(RUN / "report.json", cap=64 * 1024**2), read(RECEIPT)
    need(receipt["status"] == "saved_checks_passed_not_admitted" and
         receipt["report_sha256"] == REPORT_SHA and receipt["rows"] == 16 and
         len(report["rows"]) == 16 and report["status"] == "physical_recorded_not_independently_consumed",
         "physical receipt")
    source_report, prepared, _ = physical.pilot.replay.source.inputs()
    joined = read(physical.pilot.JOIN, cap=32 * 1024**2)
    need(sha(physical.pilot.JOIN) == physical.JOIN_SHA, "join identity")
    roster, bindings = [], {str(RUN / "report.json"): REPORT_SHA, str(RECEIPT): RECEIPT_SHA,
                            str(physical.pilot.JOIN): physical.JOIN_SHA, str(PROTOCOL): sha(PROTOCOL),
                            str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    for path in (physical.pilot.replay.source.panel.source.REPORT,
                 physical.pilot.replay.source.panel.source.RECEIPT, Path(source_report["prepared"]),
                 Path(blind.__file__).resolve()):
        bindings[str(path)] = sha(path)
    for i, row in enumerate(report["rows"]):
        plan = row["plan"]
        slot, logical, offset = plan["slot"], i // 2, i % 2
        need(row["index"] == i and slot == logical // 2 and row["arm"] == plan["arm"], "roster identity")
        item = {"index": i, "logical_pair": logical, "slot": slot, "arm": plan["arm"],
                "candidate_index": plan["candidate_index"], "physical_status": row["status"],
                "blind": None}
        if row["status"] == "censored_out_of_domain" and row["censor"]["step"] <= 5:
            item["unavailable_reason"] = "canvas_censored_before_cp5"
        else:
            checkpoints = [c for c in row["checkpoints"] if c["step"] == 5]
            need(len(checkpoints) == 1, "cp5 missing or duplicated")
            cp = checkpoints[0]
            image = RUN / cp["png"]
            need(image.parent == RUN and image.is_file() and not image.is_symlink()
                 and sha(image) == cp["file_sha256"], "cp5 image")
            task, join_task = prepared["requests"][slot]["task"], joined["task_rows"][slot]
            need(task["binding"]["task_id"] == plan["task_id"] == join_task["task_id"], "task binding")
            raw = np.frombuffer(bytes.fromhex(join_task["warmup_raw_float32_hex"]), "<f4").reshape(10, 2)
            # The established whitelist uses a triplet index convention. Map the two
            # roles into that convention, retaining the actual fixed row separately.
            anchor = {"index": 3 * logical + offset, "logical_pair": logical,
                      "role": ("selected", "zero")[offset], "task_id": plan["task_id"],
                      "prefix5_float32_hex": plan["raw15_float32_hex"][:80], "checkpoint": cp}
            item["blind"] = blind.make_row(anchor, task, raw, image.read_bytes())
            bindings[str(image)] = cp["file_sha256"]
        roster.append(item)
    need(len(roster) == 16 and all(sha(p) == d for p, d in bindings.items()), "source drift")
    return {"schema": NAME, "rows": roster, "fixed_rows": 16,
            "available_cp5": sum(x["blind"] is not None for x in roster), "bindings": bindings,
            "planner_file_isolation_established": False, "P1_admission": False,
            "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    value = build()
    print(json.dumps({"sha256": physical.write(output, value), "rows": len(value["rows"]),
                      "available_cp5": value["available_cp5"]}), flush=True)


if __name__ == "__main__":
    main()
