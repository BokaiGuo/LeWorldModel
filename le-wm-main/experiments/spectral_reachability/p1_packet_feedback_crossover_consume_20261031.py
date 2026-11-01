"""Consume 96 fixed crossover branches without imputing censored horizons."""
import argparse
import copy
import json
from pathlib import Path

from . import p1_packet_feedback_crossover_20261031 as source
from . import codex_p1_h10_paired_table_20261030_main as metric

NAME = "p1_packet_feedback_crossover_consume_20261031"
need, sha = source.need, source.sha


def validate_responses(checked, responses):
    rejected = 1 if checked["status"] == "censored_out_of_domain" else 0
    need(len(responses) + rejected == checked["returned_suffix"] and len(responses) <= 10
         and all(x["step"] == i for i, x in enumerate(responses, 1)), "response order")


def consume(prepared, prepare_sha, directory, report_sha):
    prepared, directory = Path(prepared), Path(directory)
    need(sha(prepared) == prepare_sha and json.loads(prepared.read_bytes()) == source.build(), "prepare")
    fixed = json.loads(prepared.read_bytes())
    report_path = directory / "run_report.json"
    need(sha(report_path) == report_sha, "report hash")
    report = json.loads(report_path.read_bytes())
    need(report["status"] == "recorded_not_consumed"
         and report["prepare_sha256"] == prepare_sha and len(report["shards"]) == 96, "complete report")
    original = json.loads(source.FEEDBACK_PREP.read_bytes())["roster"]
    rows = []
    for anchor, old in zip(fixed["anchors"], original, strict=True):
        for probe in fixed["banks"][anchor["logical_pair"]]:
            shard = report["shards"][4 * anchor["index"] + probe["index"]]
            need(shard["anchor"] == anchor["index"] and shard["probe"] == probe["index"]
                 and shard["status"] == "recorded_not_consumed", "shard identity")
            folder = directory / f"anchor_{anchor['index']:02d}_probe_{probe['index']:02d}"
            item = copy.deepcopy(old)
            item.update(raw10_float32_hex=probe["raw10_float32_hex"], raw10_sha256=probe["raw10_sha256"])
            checked = source.physical.saved_physical.check_row(folder, anchor, item, shard["sha256"])
            saved = json.loads((folder / "report.json").read_bytes())["row"]
            responses = saved["responses"]
            validate_responses(checked, responses)
            endpoint = None
            if len(responses) == 10:
                need(checked["status"] in ("probe_completed", "native_stopped"), "horizon status")
                frame = source.feedback.planner.source.earlier.frame_at(saved, 10, anchor["condition"])
                need(frame["state"] == responses[-1]["state"], "horizon frame")
                endpoint = metric.physical_metrics(responses[-1]["state"], anchor["goal"])
            rows.append({"anchor": anchor["index"], "logical_pair": anchor["logical_pair"],
                         "root_slot": anchor["root_slot"], "condition": anchor["condition"],
                         "arm": anchor["arm"], "role": anchor["role"], "probe": probe["name"],
                         "probe_index": probe["index"], "probe_sha256": probe["raw10_sha256"],
                         "status": checked["status"], "returned_suffix": checked["returned_suffix"],
                         "accepted_suffix_steps": len(responses),
                         "native_success": checked["last_accepted_state_metrics"]["native_success"],
                         "root": metric.physical_metrics(anchor["root_state"], anchor["goal"]),
                         "prefix": metric.physical_metrics(anchor["checkpoint"]["state"], anchor["goal"]),
                         "endpoint": endpoint, "last_accepted": checked["last_accepted_state_metrics"]})
    need(len(rows) == 96 and sha(report_path) == report_sha
         and all(sha(p) == h for p, h in fixed["bindings"].items()), "final drift")
    return {"schema": NAME, "status": "saved_checks_passed", "rows": rows,
            "fixed_branches": 96, "input_prepare_sha256": prepare_sha,
            "input_report_sha256": report_sha, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("prepared", "prepare-sha", "directory", "report-sha", "output"):
        p.add_argument("--" + option, required=True)
    a = p.parse_args()
    output = Path(a.output)
    need(output.is_absolute() and output.parent == source.ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "output")
    result = consume(a.prepared, a.prepare_sha, a.directory, a.report_sha)
    digest = source.physical.design.run._write_new(output, result)
    print(json.dumps({"rows": len(result["rows"]), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
