"""Consume and audit the true live-second-packet physical run."""
import argparse
import json
from pathlib import Path

from . import p1_live_packet_physical_20261031 as live
from . import codex_p1_fixed_suffix_censored_consume_20261030 as journal


def consume(directory, run_sha, prepared, prep_sha):
    directory, prepared = Path(directory), Path(prepared)
    live.need(live.sha(prepared) == prep_sha and live.sha(directory / "run_report.json") == run_sha,
              "input hashes")
    fixed = json.loads(prepared.read_bytes())
    report = json.loads((directory / "run_report.json").read_bytes())
    live.need(fixed == live.build() and report["status"] == "recorded_not_consumed"
              and len(report["shards"]) == 24 and report["model_before"] == report["model_after"]
              and report["host"]["device"] == "cuda:0", "run contract")
    rows = []
    for i, (item, shard) in enumerate(zip(fixed["roster"], report["shards"], strict=True)):
        live.need(shard["index"] == i and shard["status"] == "recorded_not_consumed", "shard")
        path = directory / f"row_{i:02d}" / "report.json"
        live.need(live.sha(path) == shard["report_sha256"], "row hash")
        row_report = json.loads(path.read_bytes())
        row = row_report["row"]
        events = journal.events(path.parent / "journal.jsonl", row_report["journal_chain_tail"])
        plans = [e for e in events if e["kind"] == "live_packet_plan"]
        observations = [e for e in events if e["kind"] == "fresh_observation"
                        and e["receipt"]["phase"] == "probe" and e["receipt"]["step"] == 5]
        transitions = [e for e in events if e["kind"] == "returned_transition"
                       and e["phase"] == "probe" and e["step"] == 6]
        live.need(len(plans) == 1 and len(observations) == 3 and len(transitions) == 1,
                  "live event roster")
        plan = plans[0]
        live.need(plan["anchor_index"] == i and plan["second_packet_sha256"] ==
                  row_report["live_plan"]["second_packet_sha256"]
                  and plan["winner"] in range(4)
                  and plan["calls_code_path"] == {"encode": 2, "predict": 44,
                                                  "action_encoder": 44, "backward": 40}
                  and max(e["index"] for e in observations) < plan["index"] < transitions[0]["index"],
                  "live ordering")
        live.need(row["status"] in ("probe_completed", "native_stopped", "censored_out_of_domain")
                  and len(row["responses"]) <= 10, "row status")
        rows.append({"index": i, "role": item["role"], "status": row["status"],
                     "accepted_suffix_steps": len(row["responses"]),
                     "native_success": bool(row["responses"] and
                                            row["responses"][-1]["terminated"]),
                     "winner": plan["winner"], "second_packet_sha256": plan["second_packet_sha256"],
                     "synchronized_seconds": plan["synchronized_seconds"]})
    return {"schema": "p1_live_packet_consume_20261031", "status": "saved_checks_passed",
            "rows": rows, "run_sha256": run_sha, "prepare_sha256": prep_sha,
            "host": report["host"], "model_before": report["model_before"],
            "model_after": report["model_after"], "P1_admission": False,
            "independent_confirmation": False, "equal_budget_certified": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--directory", required=True)
    p.add_argument("--run-sha", required=True)
    p.add_argument("--prepared", required=True)
    p.add_argument("--prep-sha", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    live.need(not out.exists() and out.parent == live.ROOT / "results", "output")
    value = consume(a.directory, a.run_sha, a.prepared, a.prep_sha)
    print(json.dumps({"rows": len(value["rows"]),
                      "sha256": live.physical.design.run._write_new(out, value)}))
