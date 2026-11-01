"""Saved journal and raw-action checks, without CUDA or model reexecution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from experiments.spectral_reachability import codex_p2_cp15_cuda_20261030_v1 as driver

NAME = "codex_p2_cp15_cuda_saved_20261030_v1"


def validate_slot(row, events, base, arm, case):
    need = driver.panel.require
    need((row["base"], row["arm"], row["status"]) == (base, arm, "complete"), "slot identity")
    starts = [x for x in events if x["kind"] == "slot_start"]
    ends = [x for x in events if x["kind"] == "slot_complete"]
    need(len(starts) == len(ends) == 1 and starts[0]["context_sha256"] ==
         case["feedback_context_sha256"] and ends[0]["row"] == row, "slot boundaries")
    e = [x["event"] for x in events if x["kind"] == "slot_event"]
    for axis, cap in driver.LIMIT.items():
        for phase in ("attempt", "return"):
            need([x["ordinal"] for x in e if x["kind"] == phase and x["resource"] == axis] ==
                 list(range(1, cap+1)), "operation census " + axis)
    need(row["attempted"] == row["returned"] == driver.native.LIMITS and
         row["extra_attempted"] == row["extra_returned"] == {"predictor": 39, "pred_proj": 39}
         and row["elapsed_seconds"] <= 450, "slot caps")
    candidates = []
    packets = []
    for event in e:
        if event["kind"] == "actual_cuda_raw_packet_before_transition":
            raw = bytes.fromhex(event["raw_hex"])
            need(event["dtype"] == "<f4" and event["shape"] == [5, 2] and len(raw) == 40 and
                 hashlib.sha256(raw).hexdigest() == event["raw_sha256"], "packet bits")
            values = np.frombuffer(raw, dtype="<f4")
            need(np.isfinite(values).all() and np.all(np.abs(values) <= 1), "raw domain")
            packets.append(event)
        elif event["kind"] == "candidate":
            need(len(packets) >= 3 and [x["packet"] for x in packets[-3:]] == [0, 1, 2] and
                 event["raw_hex"] == "".join(x["raw_hex"] for x in packets[-3:]), "candidate raw binding")
            candidates.append(event)
    need(len(packets) == 39 and len(candidates) == 11, "packet and candidate census")
    for event in candidates:
        witness = event["witness"]
        driver.native.bridge.validate_logits(witness["logits"], driver.native.CAP)
        terminal = np.asarray(witness["projected_terminal"], np.float64)
        goal = np.asarray(case["goal"], np.float64)
        need(terminal.shape == goal.shape == (192,) and np.isfinite(terminal).all() and
             math.isfinite(witness["cost"]) and float(np.mean((terminal-goal)**2)) == witness["cost"],
             "candidate objective")
    best = candidates[0]
    for event in candidates[1:9]:
        if event["witness"]["cost"] < best["witness"]["cost"]:
            best = event
    need(best["witness"] == row["winner"] and best["raw_hex"] == row["winner_raw_hex"]
         and all(x["raw_hex"] == row["winner_raw_hex"] and
                 x["witness"]["contexts"] == row["winner"]["contexts"] and
                 x["witness"]["projected_terminal"] == row["winner"]["projected_terminal"]
                 for x in candidates[-2:]), "winner and complete replay")
    raw = bytes.fromhex(row["winner_raw_hex"])
    need(len(raw) == 120 and hashlib.sha256(raw).hexdigest() == row["winner_raw_sha256"], "winner SHA")
    return dict(base=base, arm=arm, winner_raw_sha256=row["winner_raw_sha256"],
                cost=row["winner"]["cost"], elapsed_seconds=row["elapsed_seconds"])


def consume(run, report_sha, prep_path, prep_sha):
    run = Path(run)
    need = driver.panel.require
    need(driver.sha(run / "report.json") == report_sha and driver.sha(prep_path) == prep_sha,
         "external SHA")
    report = json.loads((run / "report.json").read_bytes())
    prep = json.loads(Path(prep_path).read_bytes())
    need(json.loads((run / "prepare.json").read_bytes()) == prep and
         report["prepare_sha256"] == prep_sha and report["status"] == "exploratory_complete" and
         report["error"] is None and report["model_before"] == report["model_after"] ==
         prep["expected_model_hash"], "terminal and model identity")
    driver.identity.verify(prep["sources"])
    driver.identity.verify(report["loaded_sources"])
    need(driver.sha(driver.PREP) == prep["panel_sha256"] == driver.PREP_SHA, "panel SHA")
    panel = json.loads(driver.PREP.read_bytes())
    events = driver.journal.read(run / "events.jsonl", report["journal_tail"], driver.CAP)
    need(events[0]["kind"] == "start" and events[0]["order"] == [list(x) for x in driver.ORDER] and
         events[-1]["kind"] == "terminal" and events[-1]["status"] == "exploratory_complete",
         "journal boundaries")
    need(len(report["rows"]) == len(driver.ORDER), "fixed denominator")
    results = []
    for row, (base, arm) in zip(report["rows"], driver.ORDER, strict=True):
        relevant = [e for e in events if e.get("base") == base and e.get("arm") == arm]
        results.append(validate_slot(row, relevant, base, arm, panel["cases"][base]))
    return dict(schema=NAME+".receipt", status="saved_checked_not_scientific",
                report_sha256=report_sha, prepare_sha256=prep_sha, rows=results,
                independent_model_reexecution=False, physical=False, scientific_pass=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--report-sha", required=True)
    p.add_argument("--prepare", required=True)
    p.add_argument("--prepare-sha", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    driver.panel.require(out.is_absolute() and out.parent == driver.ROOT / "results" and
                         out.name.startswith(NAME+"_RECEIPT_"), "exclusive receipt path")
    print(driver.write_new(out, consume(a.run, a.report_sha, a.prepare, a.prepare_sha)))


if __name__ == "__main__":
    main()
