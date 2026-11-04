"""Verify the development-only three-prefix RGB replanning experiment."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_three_prefix_physical_20261102 as source
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

base, need = source.base, source.need
ROOT = source.ROOT / "results"
CONTRACT_SHA = "4f27648d3f84cd83724c5dcbc3bbfadc1c6957ce0d978ed95c5106227f4aa514"
REPORT_SHA = "8505c1f8a6153909f1a1c1d6f7e636099ebf5e60a0df1c69bf45318c5d475e71"


def analyze():
    contract = base.read(ROOT / f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    directory = ROOT / f"{source.NAME}_RUN_DEV01"
    report = base.read(directory / "report.json", REPORT_SHA)
    need(contract == source.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["status"] == "recorded" and not report["errors"]
         and base.sha(directory / "journal.jsonl") == report["journal_sha256"], "source integrity")
    common = source.common_saved.analyze()
    capture = base.read(source.prior.previous.ROOT / f"{source.prior.previous.NAME}_CAPTURE_DEV01/report.json",
                        source.prior.previous.CAPTURE_SHA)
    goal = capture["goal_state_evaluation_only"]
    results, frames = [], 0
    need(len(report["rows"]) == len(contract["arms"]) == 3, "roster")
    for row, arm in zip(report["rows"], contract["arms"], strict=True):
        name = arm["name"]
        need(row["arm"] == name and base.read(directory / (name + ".row.json")) == row, "row binding")
        transitions = row["transitions"]
        actions = np.asarray(arm["prefix"] + arm["tail"], dtype="<f4")
        need(1 <= len(transitions) <= 15, "trajectory length")
        for i, (transition, action) in enumerate(zip(transitions, actions, strict=False)):
            need(transition["phase"] == ("prefix" if i < 5 else "tail")
                 and transition["step"] == (i + 1 if i < 5 else i - 4)
                 and np.array_equal(transition["action"], action), "action and timing")
            if i < len(transitions) - 1:
                need(not transition["terminated"] and not transition["truncated"], "early stop")
        last = transitions[-1]
        ended = last["terminated"] or last["truncated"]
        need((row["status"] == "fixed_horizon_completed" and len(transitions) == 15 and not ended)
             or (row["status"] == "native_early_stop" and ended), "censoring")
        for frame in row["observations"]:
            source.prior.previous.verify_frame(directory, frame)
            frames += 1
            if frame["phase"] in ("prefix", "tail"):
                transition = next(t for t in transitions
                                  if t["phase"] == frame["phase"] and t["step"] == frame["step"])
                need(frame["state"] == transition["state"], "frame binding")
        zero = next(r for r in common["rows"] if r["prefix"] == name and r["tail"] == "zero")
        prefix = measures(transitions[4]["state"], goal) if len(transitions) >= 5 else None
        need(prefix == zero["after_prefix"], "prefix replay")
        terminal = measures(last["state"], goal)
        delta = ({k: terminal[k] - zero["terminal"][k] for k in terminal}
                 if row["status"] == zero["status"] == "fixed_horizon_completed" else None)
        results.append({"arm": name, "status": row["status"], "steps": len(transitions),
                        "after_prefix": prefix, "terminal": terminal, "reward": last["reward"],
                        "terminated": last["terminated"], "truncated": last["truncated"],
                        "minus_own_zero_tail": delta})
    return {"schema": source.NAME + ".analysis.v1", "rows": results, "frames_verified": frames,
            "scope": "one development root; matched suffix optimizer only; anchor is not certified greedy",
            "independent_confirmation": False, "p1_admission": False, "full_cost_certified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
