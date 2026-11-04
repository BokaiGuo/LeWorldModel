"""Verify fixed same-start prefix and common-suffix physical interventions."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_common_suffix_20261102 as source
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

base, run, need = source.base, source.run, source.need
ROOT = source.ROOT/"results"
CONTRACT_SHA = "f632c868cf7c5a8cbc8db29f8144d364b0aab62027bd1f2a7aa7af18a8c42249"
REPORT_SHA = "26da24cfad081e360f905ca15f0b6fa38a5d6728957a44451d27cec5c393cc55"


def analyze():
    contract = base.read(ROOT/f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    directory = ROOT/f"{source.NAME}_RUN_DEV01"
    report = base.read(directory/"report.json", REPORT_SHA)
    need(contract == source.protocol() and report["contract_sha256"] == CONTRACT_SHA
         and report["status"] == "recorded" and not report["errors"]
         and base.sha(directory/"journal.jsonl") == report["journal_sha256"], "contract/report")
    capture = base.read(source.prior.previous.ROOT/f"{source.prior.previous.NAME}_CAPTURE_DEV01/report.json",
                        source.prior.previous.CAPTURE_SHA)
    old = base.read(source.prior.previous.ROOT/f"{source.prior.previous.NAME}_EXEC_DEV01/report.json",
                    source.prior.previous.EXEC_SHA)
    baseline = next(row for row in old["rows"] if row["arm"] == "original")
    goal = capture["goal_state_evaluation_only"]
    start = measures(baseline["transitions"][-1]["state"], goal)
    expected = [(p, t) for p in contract["prefixes"] for t in contract["tails"]]
    need(len(expected) == len(report["rows"]) == 30, "fixed roster")
    results, frames, prefix_states = [], 0, {}
    for row, (prefix, tail) in zip(report["rows"], expected, strict=True):
        name = prefix["name"]+"__"+tail["name"]
        need(row["arm"] == name and base.read(directory/(name+".row.json")) == row
             and row["status"] in ("fixed_horizon_completed", "native_early_stop"), "row")
        actions = np.asarray(prefix["actions"]+tail["actions"], dtype="<f4")
        transitions = row["transitions"]
        need(1 <= len(transitions) <= 15, "transition count")
        for i, (transition, action) in enumerate(zip(transitions, actions, strict=False)):
            need(transition["phase"] == ("prefix" if i < 5 else "tail")
                 and transition["step"] == (i+1 if i < 5 else i-4)
                 and np.array_equal(transition["action"], action), "action binding")
            if i < len(transitions)-1:
                need(not transition["terminated"] and not transition["truncated"], "stop binding")
        ended = transitions[-1]["terminated"] or transitions[-1]["truncated"]
        need((row["status"] == "fixed_horizon_completed" and len(transitions) == 15 and not ended)
             or (row["status"] == "native_early_stop" and ended), "censoring")
        obs = row["observations"]
        need(len(obs) >= 5 and [(x["phase"], x["step"]) for x in obs[:4]]
             == [("warmup", 0), ("warmup", 5), ("warmup", 10), ("shared", 5)], "observation roster")
        for frame in obs:
            source.prior.previous.verify_frame(directory, frame)
            frames += 1
        for a, b in zip(obs[:4], [*baseline["observations"][:3], baseline["observations"][7]], strict=True):
            need(all(a[k] == b[k] for k in ("pixel_sha256", "state", "snapshot")), "shared snapshot")
        for frame in obs[4:]:
            transition = next((t for t in transitions
                               if t["phase"] == frame["phase"] and t["step"] == frame["step"]), None)
            need(transition is not None and frame["state"] == transition["state"], "frame transition state")
        prefix_step = transitions[4] if len(transitions) >= 5 else None
        if prefix_step is not None:
            previous_state = prefix_states.setdefault(prefix["name"], prefix_step["state"])
            need(prefix_step["state"] == previous_state, "prefix consistent across common tails")
        results.append({"prefix": prefix["name"], "kind": prefix["kind"], "selected": prefix["selected"],
                        "tail": tail["name"], "status": row["status"], "steps": len(transitions),
                        "after_prefix": measures(prefix_step["state"], goal) if prefix_step else None,
                        "terminal": measures(transitions[-1]["state"], goal), "reward": transitions[-1]["reward"],
                        "terminated": transitions[-1]["terminated"], "truncated": transitions[-1]["truncated"]})
    for item in results:
        zero = next(r for r in results if r["prefix"] == item["prefix"] and r["tail"] == "zero")
        item["minus_own_zero_terminal"] = (
            {k: item["terminal"][k]-zero["terminal"][k] for k in item["terminal"]}
            if item["status"] == zero["status"] == "fixed_horizon_completed" else None)
    return {"schema": source.NAME+".analysis.v1", "start": start, "rows": results, "frames_verified": frames,
            "scope": "post-outcome development; shared prefix and actions replayed; no learned model inference",
            "independent_confirmation": False, "p1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
