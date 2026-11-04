"""Verify and summarize the fixed known-root common-ten intervention."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev304_common10_20261102 as source
from .p1_dev303_dynamics_saved_analysis_20261102_main import measures

base, run, saved = source.base, source.run, source.saved
CONTRACT_SHA = "a8636a62b58bb428d4a4a013155a7197726cbfcef28a23302182486474807f78"
REPORT_SHA = "56cd6d48d70833bb9e22d2c7e85108bf258fca6f902668bc81af7a695a5c99e2"
ROOT = source.ROOT / "results"
RUN = ROOT / f"{source.NAME}_RUN_DEV01"
need = source.need


def analyze():
    contract = base.read(ROOT / f"{source.NAME}_PREP_DEV01.json", CONTRACT_SHA)
    need(contract == source.protocol(), "source contract")
    report = base.read(RUN / "report.json", REPORT_SHA)
    need(report["contract_sha256"] == CONTRACT_SHA and report["status"] == "recorded"
         and not report["errors"] and base.sha(RUN / "journal.jsonl") == report["journal_sha256"], "report")
    prior = base.read(saved.ROOT / f"{saved.NAME}_EXEC_DEV01/report.json", saved.EXEC_SHA)
    capture = base.read(saved.ROOT / f"{saved.NAME}_CAPTURE_DEV01/report.json", saved.CAPTURE_SHA)
    goal = capture["goal_state_evaluation_only"]
    expected = [(p, t, old) for p, old in zip(contract["prefixes"], prior["rows"], strict=True)
                for t in contract["tails"]]
    need([r["arm"] for r in report["rows"]] == [p["name"] + "__" + t["name"] for p, t, _ in expected],
         "row roster")
    rows, frames = [], 0
    for row, (prefix, tail, old) in zip(report["rows"], expected, strict=True):
        need(base.read(RUN/(row["arm"]+".row.json")) == row and row["status"] == "fixed_horizon_completed",
             "row or censoring")
        actions = np.asarray(prefix["prefix"] + tail["actions"], dtype="<f4")
        need(actions.shape == (15, 2) and len(row["transitions"]) == 15, "action count")
        for i, (transition, action) in enumerate(zip(row["transitions"], actions, strict=True)):
            need(transition["phase"] == ("prefix" if i < 5 else "tail")
                 and transition["step"] == (i+1 if i < 5 else i-4)
                 and np.array_equal(transition["action"], action)
                 and not transition["terminated"] and not transition["truncated"], "action trace")
            if i < 5:
                need(transition["state"] == old["transitions"][i]["state"], "prefix state")
        obs = row["observations"]
        need([(o["phase"], o["step"]) for o in obs]
             == [("warmup", x) for x in (0, 5, 10)] + [("prefix", 5)]
             + [("tail", x) for x in range(1, 11)], "frame roster")
        for frame in obs:
            saved.verify_frame(RUN, frame)
            frames += 1
        for a, b in zip(obs[:4], [*old["observations"][:3], old["observations"][7]], strict=True):
            need(all(a[k] == b[k] for k in ("pixel_sha256", "state", "snapshot")), "prefix checkpoint")
        need(all(o["state"] == t["state"] for o, t in zip(obs[4:], row["transitions"][5:], strict=True)),
             "tail frames")
        rows.append({"prefix": prefix["name"], "tail": tail["name"],
                     "after_prefix": measures(row["transitions"][4]["state"], goal),
                     "terminal": measures(row["transitions"][-1]["state"], goal),
                     "reward": row["transitions"][-1]["reward"], "native_success": False,
                     "status": row["status"]})
    for row in rows:
        zero = next(r for r in rows if r["prefix"] == row["prefix"] and r["tail"] == "zero")
        row["terminal_minus_own_zero"] = {key: row["terminal"][key]-zero["terminal"][key]
                                          for key in row["terminal"]}
    return {"schema": source.NAME+".saved.v1", "rows": rows, "frames_verified": frames,
            "classification": "one_known_root_post_outcome_development", "p1_admission": False,
            "independent_confirmation": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(base.write_new(p.parse_args().output, analyze()))
