"""Execute fixed DLS four-prefix by six-tail physical crossover."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import codex_four_arm_prefix_intervention_20261030 as engine
from . import e3_cp10_dls_sourcefix_20261031 as sourcefix

ROOT = engine.ROOT
NAME = "e3_cp10_dls_crossover_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_E3_DLS_CROSSOVER_PROTOCOL_20261031.md"
PREP = ROOT / "results/e3_cp10_dls_sourcefix_20261031_root0_PREPARE01.json"
PREP_SHA = "81c672b80b37219eb52139e21685a481e89f21be1d7625fb512f08e93de727d7"
RUN = ROOT / "results/e3_cp10_dls_sourcefix_20261031_root0_RUN01"
REPORT_SHA = "62dc2d6e8a5bd3265b0b92a9c01c87625c43cdcba243ae5fd5d24aa09e28def8"
CHECK = ROOT / "results/e3_cp10_dls_sourcefix_saved_20261031_root0_CHECK01.json"
CHECK_SHA = "307f66cb841549ff4426c86b9168a26cbcc8bdb1225b8c713456003755a91691"


def build():
    need, sha = engine.require, engine.sha
    need(sha(PREP) == PREP_SHA and sha(RUN / "report.json") == REPORT_SHA
         and sha(CHECK) == CHECK_SHA, "source SHA")
    prepared = json.loads(PREP.read_bytes())
    report = json.loads((RUN / "report.json").read_bytes())
    checked = json.loads(CHECK.read_bytes())
    need(checked["report_sha256"] == REPORT_SHA and checked["source_bound_nodes"] == 15
         and report["status"] == "recorded_known_root_not_admitted"
         and not report["errors"] and report["journal_sha256"] == sha(RUN / "journal.jsonl")
         and [r["arm"] for r in report["rows"]] == list(engine.ARMS), "source contract")
    trajectories, checkpoints = {}, {}
    for row in report["rows"]:
        arm = row["arm"]
        need(row["initial_exact"] and row["control_returned"] >= (13 if arm == "random" else 15),
             "source execution length")
        trajectories[arm] = np.asarray(row["executed_raw"], dtype=np.float32)
        need(trajectories[arm].shape == (row["control_returned"], 2)
             and np.isfinite(trajectories[arm]).all() and (np.abs(trajectories[arm]) <= 1).all(),
             "source action domain")
        cp = next(c for c in row["checkpoints"] if c["step"] == 5)
        checkpoints[arm] = {key: cp[key] for key in ("state", "snapshot", "pixel_sha256")}
    tails = {
        "zero": np.zeros((10, 2), dtype=np.float32),
        "structured": trajectories["structured"][5:15],
        "native": trajectories["native"][5:15],
        "immediate": trajectories["immediate_progress"][5:15],
        "random8_plus_zero2": np.concatenate((trajectories["random"][5:13],
                                               np.zeros((2, 2), dtype=np.float32))),
        "negative_structured": -trajectories["structured"][5:15],
    }
    schedule = []
    for arm in engine.ARMS:
        for tail, raw in tails.items():
            commands = np.concatenate((trajectories[arm][:5], raw), axis=0)
            need(commands.shape == (15, 2) and np.isfinite(commands).all(), "branch action")
            schedule.append({"index": len(schedule), "trajectory_arm": arm, "prefix_role": arm,
                             "suffix_role": tail, "commands": commands.tolist(),
                             "expected_prefix_checkpoint": checkpoints[arm]})
    bindings = dict(prepared["sources"])
    for path in (PREP, RUN / "report.json", RUN / "journal.jsonl", CHECK, PROTOCOL,
                 Path(__file__).resolve(), Path(engine.__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare.v1", "fixed_branches": 24, "schedule": schedule,
            "bindings": bindings, "P1_admission": False, "fairness_proven": False,
            "independent_confirmation": False}


def configure():
    experiment = sourcefix.setup(0)
    engine.NAME, engine.PREP, engine.PREP_SHA, engine.build = NAME, PREP, PREP_SHA, build
    return experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    args = parser.parse_args()
    experiment = configure()
    if args.command == "prepare":
        result = engine.prepare(args.output)
    else:
        prepared = json.loads(PREP.read_bytes())
        original_data = engine.p3.source_data

        def with_cp10_goal(source):
            data = original_data(source)
            experiment.rebind(prepared, data)
            return data

        print("Execute 24 fixed physical crossover branches; no neural inference.", flush=True)
        with patch.object(engine.p3, "source_data", with_cp10_goal):
            result = engine.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] not in ("prepared", "recorded_not_independently_consumed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
