"""Execute every fixed nonwinning multistart suffix on the original prefix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_multistart_physical_20261030_main as original
from . import p1_uphill_adam_candidate_replay_20261031 as replay

NAME = "p1_uphill_nonwinner_physical_20261031"
ROOT = original.physical.ROOT
REPLAY = ROOT / "results/p1_uphill_adam_candidate_replay_20261031_RUN01.json"
REPLAY_SHA = "929732e4a5ed25146de3899a2ff4e1f326e813fe01260fce00a17a2adca0742a"
PROTOCOL = ROOT.parent / "docs/D2P_P1_ALL_NONWINNER_PHYSICAL_PROTOCOL_20261031_MAIN.md"
need, sha = original.need, original.sha


def select_candidate(row, rank):
    need(type(rank) is int and 0 <= rank < 3, "rank")
    need(type(row["winner"]) is int and 0 <= row["winner"] < 4, "winner")
    need(len(row["candidate_actions"]) == len(row["candidate_sha256"]) == len(row["scores"]) == 4,
         "candidate roster")
    index = [i for i in range(4) if i != row["winner"]][rank]
    raw = np.asarray(row["candidate_actions"][index], dtype="<f4")
    need(raw.shape == (2, 5, 2) and np.isfinite(raw).all() and (np.abs(raw) <= 1).all()
         and hashlib.sha256(raw.tobytes()).hexdigest() == row["candidate_sha256"][index],
         "candidate identity")
    return index, raw


def build(rank):
    prior = original.build()
    need(sha(REPLAY) == REPLAY_SHA, "replay hash")
    data = json.loads(REPLAY.read_bytes())
    plan = json.loads(original.PLAN.read_bytes())
    need(data["status"] == "complete" and data["source_sha256"] == original.PLAN_SHA
         and len(data["rows"]) == len(plan["rows"]) == len(prior["roster"]) == 24,
         "source roster")
    roster = []
    for i, (row, old, plan_row) in enumerate(zip(data["rows"], prior["roster"], plan["rows"], strict=True)):
        need(row["index"] == old["index"] == plan_row["index"] == i
             and row["logical_pair"] == old["logical_pair"] == plan_row["logical_pair"]
             and row["role"] == old["role"] == plan_row["role"]
             and row["winner"] == plan_row["selected_index"]
             and row["scores"] == plan_row["candidate_scores"]
             and row["candidate_sha256"][row["winner"]] == plan_row["selected_action_sha256"],
             "plan binding")
        index, raw = select_candidate(row, rank)
        item = dict(old)
        item.update(raw10_float32_hex=raw.tobytes().hex(),
                    raw10_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                    candidate_index=index, original_winner=row["winner"], nonwinner_rank=rank)
        roster.append(item)
    bindings = dict(prior["bindings"])
    for path in (REPLAY, PROTOCOL, Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(k) == v for k, v in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "roster": roster, "bindings": bindings,
            "replay_sha256": REPLAY_SHA, "rank": rank, "fixed_rows": 24,
            "P1_admission": False, "independent_confirmation": False,
            "fairness_proven": False, "os_isolation_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--rank", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--report-sha")
    parser.add_argument("--directory")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    physical = original.physical
    physical.NAME = NAME
    physical.engine.NAME = NAME
    physical.build = lambda: build(args.rank)
    output = Path(args.output)
    if args.command == "prepare":
        need(output.is_absolute() and output.parent == ROOT / "results"
             and output.name.startswith(NAME + f"_prepare_r{args.rank}_"), "output")
        result = {"sha256": physical.design.run._write_new(output, build(args.rank))}
    elif args.command == "run":
        need(args.authorize_physics and Path(args.output).name.startswith(NAME + f"_run_r{args.rank}_"),
             "physics authorization and rank")
        result = physical.execute(args.prepared, args.sha256, args.output)
    else:
        need(output.is_absolute() and output.parent == ROOT / "results"
             and output.name.startswith(NAME + f"_consume_r{args.rank}_"), "output")
        result = {"sha256": physical.design.run._write_new(
            output, physical.consume(args.directory, args.report_sha, args.prepared, args.sha256))}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
