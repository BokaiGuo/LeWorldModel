"""Execute all saved cp5 feedback candidates on one verified physical prefix."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import e3_cp10_dls_crossover_20261031 as prior

engine, ROOT = prior.engine, prior.ROOT
NAME = "e3_mix_candidate_physical_20261101"
PROTOCOL = ROOT.parent / "docs/D2P_E3_MIX_CANDIDATE_INTERVENTION_PROTOCOL_20261101.md"
RUN = ROOT / "results/e3_cp10_dls_feedback_mix_20261031_root0_RUN01"
REPORT_SHA = "48f29ac934881d73044956fff118f47e9f0ee80daaf5cf0e9196278e10d47632"
CHECK = ROOT / "results/e3_cp10_dls_feedback_mix_saved_20261031_root0_CHECK01.json"
CHECK_SHA = "1c26fcdd3861f3da45b84e20112589568775cbf68758a463b4038cb4cd9954b9"


def build():
    value = prior.build()
    need, sha = engine.require, engine.sha
    report = RUN / "report.json"
    need(sha(report) == REPORT_SHA and sha(CHECK) == CHECK_SHA, "source identity")
    data, verified = json.loads(report.read_bytes()), json.loads(CHECK.read_bytes())
    need(verified["report_sha256"] == REPORT_SHA and verified["checked_structured_nodes"] == 4
         and data["status"] == "recorded_known_root_not_admitted", "saved verification")
    row = data["rows"][0]
    need(row["arm"] == "structured" and row["initial_exact"], "structured identity")
    node = next(n for n in data["nodes"][row["episode_id"]] if n["boundary"] == 5)
    trace = Path(node["receipt"]["trace"])
    need(sha(trace) == node["receipt"]["trace_sha256"], "trace hash")
    events = [json.loads(line) for line in trace.read_bytes().splitlines()]
    payload = {event["kind"]: event["payload"] for event in events}
    bank = np.asarray(payload["optimized_candidates"]["raw"], dtype=np.float32)
    selection = payload["final_selection"]
    refinement = payload["winner_refinement"]
    winner = selection["index"]
    need(bank.shape == (8, 3, 5, 2) and np.isfinite(bank).all()
         and (np.abs(bank) <= 1).all() and winner == refinement["index"], "candidate bank")
    if refinement["accepted"]:
        bank[winner] = np.asarray(refinement["after_raw"], dtype=np.float32)
    need(np.array_equal(bank[winner, 0], np.asarray(node["receipt"]["commands"], dtype=np.float32)),
         "winner action binding")
    prefix = np.asarray(row["executed_raw"][:5], dtype=np.float32)
    cp = next(cp for cp in row["checkpoints"] if cp["step"] == 5)
    expected = {key: cp[key] for key in ("state", "snapshot", "pixel_sha256")}
    schedule = []
    for index in range(8):
        for variant in ("two_packets", "first_packet_zero", "zero"):
            tail = bank[index, :2].reshape(10, 2).copy()
            if variant == "first_packet_zero":
                tail[5:] = 0
            if variant == "zero":
                tail[:] = 0
            commands = np.concatenate((prefix, tail))
            schedule.append({"index": len(schedule), "trajectory_arm": "structured",
                             "prefix_role": "structured", "suffix_role": f"candidate{index}_{variant}",
                             "commands": commands.tolist(), "expected_prefix_checkpoint": expected,
                             "candidate": index, "winner": index == winner,
                             "model_h15_score": selection["scores"][index]})
    value["schema"], value["schedule"] = NAME + ".prepare.v1", schedule
    for path in (report, CHECK, trace, PROTOCOL, Path(__file__).resolve()):
        value["bindings"][str(path)] = sha(path)
    need(all(sha(path) == digest for path, digest in value["bindings"].items()), "source drift")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    args = parser.parse_args()
    experiment = prior.configure()
    engine.NAME, engine.build = NAME, build
    if args.command == "prepare":
        result = engine.prepare(args.output)
    else:
        prepared = json.loads(prior.PREP.read_bytes())
        original_data = engine.p3.source_data

        def with_goal(source):
            data = original_data(source)
            experiment.rebind(prepared, data)
            return data

        print("Execute fixed 24 candidate interventions; no neural inference.", flush=True)
        with patch.object(engine.p3, "source_data", with_goal):
            result = engine.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] not in ("prepared", "recorded_not_independently_consumed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
