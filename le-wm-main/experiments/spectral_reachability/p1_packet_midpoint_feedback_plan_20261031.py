"""Replan all packet-objective final packets from verified real midpoints."""
import argparse
from pathlib import Path

from . import p1_packet_objective_fidelity_20261031 as source
from . import p1_uphill_midpoint_feedback_plan_20261031_main as planner

NAME = "p1_packet_midpoint_feedback_plan_20261031"
PROTOCOL = source.ROOT.parent / "docs/D2P_P1_PACKET_MIDPOINT_FEEDBACK_PROTOCOL_20261031.md"


def inputs():
    entries, sources = source.inputs()
    report = planner.rgb.paired.read(source.RUN / "run_report.json", source.RUN_SHA)
    result = []
    for e, shard in zip(entries, report["shards"], strict=True):
        i = e["index"]
        folder = source.RUN / f"row_{i:02d}"
        events = planner.traces.transitions(folder, shard["report_sha256"])
        if len(events) < 5 or any(t["step"] != j + 1
                                  or t["actual_bits"] != e["raw"][0, j].tobytes().hex()
                                  for j, t in enumerate(events[:5])):
            raise ValueError("executed first packet")
        saved = planner.rgb.paired.read(folder / "report.json", shard["report_sha256"])
        frame = source.earlier.frame_at(saved["row"], 5, e["condition"])
        if frame["state"] != events[4]["state"]:
            raise ValueError("midpoint transition")
        sources[str(folder / "journal.jsonl")] = saved["journal_sha256"]
        result.append((e["blind"], {"index": i, "role": e["role"]}, e["raw"],
                       e["observations"][0]["pixels"], frame))
    if len(result) != 24:
        raise ValueError("fixed roster")
    sources[str(Path(__file__).resolve())] = source.base.sha(__file__)
    return result, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    planner.NAME = NAME
    planner.PROTOCOL = PROTOCOL
    planner.inputs = inputs
    print("Replan all 24 final packets from verified real midpoint RGB on host CUDA.", flush=True)
    planner.run(Path(parser.parse_args().output))


if __name__ == "__main__":
    main()
