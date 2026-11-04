"""Descriptive same-root FD cp5 comparisons from previously sealed events."""
import argparse
import json
from pathlib import Path

from . import reacher_pc_p1_fd_cp5_saved_20261102_codexexclusive as consumer
from . import reacher_pc_p1_fd_cp5_f64_host_20261102_codexexclusive as f64_host


def analyze(events):
    root_events = [x["payload"] for x in events if x["payload"]["kind"] == "root_returned"]
    if [x["seed"] for x in root_events] != [20263160, 20263161]:
        raise ValueError("incomplete root roster")
    rows = []
    for event in root_events:
        arms = event["result"]["arms"]
        if [(a["arm"], a["status"]) for a in arms] != [
                ("selected", "complete"), ("parameter_zero", "complete")]:
            raise ValueError("incomplete paired arms")
        selected, zero = (a["result"] for a in arms)
        s, z = selected["physical"], zero["physical"]
        if s["initial_qpos_distance"] != z["initial_qpos_distance"]:
            raise ValueError("paired initial distance differs")
        s5, z5 = s["after_first_chunk_qpos_distance"], z["after_first_chunk_qpos_distance"]
        sf, zf = s["final_qpos_distance"], z["final_qpos_distance"]
        rows.append({
            "seed": event["seed"], "initial": s["initial_qpos_distance"],
            "selected_cp5": s5, "parameter_zero_cp5": z5,
            "selected_final": sf, "parameter_zero_final": zf,
            "selected_minus_zero_cp5": s5 - z5,
            "selected_minus_zero_final": sf - zf,
            "suffix_change_difference": (sf - s5) - (zf - z5),
            "same_executed_suffix_raw": selected["action_witness_raw"][5:] == zero["action_witness_raw"][5:],
            "selected_suffix_seed": arms[0]["suffix_seed"],
            "parameter_zero_suffix_seed": arms[1]["suffix_seed"],
        })
    return {"status": "paired_development_description",
            "scope": "two exposed roots; not independent confirmation or P1 admission",
            "cost_scope": "32 CEM rollout candidates per arm; full cost not metered",
            "roots": rows}


def main():
    p = argparse.ArgumentParser()
    for name in ("plan", "plan-sha", "journal", "journal-sha", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    f64_host.configure()
    validated = consumer.consume(args.plan, args.plan_sha, args.journal, args.journal_sha)
    if validated["status"] != "complete_saved_consistency":
        raise ValueError("saved consumer did not certify all arms")
    with Path(args.journal).open() as stream:
        events = [json.loads(line) for line in stream]
    result = analyze(events)
    result.update(plan_sha256=args.plan_sha, journal_sha256=args.journal_sha)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
