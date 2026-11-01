"""Join sealed same-suffix interventions with separate real-cp5 H10 planning."""
import argparse
import csv
import json
from pathlib import Path

from . import p1_crossover_paired_table_20261031 as paired
from . import p1_packet_objective_fidelity_20261031 as fidelity
from . import p1_packet_feedback_crossover_20261031 as crossover

NAME = "p1_cp5_mechanism_planning_join_20261031"
ROOT = fidelity.ROOT
CROSS = ROOT / "results/p1_packet_feedback_crossover_consume_20261031_RUN02.json"
PLAN = fidelity.physical.PLAN
CONSUMED = fidelity.CONSUME
METRICS = paired.METRICS


def join(cross, anchors, entries, planned, consumed):
    if len(anchors) != 24 or len(entries) != 24 or len(planned) != 24 or len(consumed) != 24:
        raise ValueError("expected 24 complete planning rows")
    out = paired.build(cross)
    zero_rows = {(r["logical_pair"], r["role"]): r for r in out if r["probe"] == "zero_suffix"}
    for i, (anchor, entry, plan, result) in enumerate(zip(
            anchors, entries, planned, consumed, strict=True)):
        if not (i == anchor["index"] == entry["index"] == plan["index"] == result["index"]
                and anchor["role"] == entry["role"] == plan["role"] == result["role"]
                and anchor["logical_pair"] == entry["logical_pair"] == plan["logical_pair"]
                and anchor["root_slot"] == entry["root_slot"] == result["root_slot"]
                and anchor["condition"] == entry["condition"] == result["condition"]
                and anchor["arm"] == result["arm"]
                and anchor["prefix5_float32_hex"] == entry["blind"]["prefix_raw_f32_hex"]
                and anchor["checkpoint"]["pixel_sha256"] == entry["blind"]["cp5_pixel_sha256"]
                and result["status"] == "probe_completed" and result["returned_suffix"] == 10):
            raise ValueError("planning/crossover identity mismatch")
        if entry["action_sha256"] != plan["selected_action_sha256"]:
            raise ValueError("planned action mismatch")
        if plan["initial_f32_sha256"] != planned[3 * (i // 3)]["initial_f32_sha256"]:
            raise ValueError("unpaired initialization")
    for row in out:
        i = 3 * row["logical_pair"] + paired.ROLES.index(row["role"])
        physical = entries[i]["observations"][1]["physical"]
        row.update({"own_h10_status": consumed[i]["status"],
                    "own_h10_native_success": physical["native_success"],
                    "own_h10_action_sha256": entries[i]["action_sha256"],
                    "own_h10_init_sha256": planned[i]["initial_f32_sha256"],
                    "own_h10_planning_wall_seconds": planned[i]["planning_wall_seconds"]})
        for metric in METRICS:
            row["own_h10_" + metric] = physical[metric]
            baseline = zero_rows[(row["logical_pair"], row["role"])]["endpoint_" + metric]
            row["own_h10_delta_vs_zero_" + metric] = (
                None if baseline is None else physical[metric] - baseline)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not args.output.is_absolute() or args.output.parent != ROOT / "results"
            or not args.output.name.startswith(NAME + "_") or args.output.exists()
            or args.output.is_symlink()):
        raise ValueError("exclusive output")
    if crossover.sha(CROSS) != paired.EXPECTED:
        raise ValueError("crossover hash")
    if fidelity.base.sha(PLAN) != fidelity.physical.PLAN_SHA or fidelity.base.sha(CONSUMED) != fidelity.CONSUME_SHA:
        raise ValueError("planning source hash")
    fixed = crossover.build()
    entries, sources = fidelity.inputs()
    plan = json.loads(PLAN.read_bytes())
    result = json.loads(CONSUMED.read_bytes())
    rows = join(json.loads(CROSS.read_bytes()), fixed["anchors"], entries, plan["rows"], result["rows"])
    if any(fidelity.base.sha(path) != digest for path, digest in sources.items()):
        raise ValueError("source drift")
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "sha256": fidelity.base.sha(args.output)}), flush=True)


if __name__ == "__main__":
    main()
