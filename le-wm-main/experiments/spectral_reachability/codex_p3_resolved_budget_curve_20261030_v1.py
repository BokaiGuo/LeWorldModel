"""Descriptive two-point full-denominator budget curve from the fixed P3 run.

The two points have common per-axis ceilings within a point, not equal total
work. Missing work is unknown, never zero or imputed. No GPU or physics.
"""
import argparse
import json
from pathlib import Path

from . import codex_p3_resolved_batch_audit_20261030_v1 as audit

ROOT = audit.ROOT
NAME = "codex_p3_resolved_budget_curve_20261030_v1"
RUN = ROOT/"results/codex_p3_resolved_delivery_20261030_v1_RUN_CODEX01"
PREP = ROOT/"results/codex_p3_resolved_host_20261030_v1_PREP_CODEX01.json"
CONTRACT = ROOT/"results/codex_p3_resolved_delivery_20261030_v1_PREP_CODEX01.json"
RUN_SHA = "f94e83cbfde9927da4710e982f9129a9b90f6df23ec03dbcf3328c5176bdbdf8"
PREP_SHA = "14429025e33fe6f5c1927957d80476edf7af2fe63afe1227a4162ade9abff1bd"
CONTRACT_SHA = "d43f24099628a76f127f8ae37fff6be9a721003f3b6694e3a53638929d1c804e"


def assemble(checked, batch, prep, contract, receipt_loader):
    slots = list(audit.delivery.SLOTS)
    if (checked["status"] not in ("complete", "partial_or_failed") or
            [(r["point"], r["arm"]) for r in checked["rows"]] != slots or
            [(r["point"], r["arm"]) for r in batch["rows"]] != slots or
            contract["wall_seconds"] != 30 or contract["slots"] != [list(x) for x in slots]):
        raise ValueError("fixed roster or caps")
    points = []
    for point in (20, 30):
        caps = prep["per_arm_caps"][str(point)]
        arms = []
        for item, source in zip(checked["rows"], batch["rows"], strict=True):
            if item["point"] != point:
                continue
            status = item["delivered_and_saved"]
            row = {"arm": item["arm"], "status": "consumed" if status else "missing_or_failed",
                   "wall_seconds": item["process_wall_seconds"],
                   "axis_caps": caps, "axis_attempted": None, "axis_returned": None,
                   "allocation": None, "root_model_costs": None, "failure": None}
            if status:
                receipt = receipt_loader(source)
                actual = receipt["rows"][0]
                counts = actual["attempted"]
                if (set(counts) != set(caps) or any(type(counts[k]) is not int or
                        not 0 <= counts[k] <= caps[k] for k in caps)):
                    raise ValueError("per-axis caps")
                if actual["allocation"] != item["candidate_counts"]:
                    raise ValueError("effective allocation")
                row.update(axis_attempted=counts, axis_returned=dict(counts),
                           allocation=actual["allocation"],
                           root_model_costs=[w["cost"] for w in actual["winners"]],
                           scheduler_meter_seconds=item["scheduler_meter_seconds"])
            else:
                row["failure"] = {"deadline_exceeded": source["attempt"]["deadline_exceeded"],
                                  "exit_code": source["attempt"]["exit_code"],
                                  "unobserved_work": "unknown"}
            arms.append(row)
        if [r["arm"] for r in arms] != ["uniform", "signal", "reverse"]:
            raise ValueError("arm order")
        points.append({"budget_point": point, "arms": arms,
                       "full_roster_consumed": all(r["status"] == "consumed" for r in arms)})
    return {"schema": NAME, "source_batch_sha256": RUN_SHA, "source_prepare_sha256": PREP_SHA,
            "source_contract_sha256": CONTRACT_SHA, "points": points,
            "interpretation": "two-point observed model-cost curve; known roots; not physical outcomes",
            "resource_policy": "common vector upper bounds, no cross-axis sum, no forced padding",
            "execution_contract": "fixed full candidate quotas; early-stop or underuse is not accepted by the existing kernel",
            "wall_scope": "30 seconds process start to return including lock; saved consume excluded",
            "per_child_internal_wall_cap_seconds": prep["wall_seconds"],
            "scheduler_scope": "dispatch through sync and limit checks",
            "full_total_compute_equal": False, "flops": "unknown", "energy": "unknown",
            "model_arrays_independently_recomputed": False, "independent_confirmation": False,
            "P3_admission": False}


def build():
    if audit.sha(PREP) != PREP_SHA or audit.sha(CONTRACT) != CONTRACT_SHA:
        raise ValueError("fixed preparation hash")
    checked = audit.build(RUN, RUN_SHA)
    batch = json.loads((RUN/"report.json").read_bytes())
    prep = json.loads(PREP.read_bytes())
    contract = json.loads(CONTRACT.read_bytes())
    return assemble(checked, batch, prep, contract,
                    lambda source: json.loads(Path(source["receipt"]).read_bytes()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    output = Path(p.parse_args().output)
    if not (output.is_absolute() and output.parent == ROOT/"results" and output.name.startswith(NAME+"_")):
        raise ValueError("exclusive output")
    result = build()
    print(json.dumps({"sha256": audit.delivery.host.write_new(output, result),
                      "full_points": sum(p["full_roster_consumed"] for p in result["points"])}))


if __name__ == "__main__":
    main()
