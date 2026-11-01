"""Compare saved per-phase cooperative operation journals, not total costs."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from . import codex_four_arm_consumer_20261030 as consumer
from . import p1_uphill_complete_pairing_20261030 as pairing

ROOT = pairing.ROOT
SOURCES = (
    ("zero", "p1_uphill_cp5_physical20_20261030_prepare_RUN01.json",
     "7984bcd309a008ed63a4d0752be5726361b30e5042eb129cc42d72f7be1e42b9",
     "p1_uphill_cp5_cuda20_20261030_RUN01"),
    ("warm", "p1_uphill_cp5_warm_physical20_20261030_prepare_RUN01.json",
     "5a8eed8cc3bbca6e496121ed6a6bca2fc99b130e4a3173fa6c9ca9dd319264b5",
     "p1_uphill_cp5_warm_cuda20_20261030_RUN01"),
)
IDENTITY = ("index", "logical_pair", "root_slot", "condition", "arm", "role",
            "task_id", "cp5_pixel_sha256", "prefix5_float32_hex")
require = pairing.require


def journal_counts(raw):
    """Count successfully returned dispatches by operation and phase."""
    rows = consumer._parse(raw)
    pending = None
    counts = collections.Counter()
    for row in rows[1:-1]:
        if row["kind"] == "intent":
            require(pending is None, "overlapping operation")
            pending = (row["payload"]["phase"], row["payload"]["kind"])
        elif row["kind"] == "return":
            require(pending is not None, "orphan return")
            counts[pending] += 1
            pending = None
        else:
            raise ValueError("non-success event in completed journal")
    require(pending is None, "unfinished operation")
    return [{"phase": phase, "operation": operation, "returned_dispatches": count}
            for (phase, operation), count in sorted(counts.items())]


def read_source(source):
    label, prep_name, prep_sha, plan_name = source
    prep_path = ROOT / "results" / prep_name
    prep = pairing.read_checked(prep_path, prep_sha)
    require(prep["fixed_rows"] == 24 and len(prep["roster"]) == 24, "fixed roster")
    rows = []
    for index, entry in enumerate(prep["roster"]):
        require(entry["index"] == index, "roster order")
        result_path = ROOT / "results" / plan_name / f"row_{index:02d}_result.json"
        require(prep["bindings"][str(result_path)] == entry["plan_row_sha256"],
                "plan result bound")
        result = pairing.read_checked(result_path, entry["plan_row_sha256"])
        require(result["index"] == result["result"]["index"] == index
                and result["kind"] == "row_complete", "row result identity")
        ledger = result["result"]["ledger_receipt"]
        plan = ledger["planning"]
        journal = ROOT / "results" / plan_name / f"row_{index:02d}" / "planning/events.jsonl"
        checked = consumer.consume(journal, plan["journal_sha256"], plan["contract_sha256"],
                                   "structured", "fixed", f"row_{index:02d}")
        require(checked == plan and ledger["complete"] is True, "ledger receipt binding")
        raw = journal.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == plan["journal_sha256"], "journal drift")
        counts = journal_counts(raw)
        require(sum(x["returned_dispatches"] for x in counts) == checked["completed_events"],
                "dispatch count")
        rows.append({"identity": {key: entry[key] for key in IDENTITY},
                     "score": result["result"]["score"], "plan_row_sha256": entry["plan_row_sha256"],
                     "journal_sha256": plan["journal_sha256"],
                     "returned_dispatches": counts,
                     "charged_upper_bounds": plan["charged_upper_bounds"],
                     "encoder_attempted": ledger["encoding"]["attempted"],
                     "coverage_verified": False, "actual_usage_verified": False})
    require(hashlib.sha256(prep_path.read_bytes()).hexdigest() == prep_sha, "prepare drift")
    return label, prep_sha, rows


def paired(sources):
    require(len(sources) == 2 and sources[0][0] == "zero" and sources[1][0] == "warm",
            "source order")
    out = []
    for zero, warm in zip(sources[0][2], sources[1][2], strict=True):
        require(zero["identity"] == warm["identity"], "paired identity mismatch")
        out.append({"identity": zero["identity"], "zero": zero, "warm": warm})
    require(len(out) == 24, "paired roster")
    return {"schema": "p1_uphill_cp5_ledger_pairs_20261030", "rows": out,
            "source_prepare_sha256": {s[0]: s[1] for s in sources},
            "cooperative_dispatch_only": True, "coverage_verified": False,
            "actual_usage_verified": False, "equal_budget_certified": False,
            "P1_admission": False,
            "limitations": "Returns are cooperative dispatch events; cannot establish hook coverage, "
                           "total computational cost, wall time, setup or prefix-generation cost."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(output.is_absolute() and output.parent == ROOT / "results"
            and output.name.startswith("p1_uphill_cp5_ledger_pairs_")
            and not output.exists() and not output.is_symlink(), "exclusive output")
    value = paired([read_source(source) for source in SOURCES])
    value["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "paired_rows": len(value["rows"])}))


if __name__ == "__main__":
    main()
