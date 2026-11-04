"""Known-root P3 winner to physical-source binding; no environment execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from . import p3_venv_live_host_20261102_codexexclusive as host
from . import codex_p3_quota_journal_20261030_v1 as journal
from . import moving_history_budget_scheduler_cuda_20261027_v1 as source
from . import p1_moving_baseline_physical_20261027_v1 as physical

ROOT = host.ROOT
NAME = "p3_venv_winner_physical_prepare_20261102_codexexclusive"
RUN = ROOT / "results/p3_venv_live_host_20261102_codexexclusive_RUN_01"
CONSUME = ROOT / "results/p3_venv_live_host_20261102_codexexclusive_CONSUME_01.json"
REPORT_SHA = "19425dc87593b4354f93a391246f53fd227c0d9be26bdb85f51c6e4b72e3743f"
PREPARE_SHA = "8e0a8010a35750dddfe1d1f9b172b9aedfd474c7206def74767efded6b22f11f"
CONSUME_SHA = "1e7bf6eaff9c457cb96db645e8df2ecf446e155c032fb992414242c1ae200a21"


def require(value, message):
    if not value:
        raise ValueError(message)


def join(checked, slots, refs, task_bridge, raw_commands):
    require(len(slots) == len(refs) == 2 and len(checked["rows"]) == 2
            and [r["arm"] for r in checked["rows"]] == ["signal", "reverse"],
            "fixed two-arm two-root roster")
    bindings, plans = {}, []
    for root, (slot, ref) in enumerate(zip(slots, refs, strict=True)):
        require(slot["source_slot"] == root and slot["root_sha256"] == ref["root_sha256"],
                "source root binding")
        task_bridge(slot, ref, bindings)
    for row in checked["rows"]:
        require(len(row["winners"]) == 2, "two fixed winners")
        for root in (0, 1):
            winner, slot = row["winners"][root], slots[root]
            raw = raw_commands(winner["commands"])
            require(raw.shape == (10, 2) and str(raw.dtype) == "float32", "raw action chart")
            plan = {"arm": row["arm"], "root_slot": root, "root_sha256": slot["root_sha256"],
                    "task_id": slot["task_id"], "task_export_sha256": slot["task_sha256"],
                    "candidate_index": winner["index"], "model_cost": winner["cost"],
                    "allocation": row["allocation"], "commands": raw.astype(float).tolist(),
                    "raw_float32_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
            plan["selection_sha256"] = journal.digest(plan)
            plans.append(plan)
    return plans, bindings


def build():
    require(host.host.prior.sha(CONSUME) == CONSUME_SHA, "external consumer SHA")
    checked = host.consume(RUN, REPORT_SHA, PREPARE_SHA)
    require(journal.digest(checked) == CONSUME_SHA
            and checked["status"] == "saved_semantics_checked_known_roots", "saved consumer reconstruction")
    require(host.host.prior.sha(host.host.PREVIOUS) == host.PREDECESSOR_SHA, "predecessor SHA")
    slots = host.host.prior.prior.read(host.host.PREVIOUS)["slots"]
    data = physical.kernel.load_source(source.SOURCEPACK, source.SOURCEPACK_SHA,
                                       source.SOURCE_PREPARE, source.SOURCE_PREPARE_SHA)
    plans, bindings = join(checked, slots, data["references"], physical.task_bridge,
                           host.host.prior.prior.raw_commands)
    return {"schema": NAME + ".v1", "status": "source_join_only_not_execution_ready",
            "run_report_sha256": REPORT_SHA, "run_prepare_sha256": PREPARE_SHA,
            "consumer_sha256": CONSUME_SHA, "sourcepack_sha256": source.SOURCEPACK_SHA,
            "source_prepare_sha256": source.SOURCE_PREPARE_SHA, "task_bindings": bindings,
            "plans": plans, "physical_execution": False, "physical_authorized": False,
            "warmup_and_reset_execution_verified": False, "native_environment_identity": "unknown",
            "old_new_environment_equivalence": "unknown", "independent_outcome": False,
            "P3_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    require(path.is_absolute() and path.parent == ROOT / "results" and
            path.name.startswith(NAME + "_") and not os.path.lexists(path), "exclusive output")
    print(json.dumps({"path": str(path), "sha256": host.host.write_new(path, build())}))


if __name__ == "__main__":
    main()
