"""Known-root fixed-prefix common-tail physical intervention, no neural calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from . import codex_four_arm_prefix_intervention_20261030 as engine
from . import p1_candidate_intervention_20261101_main as implementation
from . import e3_cp10_four_arm_root0_20261031 as cp10
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import codex_four_arm_consumer_20261030 as consumer

ROOT = engine.ROOT
NAME = "p1_cp10_pair_common_tail_20261101_codex"
DOC = ROOT.parent / "docs/D2P_P1_CP10_PAIR_COMMON_TAIL_20261101_CODEX.md"
STRUCT = ROOT / "results/codex_p1_common_structured_cp10_host_20261101_exclusive_RUN_CODEX01/report.json"
CEM = ROOT / "results/p1_common_budget_cem_host_20261101_leibniz_root0_RUN_LEIBNIZ01/report.json"
EXPECTED = {
    str(STRUCT): "6c49744a051ab327ca91c22f0aa5c19261eacff51c472fdf0b1cd4915d80f4cd",
    str(CEM): "28d1df63a0aa883256229d1a7fb5a9d2b78e96086f79b29ef9205703dee8a6c2",
}


def schedule(prefixes, suffix):
    engine.require(set(prefixes) == {"structured", "cem"}, "exact two-prefix roster")
    suffix = engine.actions(suffix, 10)
    zero = np.zeros((10, 2), np.float32).tolist()
    result = []
    for i, role in enumerate(("structured", "cem")):
        prefix = engine.actions(prefixes[role], 5)
        for tail, values in (("common", suffix), ("zero", zero)):
            result.append(dict(index=len(result), candidate=i, prefix_role=role, suffix_role=tail,
                               prefix=prefix, suffix=values, commands=engine.actions(prefix + values, 15)))
    return result


def read_report(path):
    engine.require(engine.sha(path) == EXPECTED[str(path)], "historical report SHA")
    report = json.loads(path.read_bytes())
    engine.require(report["status"] == "recorded_known_root_not_admitted"
                   and report["P1_admission"] is False and len(report["rows"]) == 1, "source status")
    row = report["rows"][0]
    engine.require(row["control_returned"] >= 5 and row["first_error"] is None
                   and row["initial_exact"] is True, "historical prefix available")
    journal = path.parent / "journal.jsonl"
    engine.require(engine.sha(journal) == report["journal_sha256"], "historical journal SHA")
    events = engine.journal_reader.journal(journal.read_bytes(), report["journal_chain_tail"])
    scope = [e for e in events if e.get("episode_id") == row["episode_id"]]
    issued = [e for e in scope if e["kind"] == "authorized_dispatch"]
    returned = [e for e in scope if e["kind"] == "control_return"]
    commands = engine.actions(row["executed_raw"][:5], 5)
    engine.require(len(issued) >= 5 and len(returned) >= 5 and all(
        a["step"] == b["step"] == i and a["command"] == b["command"] == commands[i - 1]
        and a["index"] < b["index"] for i, (a, b) in enumerate(zip(issued[:5], returned[:5]), 1)),
        "historical executed prefix")
    checkpoints = {step: [p for p in row["checkpoints"] if p["step"] == step] for step in (0, 5)}
    engine.require(all(len(x) == 1 for x in checkpoints.values()), "source checkpoint identity")
    return report, row, commands, {k: v[0] for k, v in checkpoints.items()}, events


def build():
    sr, srow, sp, scp, se = read_report(STRUCT)
    cr, crow, gp, gcp, _ = read_report(CEM)
    engine.require(sr["cp10_goal_binding"] == cr["cp10_goal_binding"]
                   and sr["model_before"] == sr["model_after"] == cr["model_before"] == cr["model_after"]
                   and all(scp[0][k] == gcp[0][k] for k in ("state", "snapshot", "pixel_sha256", "goal")),
                   "shared root/target/model")
    nodes = [x for x in sr["nodes"][srow["episode_id"]] if x["boundary"] == 5]
    engine.require(len(nodes) == 1 and nodes[0]["status"] == "sealed", "structured cp5 node")
    receipt = nodes[0]["receipt"]
    plan = np.asarray(receipt["selected_full_plan"], np.float32)
    engine.require(receipt["status"] == "complete" and plan.shape == (2, 5, 2)
                   and np.array_equal(plan[0], np.asarray(receipt["commands"], np.float32)),
                   "sealed structured cp5 full plan")
    seals = [e for e in se if e.get("episode_id") == srow["episode_id"]
             and e.get("kind") == "boundary_sealed" and e.get("boundary") == 5]
    dispatch = [e for e in se if e.get("episode_id") == srow["episode_id"]
                and e.get("kind") == "authorized_dispatch" and e.get("step") == 6]
    engine.require(len(seals) == len(dispatch) == 1 and seals[0]["index"] < dispatch[0]["index"]
                   and seals[0]["plan_sha256"] == engine.digest(receipt), "cp5 pre-dispatch seal")
    ledger = receipt["ledger"]
    account = consumer.consume(ledger["journal"], ledger["sha256"], receipt["account"]["contract_sha256"],
                               "structured", "node", "structured_b05")
    engine.require(account == receipt["account"], "source node ledger")
    trace = Path(receipt["trace"])
    engine.require(engine.sha(trace) == receipt["trace_sha256"], "source node trace SHA")
    prepared_path = Path(sr["prepared"])
    engine.require(engine.sha(prepared_path) == sr["prepare_sha256"], "prepared source SHA")
    prepared = json.loads(prepared_path.read_bytes())
    bindings = dict(prepared["sources"])
    bindings.update(EXPECTED)
    for path in (prepared_path, STRUCT.parent / "journal.jsonl", CEM.parent / "journal.jsonl",
                 trace, Path(ledger["journal"]), DOC, Path(__file__).resolve(),
                 Path(implementation.__file__).resolve(), Path(engine.__file__).resolve(),
                 Path(cp10.__file__).resolve(), Path(locking.__file__).resolve()):
        bindings[str(path)] = engine.sha(path)
    engine.require(all(engine.sha(path) == digest for path, digest in bindings.items()), "source closure drift")
    branches = schedule({"structured": sp, "cem": gp}, plan.reshape(10, 2))
    return dict(schema=NAME + ".prepare.v1", schedule=branches, sources=bindings,
        goal_binding=sr["cp10_goal_binding"], source_prepared=str(prepared_path),
        source_prepare_sha256=sr["prepare_sha256"],
        root={k: scp[0][k] for k in ("state", "snapshot", "pixel_sha256", "goal")},
        prefixes=[{k: cp[k] for k in ("state", "snapshot", "pixel_sha256")} for cp in (scp[5], gcp[5])],
        tail_source="structured boundary-5 sealed selected full plan, both packets",
        physical_outcomes_known_before_design=True, outcome_selected_for_this_run=False,
        P1_admission=False, independent_confirmation=False, neural_calls=0)


def run(output):
    data = build()
    prepared = json.loads(Path(data["source_prepared"]).read_bytes())
    context = engine.p3.source_data(prepared["source"])
    binding = cp10.rebind(prepared, context)
    engine.require(binding == data["goal_binding"], "cp10 task identity")
    ref = context["references"][0]
    engine.NAME = NAME
    engine.CAPS = {key: 2 * value for key, value in engine.p3.contract()["physical_caps"].items()}
    store = engine.Store(output)
    store.put("prepared.json", data)
    started = time.monotonic()
    def check():
        engine.require(time.monotonic() - started < engine.WALL, "wall cap")
        store.check()
    meter = engine.Meter(store, check)
    report = dict(status="partial_or_failed", rows=[], errors=[], planned_branches=4, neural_calls=0,
                  P1_admission=False, independent_confirmation=False,
                  dynamics_receipts="newly collected replays, not historical backfill")
    prefix_checks = {i: value for i, value in enumerate(data["prefixes"])}
    try:
        for branch in data["schedule"]:
            row = deepcopy(engine.p3.roster()[0])
            row.update(episode_id=f"branch_{branch['index']:02}", branch=branch["index"],
                       candidate=branch["candidate"], prefix_role=branch["prefix_role"],
                       suffix_role=branch["suffix_role"], status="running", transitions=[],
                       endpoint=None, observations=[])
            report["rows"].append(row)
            meter.episode, meter.region = row["episode_id"], "setup"
            env = None
            try:
                env = meter.call("constructor", lambda: engine.old.native_factory(meter))
                implementation.episode(env, ref, context, branch, row, meter, store,
                                       data["root"], prefix_checks)
            finally:
                meter.region = "closing"
                if env is not None:
                    meter.call("close", env.close)
                store.put(row["episode_id"] + "_result.json", row)
            print(json.dumps(dict(branch=branch["index"], prefix=branch["prefix_role"],
                                  tail=branch["suffix_role"], status=row["status"])), flush=True)
        engine.require(all(engine.sha(p) == h for p, h in data["sources"].items()), "source drift")
        engine.require(not meter.pending and not meter.secondary, "physical accounting")
        report["status"] = "recorded_not_independently_consumed"
    except BaseException as exc:
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)[:1000]))
    finally:
        store.finalizing = True
        store.emit("run_finished", status=report["status"], counts=meter.counts)
        report.update(counts=meter.counts, journal_sha256=engine.sha(store.path / "journal.jsonl"),
                      journal_chain_tail=store.previous, wall_seconds=time.monotonic() - started)
        store.put("report.json", report, final=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", action="store_true", required=True)
    args = parser.parse_args()
    with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = run(args.output)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    if report["status"] != "recorded_not_independently_consumed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
