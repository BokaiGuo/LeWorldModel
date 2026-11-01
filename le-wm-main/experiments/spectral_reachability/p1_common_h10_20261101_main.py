"""Fixed three-prefix x nine-tail physical intervention; no neural calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from . import codex_four_arm_prefix_intervention_20261030 as engine
from . import p1_three_prefix_own_host_20261101_main as own

NAME = "p1_common_h10_20261101_main"
ROOT = engine.ROOT
DOC = ROOT.parent / "docs/D2P_P1_COMMON_H10_PROTOCOL_20261101_MAIN.md"
ROLES = ("selected", "zero", "greedy")


def build(root, tag):
    bindings, rows, reports = {}, {}, {}
    for role in ROLES:
        directory = ROOT / f"results/{own.NAME}_root{root}_{role}_RUN_{tag}"
        path = directory / "report.json"
        report = json.loads(path.read_bytes())
        engine.require(report["status"] == "recorded_known_root_not_admitted"
                       and len(report["rows"]) == 1, "completed source")
        row = report["rows"][0]
        engine.require(row["initial_exact"] and row["first_error"] is None
                       and row["control_returned"] >= 5, "certified prefix")
        engine.require(engine.sha(directory / "journal.jsonl") == report["journal_sha256"],
                       "source journal")
        engine.journal_reader.journal((directory / "journal.jsonl").read_bytes(),
                                      report["journal_chain_tail"])
        for item in (path, directory / "journal.jsonl", directory / "execution_binding.json"):
            bindings[str(item)] = engine.sha(item)
        rows[role], reports[role] = row, report
    roots = [next(c for c in rows[r]["checkpoints"] if c["step"] == 0) for r in ROLES]
    for other in roots[1:]:
        engine.require(all(other[k] == roots[0][k] for k in
                           ("state", "snapshot", "pixel_sha256", "goal")), "same physical root")
    report, row = reports["selected"], rows["selected"]
    node = next(n for n in report["nodes"][row["episode_id"]] if n["boundary"] == 5)
    trace = Path(node["receipt"]["trace"])
    engine.require(engine.sha(trace) == node["receipt"]["trace_sha256"], "bank trace")
    events = [json.loads(line) for line in trace.read_bytes().splitlines()]
    payload = {e["kind"]: e["payload"] for e in events}
    bank = np.asarray(payload["optimized_candidates"]["raw"], dtype=np.float32)
    engine.require(bank.shape == (8, 3, 5, 2) and np.isfinite(bank).all()
                   and (abs(bank) <= 1).all(), "complete bounded bank")
    refinement = payload["winner_refinement"]
    if refinement["accepted"]:
        bank[refinement["index"]] = np.asarray(refinement["after_raw"], dtype=np.float32)
    tails = [bank[i, :2].reshape(10, 2) for i in range(8)] + [np.zeros((10, 2), np.float32)]
    schedule = []
    for role in ROLES:
        cp = next(c for c in rows[role]["checkpoints"] if c["step"] == 5)
        image = Path(cp["path"])
        engine.require(engine.sha(image) == cp["file_sha256"], "cp5 image")
        bindings[str(image)] = engine.sha(image)
        prefix = np.asarray(rows[role]["executed_raw"][:5], dtype=np.float32)
        if role == "zero":
            engine.require(np.array_equal(prefix, np.zeros((5, 2), np.float32)), "zero intervention")
        for i, tail in enumerate(tails):
            schedule.append(dict(index=len(schedule), trajectory_arm="structured", prefix_role=role,
                                 suffix_role=f"candidate_{i}" if i < 8 else "zero",
                                 commands=engine.actions(np.concatenate((prefix, tail)), 15),
                                 expected_prefix_checkpoint={k: deepcopy(cp[k]) for k in
                                                             ("state", "snapshot", "pixel_sha256")}))
    for path in (trace, DOC, Path(__file__).resolve(), Path(engine.__file__).resolve()):
        bindings[str(path)] = engine.sha(path)
    return dict(schema=NAME + ".prepare.v1", root=root, fixed_branches=27, schedule=schedule,
                bindings=bindings, neural_calls=0, P1_admission=False, independent_confirmation=False,
                selection="all eight selected-cp5 bank tails plus zero, shared by all prefixes",
                caveat="selected-context tail bank is asymmetric; not independent planning capacity")


def run(root, tag, output):
    experiment = own.setup(root, "selected")
    prepared = ROOT / f"results/{own.NAME}_root{root}_selected_PREP_{tag}.json"
    source = experiment.runner.verify(str(prepared), engine.sha(prepared))
    data = engine.p3.source_data(source["source"])
    experiment.rebind(source, data)
    ref = data["references"][root]
    plan = build(root, tag)
    plan["bindings"].update(source["sources"])
    plan["bindings"][str(prepared)] = engine.sha(prepared)
    engine.NAME = NAME
    engine.CAPS = {k: 9 * v for k, v in engine.p3.contract()["physical_caps"].items()}
    store = engine.Store(output)
    store.put("prepared.json", plan)
    start = time.monotonic()

    def check():
        engine.require(time.monotonic() - start < engine.WALL, "wall cap")
        store.check()

    meter = engine.Meter(store, check)
    report = dict(schema=NAME + ".report.v1", status="partial_or_failed", fixed_branches=27,
                  rows=[], errors=[], P1_admission=False, independent_confirmation=False)
    try:
        for branch in plan["schedule"]:
            row = deepcopy(engine.p3.roster()[0])
            row.update(episode_id=f"branch_{branch['index']:02}", branch=branch["index"],
                       prefix_role=branch["prefix_role"], suffix_role=branch["suffix_role"],
                       status="running", transitions=[], endpoint=None, prefix_verified=False)
            report["rows"].append(row)
            meter.episode, meter.region = row["episode_id"], "setup"
            env = None
            try:
                env = meter.call("constructor", lambda: engine.old.native_factory(meter))
                engine.episode(env, ref, data, branch, row, meter, store)
            finally:
                meter.region = "closing"
                if env is not None:
                    meter.call("close", env.close)
                store.put(row["episode_id"] + "_result.json", row)
            print(json.dumps(dict(root=root, branch=branch["index"], status=row["status"])), flush=True)
        engine.require(all(engine.sha(p) == h for p, h in plan["bindings"].items()), "source drift")
        engine.require(not meter.pending and not meter.secondary, "physical accounting")
        report["status"] = "recorded_not_independently_consumed"
    except BaseException as exc:
        report["errors"].append(dict(type=type(exc).__name__, message=str(exc)))
    finally:
        store.finalizing = True
        store.emit("run_finished", status=report["status"], counts=meter.counts)
        report.update(counts=meter.counts, journal_sha256=engine.sha(store.path / "journal.jsonl"),
                      journal_chain_tail=store.previous, wall_seconds=time.monotonic() - start)
        store.put("report.json", report, final=True)
    print(json.dumps(dict(output=str(output), status=report["status"])), flush=True)
    return report["status"] == "recorded_not_independently_consumed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    parser.add_argument("--tag", default="MAIN01")
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.root, args.tag, args.output) else 1)
