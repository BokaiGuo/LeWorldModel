"""Six fixed physical branches from the same known-root diag32 model bank."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from . import codex_p1_diag32_fixed_physical_20261101_exclusive as fixed
from . import p1_cp10_pair_common_tail_20261101_codex as pair

engine = fixed.engine
ROOT = fixed.ROOT
NAME = "p1_diag32_samebank_tail_20261101_codex"
DOC = ROOT.parent / "docs/D2P_P1_DIAG32_SAMEBANK_TAIL_20261101_CODEX.md"
PHYSICAL = ROOT / "results/codex_p1_diag32_fixed_physical_20261101_exclusive_run_CODEXPHYS01/report.json"
ZERO = ROOT / "results/p1_three_prefix_own_host_20261101_main_root0_zero_RUN_MAIN01/report.json"
ZERO_SHA = "43cc73eb263e018f94501c5712f15affe486ccd7262d0530aaa9c6e76bd9020e"
PHYSICAL_SHA = "01095a298ed35e8ccb6b68dbaa57f63e11f9fb6015e2e39cc36ee1325bcca330"


def choose_greedy(rows):
    engine.require(len(rows) == 264 and [(x["candidate"], x["iteration"]) for x in rows] ==
                   [(c, i) for c in range(8) for i in range(33)], "diag32 full pool")
    engine.require(all(np.isfinite(x["q1"]) for x in rows), "finite scores")
    return min(rows, key=lambda x: (x["q1"], x["candidate"], x["iteration"]))


def schedule(selected, greedy):
    selected = engine.actions(selected, 15)
    prefixes = (("selected", selected[:5]), ("zero", np.zeros((5, 2))),
                ("greedy", engine.actions(greedy, 15)[:5]))
    branches = []
    for candidate, (role, prefix) in enumerate(prefixes):
        prefix = engine.actions(prefix, 5)
        for tail_role, tail in (("common", selected[5:]), ("zero", np.zeros((10, 2)))):
            tail = engine.actions(tail, 10)
            branches.append(dict(index=len(branches), candidate=candidate, prefix_role=role,
                                 suffix_role=tail_role, prefix=prefix, suffix=tail,
                                 commands=engine.actions(prefix + tail, 15)))
    return branches


def build():
    require = engine.require
    require(engine.sha(PHYSICAL) == PHYSICAL_SHA and engine.sha(ZERO) == ZERO_SHA,
            "physical source SHA")
    contract = fixed.build()
    _, _, root, selected, chosen, _, _ = fixed.inputs()
    prior = json.loads(PHYSICAL.read_bytes())
    row = prior["row"]
    require(prior["status"] == "recorded_development" and not prior["errors"]
            and row["executed_raw"] == selected and len(row["observations"]) == 16,
            "selected physical identity")
    cp5 = row["observations"][5]
    require(cp5["step"] == 5 and all(row["observations"][0][k] == root[k]
            for k in ("state", "snapshot", "pixel_sha256", "goal")), "selected root")
    pair.EXPECTED[str(ZERO)] = ZERO_SHA
    zr, zrow, zp, zcp, _ = pair.read_report(ZERO)
    require(np.array_equal(np.asarray(zp), np.zeros((5, 2)))
            and all(zcp[0][k] == root[k] for k in ("state", "snapshot", "pixel_sha256", "goal")),
            "zero root and prefix")
    report = json.loads((fixed.DIAG_DIR / "report.json").read_bytes())
    greedy = choose_greedy(report["rows"][0]["evaluations"])
    bindings = dict(contract["bindings"])
    for path in (Path(__file__).resolve(), DOC, PHYSICAL, ZERO, ZERO.parent / "journal.jsonl",
                 Path(pair.implementation.__file__).resolve(), Path(pair.__file__).resolve()):
        bindings[str(path)] = engine.sha(path)
    require(all(engine.sha(p) == h for p, h in bindings.items()), "source closure")
    return dict(schema=NAME + ".prepare.v1",
                schedule=schedule(selected, np.asarray(greedy["actions"], np.float32).reshape(15, 2)),
                sources=bindings, source_prepared=str(fixed.SOURCE_PREP),
                goal_binding=contract["root_binding"], root=contract["root"],
                prefixes={str(i): {k: cp[k] for k in ("state", "snapshot", "pixel_sha256")}
                          for i, cp in ((0, cp5), (1, zcp[5]))},
                selected=dict(candidate=chosen["candidate"], iteration=chosen["iteration"],
                              q1=chosen["q1"], q3=chosen["q3"]),
                greedy=dict(candidate=greedy["candidate"], iteration=greedy["iteration"],
                            q1=greedy["q1"], q3=greedy["q3"], pool_size=264,
                            rule="min(q1,candidate,iteration) over 8x33 recorded plans"),
                physical_outcomes_known_before_design=True, P1_admission=False,
                independent_confirmation=False, neural_calls=0)


def run(output):
    data = build()
    prepared = json.loads(Path(data["source_prepared"]).read_bytes())
    context = engine.p3.source_data(prepared["source"])
    binding = fixed.cp10.rebind(prepared, context)
    engine.require(binding == data["goal_binding"], "task identity")
    ref = context["references"][0]
    engine.NAME = NAME
    engine.CAPS = {key: 3 * value for key, value in engine.p3.contract()["physical_caps"].items()}
    store = engine.Store(output)
    store.put("prepared.json", data)
    started = time.monotonic()
    def check():
        engine.require(time.monotonic() - started < engine.WALL, "wall cap")
        store.check()
    meter = engine.Meter(store, check)
    report = dict(status="partial_or_failed", rows=[], errors=[], planned_branches=6, neural_calls=0,
                  P1_admission=False, independent_confirmation=False,
                  dynamics_receipts="newly collected replays, not historical backfill")
    prefix_checks = {int(k): v for k, v in data["prefixes"].items()}
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
                pair.implementation.episode(env, ref, context, branch, row, meter, store,
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
    with fixed.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = run(args.output)
    print(json.dumps(dict(status=report["status"], errors=report["errors"])), flush=True)
    if report["status"] != "recorded_not_independently_consumed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
