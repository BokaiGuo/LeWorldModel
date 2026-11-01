"""Physical replay of the complete wall-delivery panel, no neural execution."""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from experiments.spectral_reachability import codex_p3_online_budget_physical_host_20261030_v1 as host
from experiments.spectral_reachability import codex_p3_online_budget_physical_saved_20261030_v1 as saved
from experiments.spectral_reachability import codex_p3_resolved_host_20261030_v1 as producer
from experiments.spectral_reachability import codex_p3_queued_cost_batch_20261030_v1 as batch

ROOT = host.ROOT
STEM = "codex_p3_resolved_physical_20261030_v1"
BATCH = ROOT / "results/codex_p3_resolved_delivery_20261030_v1_RUN_CODEX01"
BATCH_SHA = "f94e83cbfde9927da4710e982f9129a9b90f6df23ec03dbcf3328c5176bdbdf8"
CONTRACT = ROOT / "results/codex_p3_resolved_delivery_20261030_v1_PREP_CODEX01.json"
CONTRACT_SHA = "d43f24099628a76f127f8ae37fff6be9a721003f3b6694e3a53638929d1c804e"


def check_batch(report, events):
    host.require(report["contract_sha256"] == CONTRACT_SHA and report["error"] is None and
                 report["scientific_pass"] is False and len(report["rows"]) == 6,
                 "wall batch terminal")
    rows = report["rows"]
    host.require([(r["point"], r["arm"]) for r in rows] == batch.SLOTS and
                 [e["row"] for e in events if e["kind"] == "slot_result"] == rows and
                 events[-1]["kind"] == "batch_terminal" and
                 events[-1]["completed_slots"] == 6 and events[-1]["failure"] is None,
                 "wall batch journal roster")
    host.require(all(r["delivered_and_saved"] is True and r["attempt"]["exit_code"] == 0 and
                     r["attempt"]["deadline_exceeded"] is False and
                     0 <= r["attempt"]["wall_seconds"] <= 30 for r in rows), "deadline delivery")
    host.require(any(next(r for r in rows if r["point"] == p and r["arm"] == "signal")["allocation"]
                     != next(r for r in rows if r["point"] == p and r["arm"] == "reverse")["allocation"]
                     for p in (20, 30)), "no resolved execution")
    return rows


def build(point):
    host.configure(point)
    host.require(host.sha(BATCH / "report.json") == BATCH_SHA and
                 host.sha(CONTRACT) == CONTRACT_SHA, "wall source hashes")
    report = host.helper.read(BATCH / "report.json")
    contract = host.helper.read(CONTRACT)
    host.require(contract["wall_seconds"] == 30 and contract["slots"] == [list(x) for x in batch.SLOTS],
                 "fixed wall contract")
    events = host.bind.host.journal.read(BATCH / "events.jsonl", report["journal_tail"], 2*1024**2)
    rows = check_batch(report, events)
    old_prepare, data = host.inputs()
    original = {(x["root_slot"], x["arm"]): x for x in old_prepare["plans"]}
    selected = []
    bindings = dict(old_prepare["bindings"])
    source_paths = [Path(__file__), ROOT / "tests/test_codex_p3_resolved_physical_20261030_v1.py",
                    ROOT / "innerdocs/codex_p3_resolved_physical_protocol_20261030_v1.md",
                    BATCH / "report.json", BATCH / "events.jsonl", CONTRACT]
    for root, arm in host.spec.ORDER:
        r = next(x for x in rows if x["point"] == point and x["arm"] == arm)
        run, receipt = Path(r["run"]), Path(r["receipt"])
        host.require(host.sha(run / "report.json") == r["report_sha256"] and
                     host.sha(receipt) == r["receipt_sha256"], "producer receipt hashes")
        # A clean process reconstructs the saved consumer directly from journal.
        program = ("from pathlib import Path; from experiments.spectral_reachability import "
                   "codex_p3_resolved_host_20261030_v1 as h; o=Path(%r); "
                   "r=h.prior.prior.read(o/'report.json'); "
                   "p,t,b=h.verify_prepare(o/'prepare.json',r['prepare_sha256']); "
                   "v=h.consumer.consume(o/'events.jsonl',r['journal_tail'],h.CAP,t,b,r['point'],r['arm']); "
                   "print(h.journal.digest(v))") % str(run)
        fresh = subprocess.check_output([sys.executable, "-B", "-c", program], text=True).strip().splitlines()[-1]
        host.require(fresh == r["receipt_sha256"], "fresh producer consumer")
        checked = host.helper.read(receipt)
        host.require(checked["status"] == "semantic_saved_checked_not_scientifically_admitted" and
                     len(checked["rows"]) == 1 and
                     (checked["rows"][0]["point"], checked["rows"][0]["arm"]) == (point, arm),
                     "saved slot identity")
        winner = checked["rows"][0]["winners"][root]
        raw = host.spec.raw(winner["commands"])
        origin = original[(root, arm)]
        host.require(origin["root_sha256"] == data["references"][root]["root_sha256"], "root identity")
        plan = dict(root_slot=root, arm=arm, root_sha256=origin["root_sha256"],
                    task_id=origin["task_id"], commands=raw.astype(float).tolist(),
                    source_candidate_sha256=host.kernel.source.digest(raw.astype(float).tolist()),
                    raw_float32_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
        plan["selection_digest"] = host.kernel.source.digest(plan)
        selected.append(plan)
        source_paths.extend([receipt, run / "report.json", run / "prepare.json", run / "events.jsonl"])
    for path in source_paths:
        path = Path(path).resolve()
        bindings[str(path)] = host.sha(path)
    host.NAME = STEM + "_P" + str(point)
    p = dict(schema=host.NAME + ".prepare", point=point, wall_batch_sha256=BATCH_SHA,
             wall_contract_sha256=CONTRACT_SHA, contract=old_prepare["contract"], plans=selected,
             bindings=bindings, physical_execution=False, scientific_pass=False)
    return p, data


def verify(path, expected, point):
    host.require(host.sha(path) == expected, "external prepare hash")
    p = host.helper.read(path)
    actual, data = build(point)
    host.require(p == actual, "physical prepare reconstruction")
    return p, data


def exclusive(path, prefix, point):
    p = Path(path)
    host.require(p.is_absolute() and p.parent == ROOT / "results" and
                 p.name.startswith(STEM + f"_P{point}_" + prefix) and not os.path.lexists(p),
                 "exclusive output")
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "consume"))
    parser.add_argument("--point", required=True, type=int, choices=(20, 30))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare")
    parser.add_argument("--sha256")
    parser.add_argument("--run")
    parser.add_argument("--report-sha")
    a = parser.parse_args()
    if a.action == "prepare":
        p, _ = build(a.point)
        print(host.bind.host.write_new(exclusive(a.output, "PREP_", a.point), p))
    elif a.action == "run":
        p, data = verify(a.prepare, a.sha256, a.point)
        out = exclusive(a.output, "RUN_", a.point)
        store = host.helper.Store(out)
        store.put("prepare.json", p)
        def factory():
            import gymnasium as gym
            import stable_worldmodel
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                            with_target=False, render_action=False)
        result = host.execute(p, data, store, factory)
        print(result["status"])
        host.require(result["status"] == "recorded_not_admitted", "partial preserved; no retry")
    else:
        p, data = verify(a.prepare, a.sha256, a.point)
        host.verify = lambda path, sha: (p, data) if host.sha(path) == sha else (_ for _ in ()).throw(ValueError("prepare"))
        result = saved.consume(a.run, a.report_sha, a.prepare, a.sha256)
        print(host.bind.host.write_new(exclusive(a.output, "CONSUME_", a.point), result))


if __name__ == "__main__":
    main()
