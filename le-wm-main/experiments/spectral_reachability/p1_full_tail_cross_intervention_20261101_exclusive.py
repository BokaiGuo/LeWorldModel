"""Fixed full-tail cross intervention on the two known P1 development roots."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import codex_four_arm_prefix_intervention_20261030 as engine
from . import codex_p1_diag32_cp5_hold_20261101_exclusive as root0_hold
from . import codex_p1_diag32_cp5_live_20261101_exclusive as root0_live
from . import codex_p1_diag32_fixed_physical_20261101_exclusive as root0_source
from . import codex_p1_root1_candidate_binding_20261101_exclusive as root1_source
from . import e3_cp10_four_arm_root0_20261031 as root0_goal
from . import e3_cp10_four_arm_root1_20261031 as root1_goal
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import p1_candidate_intervention_20261101_main as physical

ROOT = engine.ROOT
NAME = "p1_full_tail_cross_intervention_20261101_exclusive"
ROOT0_LIVE = ROOT / "results/codex_p1_diag32_cp5_live_20261101_exclusive_run_CODEXLIVE01/report.json"
ROOT0_LIVE_SHA = "f8df21134c839ad5dcd0c3c2c3aa9e99601b7eae1e9446dc554a15b35a6a59be"
ROOT1_BINDING = ROOT / "results/codex_p1_root1_candidate_binding_20261101_exclusive_CODEX01.json"
ROOT1_BINDING_SHA = "2469d38911edec6777d20343b76dcecef93714fc841d9839ba123d2956446ee3"
ROLES = ("selected", "zero", "greedy")
ROOT0_ROLE = {"selected": "selected", "zero": "zero", "greedy": "same_bank_greedy"}


def canonical_raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest_actions(value):
    array = np.asarray(value, dtype="<f4")
    engine.require(array.shape == (10, 2) and np.isfinite(array).all()
                   and bool((np.abs(array) <= 1).all()), "bounded ten-step tail")
    return hashlib.sha256(array.tobytes()).hexdigest()


def unique_tail_bank(items):
    """Deduplicate only byte-identical actions while preserving all origin aliases."""
    bank = []
    by_digest = {}
    for label, actions in items:
        array = np.asarray(actions, dtype=np.float32).reshape(10, 2)
        digest = digest_actions(array)
        if digest not in by_digest:
            by_digest[digest] = len(bank)
            bank.append({"label": label, "aliases": [label], "sha256": digest,
                         "actions": array.tolist()})
        else:
            bank[by_digest[digest]]["aliases"].append(label)
    return bank


def schedule(root_slot, prefixes, tails):
    engine.require(tuple(prefixes) == ROLES, "fixed role order")
    rows = []
    for role in ROLES:
        prefix = np.asarray(prefixes[role], dtype=np.float32)
        engine.require(prefix.shape == (5, 2) and np.isfinite(prefix).all()
                       and bool((np.abs(prefix) <= 1).all()), "bounded prefix")
        for tail in tails:
            suffix = np.asarray(tail["actions"], dtype=np.float32)
            rows.append({
                "index": len(rows),
                "root_slot": root_slot,
                "candidate": ROLES.index(role),
                "prefix_role": role,
                "suffix_role": tail["label"],
                "suffix_aliases": tail["aliases"],
                "prefix": prefix.tolist(),
                "suffix": suffix.tolist(),
                "commands": np.concatenate((prefix, suffix)).astype(np.float32).tolist(),
                "prefix_sha256": hashlib.sha256(prefix.astype("<f4").tobytes()).hexdigest(),
                "suffix_sha256": tail["sha256"],
            })
    return rows


def root0_contract():
    engine.require(engine.sha(ROOT0_LIVE) == ROOT0_LIVE_SHA, "root0 live report SHA")
    hold = root0_hold.build()
    live = json.loads(ROOT0_LIVE.read_bytes())
    engine.require(live["status"] == "recorded_development" and not live["errors"], "root0 live status")
    live_rows = {row["role"]: row for row in live["rows"]}
    engine.require(set(live_rows) == set(root0_live.build()["roster"]), "root0 role roster")
    preseal = json.loads(root0_live.PRESEAL.read_bytes())
    prefixes = {role: preseal["prefixes"][ROOT0_ROLE[role]] for role in ROLES}
    optimized = {
        role: live_rows[ROOT0_ROLE[role]]["suffix"]["actions"]
        for role in ROLES
    }
    tails = unique_tail_bank([
        ("zero", np.zeros((10, 2), np.float32)),
        ("selected_hold", hold["hold_tails"]["selected"]["raw"]),
        ("zero_hold", hold["hold_tails"]["zero"]["raw"]),
        ("greedy_hold", hold["hold_tails"]["same_bank_greedy"]["raw"]),
        ("selected_optimized", optimized["selected"]),
        ("zero_optimized", optimized["zero"]),
        ("greedy_optimized", optimized["greedy"]),
    ])
    engine.require([tail["label"] for tail in tails] == [
        "zero", "selected_hold", "greedy_hold", "selected_optimized",
        "zero_optimized", "greedy_optimized"
    ], "root0 unique fixed tail order")
    cp5 = {}
    for role in ROLES:
        observations = live_rows[ROOT0_ROLE[role]]["observations"]
        matches = [x for x in observations if x["step"] == 5]
        engine.require(len(matches) == 1, "root0 cp5")
        cp5[role] = {k: matches[0][k] for k in ("state", "snapshot", "pixel_sha256")}
    prepared, _, root, _, _, _, _ = root0_source.inputs()
    return {
        "root_slot": 0,
        "source_prepared": str(root0_source.SOURCE_PREP),
        "source_prepare_sha256": engine.sha(root0_source.SOURCE_PREP),
        "goal_binding": prepared["cp10_identity"],
        "root": {k: root[k] for k in ("state", "snapshot", "pixel_sha256", "goal")},
        "prefixes": prefixes,
        "prefix_checks": cp5,
        "tails": tails,
        "schedule": schedule(0, prefixes, tails),
        "identity_note": "diag32 candidate4 iteration13, zero, and same-bank q1 greedy",
    }


def root1_contract():
    engine.require(engine.sha(ROOT1_BINDING) == ROOT1_BINDING_SHA, "root1 binding SHA")
    binding = json.loads(ROOT1_BINDING.read_bytes())
    engine.require(binding == root1_source.build() and binding["root_slot"] == 1, "root1 source closure")
    candidates = binding["candidates"]
    prefixes = {role: candidates[role]["prefix"] for role in ROLES}
    engine.require(np.array_equal(prefixes["selected"], prefixes["greedy"]), "root1 prefix alias")
    tails = unique_tail_bank([
        ("zero", np.zeros((10, 2), np.float32)),
        ("selected_hold", candidates["selected"]["hold"]),
        ("zero_hold", candidates["zero"]["hold"]),
        ("greedy_hold", candidates["greedy"]["hold"]),
        ("selected_optimized", candidates["selected"]["optimized"]),
        ("zero_optimized", candidates["zero"]["optimized"]),
        ("greedy_optimized", candidates["greedy"]["optimized"]),
    ])
    engine.require([tail["label"] for tail in tails] == [
        "zero", "selected_hold", "selected_optimized", "zero_optimized"
    ], "root1 unique fixed tail order")
    source_report = Path(candidates["selected"]["source"]["report"])
    report = json.loads(source_report.read_bytes())
    engine.require(engine.sha(source_report) == candidates["selected"]["source"]["report_sha256"],
                   "root1 report SHA")
    roots = [x for x in report["rows"][0]["checkpoints"] if x["step"] == 0]
    engine.require(len(roots) == 1 and roots[0]["pixel_sha256"] == binding["root_pixel_sha256"],
                   "root1 root checkpoint")
    return {
        "root_slot": 1,
        "source_prepared": report["prepared"],
        "source_prepare_sha256": report["prepare_sha256"],
        "goal_binding": binding["goal_binding"],
        "root": {k: roots[0][k] for k in ("state", "snapshot", "pixel_sha256", "goal")},
        "prefixes": prefixes,
        "prefix_checks": {
            role: {k: candidates[role]["required_cp5"][k]
                   for k in ("state", "snapshot", "pixel_sha256")}
            for role in ROLES
        },
        "tails": tails,
        "schedule": schedule(1, prefixes, tails),
        "identity_note": "root1 selected and greedy are byte-identical aliases; zero is distinct",
    }


def build():
    roots = [root0_contract(), root1_contract()]
    branches = [branch for root in roots for branch in root["schedule"]]
    for index, branch in enumerate(branches):
        branch["global_index"] = index
    bindings = {}
    for path in (
        Path(__file__).resolve(), Path(engine.__file__).resolve(),
        Path(physical.__file__).resolve(), Path(locking.__file__).resolve(),
        Path(root0_hold.__file__).resolve(), Path(root0_live.__file__).resolve(),
        Path(root0_source.__file__).resolve(), Path(root1_source.__file__).resolve(),
        Path(root0_goal.__file__).resolve(), Path(root1_goal.__file__).resolve(),
        ROOT0_LIVE, ROOT1_BINDING, root0_live.PRESEAL,
    ):
        bindings[str(path)] = engine.sha(path)
    for root in roots:
        prepared = Path(root["source_prepared"])
        engine.require(engine.sha(prepared) == root["source_prepare_sha256"], "prepared source SHA")
        bindings[str(prepared)] = root["source_prepare_sha256"]
    engine.require(all(engine.sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {
        "schema": NAME + ".protocol.v1",
        "roots": roots,
        "branches": branches,
        "planned_branches": 30,
        "intervention": "full Cartesian product of every byte-unique saved zero/hold/optimized tail and all three saved prefix roles",
        "deduplication": "byte-identical tails share aliases; no outcome-based filtering",
        "primary_estimand": "whole nonzero-tail bank paired means and success counts; no best-tail selector",
        "metrics": ["native_success", "native_distance", "object_position_error",
                    "wrapped_object_angle_error"],
        "known_development_roots": 2,
        "physical_outcomes_known_before_design": True,
        "outcomes_used_to_include_or_exclude_tails": False,
        "new_neural_calls": 0,
        "host_cuda_required_if_neural_calls_added": True,
        "independent_confirmation": False,
        "P1_admission": False,
        "bindings": bindings,
    }


def run(protocol_path, protocol_sha, output):
    engine.require(engine.sha(protocol_path) == protocol_sha
                   and json.loads(protocol_path.read_bytes()) == build(), "protocol drift")
    protocol = json.loads(protocol_path.read_bytes())
    root_data = {}
    for item in protocol["roots"]:
        prepared_path = Path(item["source_prepared"])
        prepared = json.loads(prepared_path.read_bytes())
        data = engine.p3.source_data(prepared["source"])
        rebound = (root0_goal.rebind(prepared, data) if item["root_slot"] == 0
                   else root1_goal.rebind(prepared, data))
        engine.require(rebound == item["goal_binding"], "cp10 goal rebind")
        root_data[item["root_slot"]] = (data, data["references"][item["root_slot"]])
    engine.NAME = NAME
    engine.CAPS = {key: 15 * value for key, value in engine.p3.contract()["physical_caps"].items()}
    store = engine.Store(output)
    store.put("protocol.json", protocol)
    started = time.monotonic()

    def check():
        engine.require(time.monotonic() - started < 3600, "wall cap")
        store.check()

    meter = engine.Meter(store, check)
    report = {
        "schema": NAME + ".report.v1",
        "status": "partial_or_failed",
        "rows": [],
        "errors": [],
        "planned_branches": protocol["planned_branches"],
        "protocol_sha256": protocol_sha,
        "new_neural_calls": 0,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    prefix_checks = {
        root["root_slot"]: {ROLES.index(role): value for role, value in root["prefix_checks"].items()}
        for root in protocol["roots"]
    }
    roots = {root["root_slot"]: root for root in protocol["roots"]}
    try:
        for branch in protocol["branches"]:
            row = deepcopy(engine.p3.roster()[branch["root_slot"]])
            row.update(
                episode_id=f"r{branch['root_slot']}_b{branch['index']:02d}",
                global_index=branch["global_index"],
                branch=branch["index"],
                source_slot=branch["root_slot"],
                candidate=branch["candidate"],
                prefix_role=branch["prefix_role"],
                suffix_role=branch["suffix_role"],
                suffix_aliases=branch["suffix_aliases"],
                status="running",
                transitions=[],
                endpoint=None,
                observations=[],
            )
            report["rows"].append(row)
            meter.episode, meter.region = row["episode_id"], "setup"
            env = None
            try:
                env = meter.call("constructor", lambda: engine.old.native_factory(meter))
                data, ref = root_data[branch["root_slot"]]
                physical.episode(env, ref, data, branch, row, meter, store,
                                 roots[branch["root_slot"]]["root"],
                                 prefix_checks[branch["root_slot"]])
            finally:
                meter.region = "closing"
                if env is not None:
                    meter.call("close", env.close)
                store.put(row["episode_id"] + "_result.json", row)
            print(json.dumps({
                "root": branch["root_slot"],
                "branch": branch["index"],
                "prefix": branch["prefix_role"],
                "tail": branch["suffix_role"],
                "status": row["status"],
            }), flush=True)
        engine.require(all(engine.sha(path) == digest for path, digest in protocol["bindings"].items()),
                       "post-run source drift")
        engine.require(not meter.pending and not meter.secondary, "physical accounting")
        report["status"] = "recorded_development"
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
    finally:
        store.finalizing = True
        store.emit("run_finished", status=report["status"], counts=meter.counts)
        report.update(
            counts=meter.counts,
            pending=meter.pending,
            secondary_errors=meter.secondary,
            journal_sha256=engine.sha(store.path / "journal.jsonl"),
            journal_chain_tail=store.previous,
            wall_seconds=time.monotonic() - started,
        )
        store.put("report.json", report, final=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--authorize-physical", action="store_true")
    args = parser.parse_args()
    engine.require(args.tag.isalnum(), "alphanumeric tag")
    protocol_path = ROOT / "results" / f"{NAME}_PROTOCOL_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_run_{args.tag}"
    if args.phase == "prepare":
        engine.require(args.sha256 is None and not os.path.lexists(protocol_path), "exclusive protocol")
        raw = canonical_raw(build())
        with protocol_path.open("xb") as stream:
            engine.require(stream.write(raw) == len(raw), "complete protocol write")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"protocol": str(protocol_path), "sha256": engine.sha(protocol_path)}))
        return 0
    engine.require(args.authorize_physical and args.sha256 is not None
                   and len(args.sha256) == 64 and not os.path.lexists(output), "run authorization")
    with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = run(protocol_path, args.sha256, output)
    print(json.dumps({"output": str(output), "status": report["status"],
                      "report_sha256": engine.sha(output / "report.json"),
                      "errors": report["errors"]}), flush=True)
    return int(report["status"] != "recorded_development")


if __name__ == "__main__":
    raise SystemExit(main())
