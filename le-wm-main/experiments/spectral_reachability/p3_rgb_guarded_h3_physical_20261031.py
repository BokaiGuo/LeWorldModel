"""Replay all fixed real-context replans with an enforced physical cp15 guard."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from . import p3_h3_real_context_replay_20261031 as replay
from . import p3_raw_h3_physical_source_20261031 as old
from . import p3_rgb_guarded_h3_batch_20261031 as guarded_batch
from . import p3_rgb_guarded_h3_saved_20261031 as guarded_saved
from . import p3_rgb_edge_preflight_20261031 as edge

ROOT, host, batch, need = old.ROOT, old.host, guarded_batch, old.require
NAME = "p3_rgb_guarded_h3_physical_20261031"
PLAN = ROOT / "results/p3_h3_real_context_replan_20261031_RUN01.json"
PLAN_SHA = "7abd67ecf6d87885588062134b6114c8a614b7adc438e599a63e207a3385bdaf"
PROTOCOL = ROOT.parent / "docs/D2P_P3_RGB_GUARDED_BRANCH_PROTOCOL_20261031.md"
GUARD_KEYS = ("state", "snapshot", "pixel_sha256")


def build():
    need(host.sha(PLAN) == PLAN_SHA, "replan hash")
    plan = json.loads(PLAN.read_bytes())
    entries, excluded, bindings = replay.collect()
    need(plan["status"] == "complete" and len(entries) == len(plan["rows"]) == 52
         and excluded == plan["excluded"], "fixed roster")
    _, base_bindings, _, data = old.prepared.h1.prepare()
    original = json.loads(old.MANIFEST.read_bytes())
    need(host.sha(old.MANIFEST) == old.MANIFEST_SHA and len(original["plans"]) == 72,
         "original physical manifest")
    for path, digest in {**base_bindings, **plan["sources"]}.items():
        need(path not in bindings or bindings[path] == digest, "binding conflict")
        bindings[path] = digest
    for path in (PLAN, old.MANIFEST, PROTOCOL, Path(__file__).resolve(),
                 Path(batch.__file__).resolve(), Path(batch.branch.__file__).resolve(),
                 Path(guarded_saved.__file__).resolve(), Path(edge.__file__).resolve(),
                 Path(edge.features.__file__).resolve()):
        bindings[str(path)] = host.sha(path)
    pairs = []
    for entry, selected in zip(entries, plan["rows"], strict=True):
        logical, variant = entry["logical_pair"], entry["variant"]
        need(all(selected[k] == entry[k] for k in ("logical_pair", "variant"))
             and selected["decision_index"] == entry["key"][0]
             and selected["candidate"] == entry["key"][1]
             and selected["policy"] == entry["key"][2]
             and selected["root_slot"] == entry["key"][3], "replan row identity")
        offset = 0 if variant == "old" else 1
        source_plan = original["plans"][2 * logical + offset]
        source_actions = np.asarray(source_plan["commands"], dtype="<f4")
        prefix = bytes.fromhex(selected["prefix15_float32_hex"])
        suffix = bytes.fromhex(selected["chosen_suffix_float32_hex"])
        need(source_actions.shape == (20, 2) and source_actions[:15].tobytes() == prefix
             and source_actions[15:].tobytes().hex() == selected["original_suffix_float32_hex"]
             and len(suffix) == 40
             and hashlib.sha256(prefix + suffix).hexdigest() == selected["action_sha256"],
             "physical action binding")
        b, pair = divmod(logical, 3)
        path = ROOT / f"results/p3_raw_h3_physical_run_20261031_BATCH{b:02d}_RUN01/report.json"
        report = json.loads(path.read_bytes())
        old_row = report["rows"][2 * pair + offset]
        checkpoints = [c for c in old_row["checkpoints"] if c["phase"] == "plan" and c["step"] == 15]
        need(len(checkpoints) == 1
             and checkpoints[0]["pixel_sha256"] == selected["context_pixel_sha256"][2],
             "source cp15")
        guard = {key: checkpoints[0][key] for key in GUARD_KEYS}
        actions = np.frombuffer(prefix + suffix, dtype="<f4").reshape(20, 2)
        need(np.isfinite(actions).all() and (np.abs(actions) <= 1).all(), "action bounds")
        baseline = {**source_plan, "variant": "old"}
        baseline["selection_digest"] = batch.kernel.source.digest(
            {k: v for k, v in baseline.items() if k != "selection_digest"})
        revised = {**baseline, "variant": "new", "commands": actions.astype(float).tolist(),
                   "raw_float32_sha256": selected["action_sha256"]}
        revised["selection_digest"] = batch.kernel.source.digest(
            {k: v for k, v in revised.items() if k != "selection_digest"})
        pairs.append({"logical_pair": logical, "source_variant": variant, "plans": [baseline, revised],
                      "guard": guard, "padding": False})
    padded = pairs + [{**pairs[i], "padding": True} for i in (0, 1)]
    need(len(padded) == 54, "batch padding")
    for i in range(18):
        batch.validate_plans([p for item in padded[3*i:3*i+3] for p in item["plans"]])
    need(all(host.sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".manifest", "pairs": padded, "fixed_pairs": 52, "padding_pairs": 2,
            "excluded": excluded, "bindings": bindings, "physical_execution": False,
            "P1_admission": False, "P3_admission": False}, data


def prepare_batch(manifest, index):
    need(type(index) is int and 0 <= index < 18, "batch index")
    items = manifest["pairs"][3*index:3*index+3]
    plans = [p for item in items for p in item["plans"]]
    batch.validate_plans(plans)
    return {"schema": batch.NAME + ".prepare", "contract": batch.contract(), "plans": plans,
            "bindings": manifest["bindings"], "batch_index": index,
            "guards": [item["guard"] for item in items for _ in range(2)],
            "identities": [{k: item[k] for k in ("logical_pair", "source_variant", "padding")}
                           for item in items for _ in range(2)],
            "physical_execution": False, "scientific_pass": False}


def execute_guarded(p, data, directory):
    require_disk = shutil.disk_usage(directory.parent).free >= host.helper.CAP + host.helper.FLOOR
    need(require_disk, "disk reservation")
    store = host.helper.Store(directory)
    store.put("prepare.json", p)
    original_capture = batch.branch.capture
    matched = []

    def capture(env, obs, phase, step, row, ops, store, index):
        frame, snapshot = original_capture(env, obs, phase, step, row, ops, store, index)
        if phase == "plan" and step == 15:
            record = row["checkpoints"][-1]
            need(all(record[key] == p["guards"][index][key] for key in GUARD_KEYS),
                 "cp15 mismatch before suffix dispatch")
            store.emit("cp15_guard_passed", branch=index, pixel_sha256=record["pixel_sha256"])
            matched.append(index)
        return frame, snapshot

    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                        with_target=False, render_action=False)

    batch.branch.capture = capture
    try:
        result = batch.execute(p, data, store, factory)
    finally:
        batch.branch.capture = original_capture
    if result["status"] == "recorded_not_admitted":
        need(matched == list(range(6)), "all cp15 guards")
        checked = guarded_saved.consume(directory, host.sha(directory / "report.json"),
                                        host.sha(directory / "prepare.json"), p, data["references"])
        report = json.loads((directory / "report.json").read_bytes())
        need(all(all(next(c for c in row["checkpoints"] if c["phase"] == "plan" and c["step"] == 15)[k]
                     == p["guards"][i][k] for k in GUARD_KEYS)
                 for i, row in enumerate(report["rows"])), "saved cp15 guard")
        return {"status": checked["status"], "report_sha256": host.sha(directory / "report.json"),
                "prepare_sha256": host.sha(directory / "prepare.json"), "batch": p["batch_index"]}
    return {"status": result["status"], "batch": p["batch_index"],
            "report_sha256": host.sha(directory / "report.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--sha256")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
         and not os.path.lexists(out), "exclusive output")
    manifest, data = build()
    if args.command == "prepare":
        raw = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with out.open("xb") as stream:
            stream.write(raw)
        result = {"status": "prepared", "sha256": hashlib.sha256(raw).hexdigest(), "pairs": 52}
    else:
        need(args.authorize_physics and args.manifest and args.sha256
             and host.sha(args.manifest) == args.sha256
             and json.loads(Path(args.manifest).read_bytes()) == manifest, "manifest authority")
        result = execute_guarded(prepare_batch(manifest, args.batch), data, out)
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(result["status"] == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
