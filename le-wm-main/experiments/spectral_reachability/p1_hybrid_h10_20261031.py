"""Fixed existing H10 suffixes from the two sealed hybrid prefixes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

from . import p1_response_prefix_h10_20261031 as existing

ROOT = existing.ROOT
engine = existing.engine
need, sha = existing.need, existing.sha
NAME = "p1_hybrid_h10_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_HYBRID_H10_PROTOCOL_20261031.md"
RUNS = [
    (ROOT / "results/p1_hybrid_root0_20261031_RUN01", "315235810c69d1cc57b00bbb06c5d6fec381214d0a007c5f1d98048ce60f4141", 0),
    (ROOT / "results/p1_hybrid_root1_20261031_RUN02", "48db2f48c87c786d8a835a099c4a0ac009ad52e396ca768f4416bea6888ee3a4", 12),
]
CHECKS = [
    (ROOT / "results/p1_hybrid_root0_saved_20261031_RUN01.json", "5a6feb15e85e31c93f86ad1e904c023fa25ea0ee352465af666002dd51e4ac12"),
    (ROOT / "results/p1_hybrid_root1_saved_20261031_RUN02.json", "c2ae7caee38b0103e6964476e32cb534e3a2bbe52fce67127b2a0b8f8799dd06"),
]


def read(path):
    return json.loads(Path(path).read_bytes())


def build():
    need(sha(existing.BANKS) == existing.BANKS_SHA, "prior bank hash")
    old = read(existing.BANKS)
    anchors, banks = [], []
    bindings = {str(existing.BANKS): existing.BANKS_SHA}
    for path in (Path(__file__).resolve(), PROTOCOL, Path(engine.__file__).resolve()):
        bindings[str(path)] = sha(path)
    for (folder, digest, old_index), (check_path, check_sha) in zip(RUNS, CHECKS, strict=True):
        report_path = folder / "report.json"
        need(sha(report_path) == digest and sha(check_path) == check_sha, "hybrid sources")
        report, check = read(report_path), read(check_path)
        need(check["report_sha256"] == digest and check["nodes"] == 4
             and report["status"] == "recorded_known_root_not_admitted"
             and report["rows"][0]["status"] == "completed20", "completed source")
        row = report["rows"][0]
        journal = folder / "journal.jsonl"
        need(sha(journal) == report["journal_sha256"], "original journal hash")
        prefix = np.asarray(row["executed_raw"][:5], dtype="<f4")
        receipt = report["nodes"][row["episode_id"]][0]["receipt"]
        need(prefix.shape == (5, 2) and np.array_equal(prefix, np.asarray(receipt["commands"], np.float32)),
             "selected prefix binding")
        trace = [e["receipt"] for e in map(json.loads, journal.read_bytes().splitlines())
                 if e["kind"] == "fresh_observation" and
                 (e["receipt"]["phase"] == "warmup" or
                  (e["receipt"]["phase"] == "plan" and e["receipt"]["step"] <= 5))]
        need([(r["phase"], r["step"]) for r in trace] ==
             [("warmup", k) for k in range(11)] + [("plan", k) for k in range(6)], "complete source trace")
        for item in trace:
            image = Path(item["path"])
            need(image.parent == folder and sha(image) == item["file_sha256"], "source image hash")
            bindings[str(image)] = item["file_sha256"]
        anchor = deepcopy(old["anchors"][old_index])
        cp0, cp5 = row["checkpoints"][0], row["checkpoints"][1]
        need(anchor["root_state"] == cp0["state"] and anchor["goal"] == cp0["goal"]
             and cp5["step"] == 5, "matching root and task")
        anchor.update(prefix5_float32_hex=prefix.tobytes().hex(), checkpoint=cp5, hybrid_trace=trace)
        anchors.append(anchor)
        bank = old["probe_banks"][anchor["logical_pair"]]
        need(len(bank) == 42 and [p["index"] for p in bank] == list(range(42)), "fixed 42 bank")
        for p in bank:
            need(hashlib.sha256(bytes.fromhex(p["raw10_float32_hex"])).hexdigest() == p["raw10_sha256"],
                 "probe bytes")
        banks.append(bank)
        for path in (report_path, journal, check_path):
            bindings[str(path)] = sha(path)
    need(all(sha(p) == h for p, h in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "anchors": anchors, "banks": banks, "bindings": bindings,
            "fixed_anchors": 2, "fixed_branches": 84, "P1_admission": False,
            "independent_confirmation": False}


def compare_trace(row, anchor, store, meter):
    # Source uses a single off-overlay observation; engine additionally captures
    # its off/on/off triplets. Do not claim independent verification of the two
    # extra overlay frames against source observations that were never saved.
    need(len(row["trajectory"]) == len(anchor["hybrid_trace"]) == 17
         and len(store.observations) == 51, "complete replay trace")
    for actual, original in zip(row["trajectory"], anchor["hybrid_trace"], strict=True):
        need(all(actual[k] == original[k] for k in ("phase", "step", "pixel_sha256", "state", "snapshot")),
             "original single-overlay trace mismatch")
    cp = row["checkpoints"][-1]
    need(cp["step"] == 5 and all(cp[k] == anchor["checkpoint"][k]
         for k in ("state", "snapshot", "pixel_sha256")), "hybrid cp5 mismatch")
    store.emit("suffix_start_verified", **meter.where, checkpoint=cp,
               restoration="full replay with original single-overlay state and image binding",
               cache_identity="no snapshot restoration")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "run"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--prepared")
    ap.add_argument("--sha256")
    ap.add_argument("--authorize-physics", action="store_true")
    args = ap.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results" and output.name.startswith(NAME + "_")
         and not output.exists() and not output.is_symlink(), "exclusive output")
    fixed = build()
    if args.command == "prepare":
        print(json.dumps({"sha256": engine._write_new(output, fixed), "fixed_branches": 84}))
        return
    need(args.authorize_physics and args.prepared and sha(args.prepared) == args.sha256
         and read(args.prepared) == fixed and shutil.disk_usage(output.parent).free >= engine.FLOOR + 2*engine.CAP,
         "prepare, physics authorization, disk")
    _, data = existing.physical.bundle.physical.pilot.material(existing.physical.bundle.physical.JOIN_SHA)
    existing.physical.bundle.physical.prior.native.constructor_contract()
    output.mkdir(mode=0o700)
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    bindings = dict(fixed["bindings"], **{str(Path(args.prepared)): args.sha256})
    shards = []
    with patch.object(engine.prior, "compare_trace", compare_trace):
        for anchor, bank in zip(fixed["anchors"], fixed["banks"], strict=True):
            folder = output / f"anchor_{anchor['index']:02d}"
            result = engine.execute_shard(anchor, bank, data, folder, factory, bindings)
            shards.append({"index": anchor["index"], "status": result["status"],
                           "sha256": sha(folder / "report.json")})
            print(json.dumps(shards[-1]), flush=True)
            if result["status"] != "recorded_not_consumed":
                break
    status = "recorded_not_consumed" if len(shards) == 2 and all(
        s["status"] == "recorded_not_consumed" for s in shards) else "partial_or_failed"
    value = {"schema": NAME + ".run", "status": status, "shards": shards,
             "prepare_sha256": args.sha256, "P1_admission": False, "independent_confirmation": False}
    print(json.dumps({"status": status, "sha256": engine._write_new(output / "run_report.json", value)}), flush=True)
    if status != "recorded_not_consumed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
