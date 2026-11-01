"""Execute and consume all fixed split-budget feedback crossover branches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_packet_feedback_crossover_20261031 as engine
from . import p1_packet_feedback_crossover_consume_20261031 as consumer
from . import p1_split_budget_feedback_physical_20261031 as split

ROOT = split.ROOT
NAME = "p1_split_budget_crossover_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_SPLIT_BUDGET_CROSSOVER_PROTOCOL_20261031.md"
PREP = ROOT / "results/p1_split_budget_feedback_physical_20261031_prepare_RUN01.json"
CONSUME = ROOT / "results/p1_split_budget_feedback_physical_20261031_consume_RUN01.json"
ANALYSIS = ROOT / "results/p1_split_budget_feedback_analysis_20261031_RUN01.json"
PREP_SHA = "0449e00529be43f81fb244fdade7ce06afdb640bb7965065786c6f7ff91f65be"
CONSUME_SHA = "5e9fc8f802e7c0c4fb4adb6b2dfea556b7fdf474142d8b342a0a1ae0226e32d6"
ANALYSIS_SHA = "7d5e7069fb27456bb5dee5c015fc3d7be6a5471306c540c3d3664e861b01b2d6"
need, sha = engine.need, engine.sha


def build():
    need(all(sha(p) == h for p, h in ((PREP, PREP_SHA), (CONSUME, CONSUME_SHA),
                                      (ANALYSIS, ANALYSIS_SHA))), "feedback sources")
    fixed = json.loads(PREP.read_bytes())
    need(fixed == split.build(), "feedback preparation rebuild")
    observed = json.loads(CONSUME.read_bytes())
    analysis = json.loads(ANALYSIS.read_bytes())
    anchor_path = engine.physical.bundle.PREP
    need(sha(anchor_path) == engine.physical.PREP_SHA, "anchor source")
    anchors = json.loads(anchor_path.read_bytes())["anchors"]
    need(analysis["status"] == "saved_midpoint_verified"
         and len(anchors) == len(fixed["roster"]) == len(observed["rows"]) == len(analysis["rows"]) == 24,
         "fixed roster")
    banks = []
    roles = ("selected", "zero", "greedy")
    for pair in range(8):
        bank = [{"index": 0, "name": "zero_suffix", "raw10_float32_hex": bytes(80).hex()}]
        triplet = anchors[3 * pair:3 * pair + 3]
        need(all(all(a[k] == triplet[0][k] for k in ("root_slot", "condition", "arm", "task_id",
                 "root_state", "goal")) for a in triplet), "paired task")
        for offset, role in enumerate(roles):
            i = 3 * pair + offset
            item, result, anchor, checked = fixed["roster"][i], observed["rows"][i], anchors[i], analysis["rows"][i]
            need(item["index"] == result["index"] == anchor["index"] == checked["index"] == i
                 and item["role"] == result["role"] == anchor["role"] == checked["role"] == role
                 and item["logical_pair"] == anchor["logical_pair"] == checked["logical_pair"] == pair
                 and item["prefix5_float32_hex"] == anchor["prefix5_float32_hex"], "anchor identity")
            raw = bytes.fromhex(item["raw10_float32_hex"])
            need(len(raw) == 80 and hashlib.sha256(raw).hexdigest() == item["raw10_sha256"]
                 and np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
                 and (np.abs(np.frombuffer(raw, dtype="<f4")) <= 1).all(), "feedback action")
            bank.append({"index": offset + 1, "name": role + "_feedback_tail", "raw10_float32_hex": raw.hex()})
        for item in bank:
            item["raw10_sha256"] = hashlib.sha256(bytes.fromhex(item["raw10_float32_hex"])).hexdigest()
        banks.append(bank)
    bindings = dict(fixed["bindings"])
    for path in (PREP, CONSUME, ANALYSIS, anchor_path, PROTOCOL, Path(__file__).resolve(),
                 Path(engine.__file__).resolve(), Path(consumer.__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(p) == h for p, h in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "anchors": anchors, "banks": banks,
            "bindings": bindings, "fixed_branches": 96, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--report-sha")
    parser.add_argument("--directory")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    engine.NAME, engine.FEEDBACK_PREP, engine.build = NAME, PREP, build
    out = Path(args.output)
    if args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_prepare_") and not out.exists(), "output")
        result = {"sha256": engine.physical.design.run._write_new(out, build())}
    elif args.command == "run":
        need(args.authorize_physics, "physical authorization")
        print("Execute all 96 fixed split-budget crossover branches; no neural inference.", flush=True)
        result = engine.execute(args.prepared, args.sha256, args.output)
    else:
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_consume_") and not out.exists(), "output")
        value = consumer.consume(args.prepared, args.sha256, args.directory, args.report_sha)
        value["schema"] = NAME + ".consume"
        result = {"sha256": engine.physical.design.run._write_new(out, value), "rows": len(value["rows"])}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
