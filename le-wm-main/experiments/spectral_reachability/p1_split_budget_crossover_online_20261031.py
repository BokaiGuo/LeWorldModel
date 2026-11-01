"""Prepare and execute the fixed-tail crossover for the repaired online prefix run."""
import argparse
import json
from pathlib import Path

from . import p1_packet_feedback_crossover_20261031 as engine
from . import p1_packet_feedback_crossover_consume_20261031 as consumer
from . import p1_split_budget_online_20261031 as online

ROOT = online.ROOT
NAME = "p1_split_budget_crossover_online_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_SPLIT_BUDGET_CROSSOVER_PROTOCOL_20261031.md"
FEEDBACK_PREP = ROOT / "results/p1_split_budget_online_20261031_prepare_RUN05.json"
FEEDBACK_CONSUME = ROOT / "results/p1_split_budget_online_20261031_consume_RUN05.json"
FEEDBACK_PREP_SHA = "0da343ecdbbebff7e419fff946c06e6ebdc3c2a3a37607f1fc450d89cb2c96e0"
FEEDBACK_CONSUME_SHA = "47878f5efc9514cdd97a016a1663dae6201f92e0b1aa4dc0fdcc7c9b4af77eef"
EXISTING_PREP = ROOT / "results/p1_split_budget_crossover_20261031_prepare_RUN01.json"
EXISTING_PREP_SHA = "5245676e03c61592f3d4ae66d2559afab009ebf9b3f5b2349d9e07651a712243"
need, sha = engine.need, engine.sha


def reject_duplicate_panel(proposed, existing):
    """Refuse the known duplicate before launching another physical panel."""
    need(isinstance(existing.get("anchors"), list) and len(existing["anchors"]) == 24
         and isinstance(existing.get("banks"), list) and len(existing["banks"]) == 8,
         "existing panel roster")
    same_anchors = proposed["anchors"] == existing["anchors"]
    same_banks = proposed["banks"] == existing["banks"]
    need(not (same_anchors and same_banks), "duplicate completed crossover panel")


def build():
    need(sha(FEEDBACK_PREP) == FEEDBACK_PREP_SHA
         and sha(FEEDBACK_CONSUME) == FEEDBACK_CONSUME_SHA, "feedback sources")
    fixed = json.loads(FEEDBACK_PREP.read_bytes())
    observed = json.loads(FEEDBACK_CONSUME.read_bytes())
    anchors_path = engine.physical.bundle.PREP
    need(len(fixed["roster"]) == len(observed["rows"]) == 24
         and sha(anchors_path) == engine.physical.PREP_SHA, "fixed roster")
    anchors = json.loads(anchors_path.read_bytes())["anchors"]
    need(len(anchors) == 24, "anchor count")
    banks = []
    for pair in range(8):
        bank = [{"index": 0, "name": "zero_suffix", "raw10_float32_hex": bytes(80).hex()}]
        triplet = anchors[3 * pair:3 * pair + 3]
        need(all(all(a[k] == triplet[0][k] for k in
                     ("root_slot", "condition", "arm", "task_id", "root_state", "goal"))
                  for a in triplet), "paired task")
        for offset, role in enumerate(("selected", "zero", "greedy")):
            i = 3 * pair + offset
            item, result, anchor = fixed["roster"][i], observed["rows"][i], anchors[i]
            need(item["index"] == result["index"] == anchor["index"] == i
                 and item["role"] == result["role"] == anchor["role"] == role
                 and item["logical_pair"] == anchor["logical_pair"] == pair
                 and item["prefix5_float32_hex"] == anchor["prefix5_float32_hex"],
                 "anchor identity")
            bank.append({"index": offset + 1, "name": role + "_feedback_tail",
                         "raw10_float32_hex": item["raw10_float32_hex"]})
        for item in bank:
            import hashlib
            item["raw10_sha256"] = hashlib.sha256(
                bytes.fromhex(item["raw10_float32_hex"])).hexdigest()
        banks.append(bank)
    bindings = dict(fixed["bindings"])
    for path in (FEEDBACK_PREP, FEEDBACK_CONSUME, anchors_path, PROTOCOL,
                 Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    proposed = {"schema": NAME + ".prepare", "anchors": anchors, "banks": banks,
                "bindings": bindings, "fixed_branches": 96, "P1_admission": False,
                "independent_confirmation": False, "fairness_proven": False}
    need(sha(EXISTING_PREP) == EXISTING_PREP_SHA, "existing panel source")
    reject_duplicate_panel(proposed, json.loads(EXISTING_PREP.read_bytes()))
    return proposed


def configure():
    engine.NAME = NAME
    engine.FEEDBACK_PREP = FEEDBACK_PREP
    engine.build = build
    consumer.source.FEEDBACK_PREP = FEEDBACK_PREP
    consumer.source.build = build


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
    configure()
    out = Path(args.output)
    if args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_prepare_") and not out.exists(), "output")
        result = {"sha256": engine.physical.design.run._write_new(out, build())}
    elif args.command == "run":
        need(args.authorize_physics, "physical authorization")
        result = engine.execute(args.prepared, args.sha256, args.output)
    else:
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_consume_") and not out.exists(), "output")
        value = consumer.consume(args.prepared, args.sha256, args.directory, args.report_sha)
        result = {"sha256": engine.physical.design.run._write_new(out, value),
                  "rows": len(value["rows"])}
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
