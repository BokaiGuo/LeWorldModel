"""Execute all three feedback tails and zero tail on all three fixed prefixes."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import p1_packet_midpoint_feedback_physical_20261031 as feedback

physical = feedback.physical
ROOT = feedback.ROOT
NAME = "p1_packet_feedback_crossover_20261031"
PROTOCOL = ROOT.parent / "docs/D2P_P1_PACKET_FEEDBACK_CROSSOVER_PROTOCOL_20261031.md"
FEEDBACK_PREP = ROOT / "results/p1_packet_midpoint_feedback_physical_20261031_prepare_RUN01.json"
FEEDBACK_PREP_SHA = "b2d5adc0aa545ece6da270174ef76a9b30f8fcb1eccfe0574268b5c52aaee8ab"
FEEDBACK_CONSUME = ROOT / "results/p1_packet_midpoint_feedback_physical_20261031_consume_RUN01.json"
FEEDBACK_CONSUME_SHA = "4cc60710f3d2a2b0f64c92ff5ce00d842df96720957772016c6f2b18a0e6fbe6"
need, sha = physical.need, physical.sha


def build():
    need(sha(FEEDBACK_PREP) == FEEDBACK_PREP_SHA and sha(FEEDBACK_CONSUME) == FEEDBACK_CONSUME_SHA,
         "feedback sources")
    fixed = json.loads(FEEDBACK_PREP.read_bytes())
    need(fixed == feedback.build(), "feedback preparation rebuild")
    observed = json.loads(FEEDBACK_CONSUME.read_bytes())
    anchors_path = physical.bundle.PREP
    need(sha(anchors_path) == physical.PREP_SHA, "anchor source")
    anchors = json.loads(anchors_path.read_bytes())["anchors"]
    need(len(anchors) == len(fixed["roster"]) == len(observed["rows"]) == 24, "fixed roster")
    banks = []
    roles = ("selected", "zero", "greedy")
    for pair in range(8):
        bank = [{"index": 0, "name": "zero_suffix", "raw10_float32_hex": bytes(80).hex()}]
        for offset, role in enumerate(roles):
            item = fixed["roster"][3 * pair + offset]
            result = observed["rows"][3 * pair + offset]
            anchor = anchors[3 * pair + offset]
            need(item["index"] == result["index"] == anchor["index"] == 3 * pair + offset
                 and item["role"] == result["role"] == anchor["role"] == role
                 and item["logical_pair"] == anchor["logical_pair"] == pair
                 and item["prefix5_float32_hex"] == anchor["prefix5_float32_hex"], "anchor identity")
            raw = bytes.fromhex(item["raw10_float32_hex"])
            need(len(raw) == 80 and hashlib.sha256(raw).hexdigest() == item["raw10_sha256"]
                 and np.isfinite(np.frombuffer(raw, dtype="<f4")).all()
                 and (np.abs(np.frombuffer(raw, dtype="<f4")) <= 1).all(), "feedback action")
            bank.append({"index": offset + 1, "name": role + "_feedback_tail",
                         "raw10_float32_hex": raw.hex()})
        for item in bank:
            item["raw10_sha256"] = hashlib.sha256(bytes.fromhex(item["raw10_float32_hex"])).hexdigest()
        triplet = anchors[3 * pair:3 * pair + 3]
        need(all(all(a[k] == triplet[0][k] for k in ("root_slot", "condition", "arm", "task_id",
                 "root_state", "goal")) for a in triplet), "paired task")
        banks.append(bank)
    bindings = dict(fixed["bindings"])
    for path in (FEEDBACK_PREP, FEEDBACK_CONSUME, anchors_path, PROTOCOL, Path(__file__).resolve()):
        bindings[str(path)] = sha(path)
    need(all(sha(p) == h for p, h in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "anchors": anchors, "banks": banks,
            "bindings": bindings, "fixed_branches": 96, "P1_admission": False,
            "independent_confirmation": False, "fairness_proven": False}


def execute(prepared, digest, output):
    need(sha(prepared) == digest and json.loads(Path(prepared).read_bytes()) == build(), "prepare binding")
    fixed = build()
    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_run_") and not output.exists()
         and not output.is_symlink() and shutil.disk_usage(output.parent).free >= 3 * 1024**3,
         "output and disk")
    old_report = physical.design.uphill.runner.native.read(
        physical.design.PHYS / "report.json", cap=64 * 1024**2)
    binding = old_report["execution_binding"]
    _, data = physical.design.uphill.verify(binding["prepared"], binding["prepare_sha256"],
                                            physical.design.JOIN_SHA)
    physical.design.uphill.runner.native.constructor_contract()
    output.mkdir(mode=0o700)

    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

    shards = []
    for anchor, original_item in zip(fixed["anchors"], json.loads(FEEDBACK_PREP.read_bytes())["roster"], strict=True):
        for probe in fixed["banks"][anchor["logical_pair"]]:
            item = copy.deepcopy(original_item)
            item.update(raw10_float32_hex=probe["raw10_float32_hex"], raw10_sha256=probe["raw10_sha256"])
            folder = output / f"anchor_{anchor['index']:02d}_probe_{probe['index']:02d}"
            report = physical.engine.execute(anchor, item, data, folder, factory)
            shards.append({"anchor": anchor["index"], "probe": probe["index"],
                           "status": report["status"], "sha256": sha(folder / "report.json")})
            print(json.dumps(shards[-1]), flush=True)
            if report["status"] != "recorded_not_consumed":
                break
        if shards[-1]["status"] != "recorded_not_consumed":
            break
    need(all(sha(p) == h for p, h in fixed["bindings"].items()), "final source drift")
    status = "recorded_not_consumed" if len(shards) == 96 and all(
        s["status"] == "recorded_not_consumed" for s in shards) else "partial_or_failed"
    value = {"schema": NAME + ".run", "status": status, "shards": shards,
             "prepare_sha256": digest, "fixed_branches": 96, "P1_admission": False}
    return {"status": status, "sha256": physical.design.run._write_new(output / "run_report.json", value)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    if args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_prepare_") and not out.exists(), "output")
        result = {"sha256": physical.design.run._write_new(out, build())}
    else:
        need(args.authorize_physics, "physical authorization")
        print("Execute all 96 fixed same-suffix branches; no neural inference.", flush=True)
        result = execute(args.prepared, args.sha256, args.output)
    print(json.dumps(result), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
