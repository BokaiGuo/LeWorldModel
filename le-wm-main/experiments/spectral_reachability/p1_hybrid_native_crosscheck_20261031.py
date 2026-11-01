"""Cross-check sealed hybrid physical transitions against native task metrics."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_hybrid_h10_saved_20261031 as h10_saved
from . import p1_hybrid_cp5_physical_saved_20261031 as cp5_saved
from .codex_p1_h10_paired_table_20261030_main import physical_metrics

ROOT = h10_saved.hybrid.ROOT
sha, need = h10_saved.hybrid.sha, h10_saved.require
H10_DIR = ROOT / "results/p1_hybrid_h10_20261031_RUN01"
H10_SHA = "fbc3c108392abc3943fda110c1cebdb92a947b11ee633190369b7ed33e6f06a1"
H10_PREPARE = ROOT / "results/p1_hybrid_h10_20261031_prepare_RUN01.json"
H10_PREPARE_SHA = "724a6ec10566b31e50f2287678ac926986a4bab4b1951a13aee694c5735f7433"
CP5_DIR = ROOT / "results/p1_hybrid_cp5_physical_20261031_RUN01"
CP5_SHA = "16e520bdbb664fe80a91cae030c414c73d84b86ad2b0698e20b3840d0376a67d"
CP5_PREPARE = ROOT / "results/p1_hybrid_cp5_physical_20261031_prepare_RUN01.json"
CP5_PREPARE_SHA = "b91993a25e7c504d16015f0509685ef76566e9655d451a92918c10f06d6b243a"
CP5_RECEIPT = ROOT / "results/p1_hybrid_cp5_physical_saved_20261031_RUN01.json"
CP5_RECEIPT_SHA = "70068a853c359b37d74dcac17f0121e9978b7227e4c85373582cb74a3b9764ff"
PAIR = ROOT / "results/p1_hybrid_h10_paired_20261031_RUN01.json"
PAIR_SHA = "bc7d5cad0ca0f8f63b4194aaba9e6473ad0caeff2ded7220c1bfe7949614168a"
metric = h10_saved.run.base.previous.metric


def verify_transitions(events, anchor, rows):
    transitions = [e for e in events if e["kind"] == "returned_transition"]
    ops = {}
    for event in events:
        if event["kind"] == "operation_returned":
            receipt = event["receipt"]
            if receipt.get("resource") == "public_step":
                key = (receipt.get("branch"), receipt.get("phase"), receipt.get("step"))
                need(key not in ops, "duplicate public step")
                ops[key] = event
    checked = {"plan": 0, "probe": 0}
    branches = set()
    for event in transitions:
        phase = event["phase"]
        key = (event["branch"], phase, event["step"])
        need(key in ops and ops[key]["index"] < event["index"]
             and ops[key]["receipt"]["raw_bits"] == event["actual_bits"] == event["requested_bits"],
             "public action binding")
        if phase == "warmup":
            continue
        need(phase in checked and event["goal_state"] == anchor["goal"], "phase and goal")
        native = metric(anchor["goal"], event["state"])
        recomputed = physical_metrics(event["state"], anchor["goal"])
        need(event["reward"] == -native["native_distance"]
             and event["terminated"] == native["native_success"]
             and native["native_success"] == recomputed["native_success"], "native reward/termination")
        checked[phase] += 1
        branches.add(event["branch"])
    need(branches == set(range(len(rows))), "branch coverage")
    for index, row in enumerate(rows):
        probe = [event for event in transitions if event["branch"] == index and event["phase"] == "probe"]
        expected = len(row["responses"]) + int(row.get("status") == "censored_out_of_domain")
        need(len(probe) == expected, "probe response denominator")
        for response, event in zip(row["responses"], probe, strict=False):
            delta = np.asarray(event["state"]) - np.asarray(row["start_state"])
            need(response["delta_native7"] == delta.tolist()
                 and response["delta_angle_wrapped"] == float(np.arctan2(np.sin(delta[4]), np.cos(delta[4])))
                 and response["terminated"] == event["terminated"], "response arithmetic")
    return checked


def run():
    for path, digest in ((H10_DIR / "run_report.json", H10_SHA), (H10_PREPARE, H10_PREPARE_SHA),
                         (CP5_DIR / "run_report.json", CP5_SHA), (CP5_PREPARE, CP5_PREPARE_SHA),
                         (CP5_RECEIPT, CP5_RECEIPT_SHA), (PAIR, PAIR_SHA)):
        need(sha(path) == digest, "pinned source")
    h10 = h10_saved.hybrid.read(H10_DIR / "run_report.json")
    cp5 = h10_saved.hybrid.read(CP5_DIR / "run_report.json")
    hp, pp = h10_saved.hybrid.read(H10_PREPARE), h10_saved.hybrid.read(CP5_PREPARE)
    need(h10["status"] == cp5["status"] == "recorded_not_consumed"
         and len(h10["shards"]) == len(cp5["shards"]) == 2, "complete runs")
    result = []
    for label, directory, report, prepared in (
            ("common_h10", H10_DIR, h10, hp), ("own_cp5", CP5_DIR, cp5, pp)):
        for anchor, shard in zip(prepared["anchors"], report["shards"], strict=True):
            name = ("anchor_" if label == "common_h10" else "row_") + f"{anchor['index']:02d}"
            folder = directory / name
            need(shard["index"] == anchor["index"] and sha(folder / "report.json") == shard["sha256"],
                 "shard binding")
            document = h10_saved.hybrid.read(folder / "report.json")
            journal = folder / "journal.jsonl"
            need(sha(journal) == document["journal_sha256"], "journal SHA")
            events = h10_saved.saved.events(journal, document["journal_chain_tail"])
            rows = document["rows"] if label == "common_h10" else [document["row"]]
            checks = verify_transitions(events, anchor, rows)
            result.append({"kind": label, "root_slot": anchor["root_slot"], "branches": len(rows),
                           "transition_checks": checks})
    return {"schema": "p1_hybrid_native_crosscheck_20261031", "status": "checked",
            "rows": result, "P1_admission": False, "independent_confirmation": False,
            "limitations": "Saved data on two known roots; no new physical or model execution."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    need(output.is_absolute() and output.parent == ROOT / "results" and
         output.name.startswith("p1_hybrid_native_crosscheck_20261031_") and not output.exists(),
         "exclusive output")
    result = run()
    print(json.dumps({"sha256": h10_saved.run._write_new(output, result), "rows": result["rows"]}))


if __name__ == "__main__":
    main()
