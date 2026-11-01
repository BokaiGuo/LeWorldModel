"""Bounded saved-reset collision receipt; no environment, model, or freshness certification."""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import codex_p1_original_reset_inventory_20261101 as inventory
from experiments.spectral_reachability import p1_conditional_procedural_source_20261027_v1 as runner
from experiments.spectral_reachability import p1_explicit_collision_check_20261101_hypatia as prior
from experiments.spectral_reachability import p1_journal_collision_supplement_20261101_hypatia as journal

ROOT = prior.base.ROOT
NAME = "p1_exploration_run01_receipt_20261101_hypatia"
REQUEST = ROOT / "results/p1_exploration_seed_preseal_20261101_hypatia_request.json"
DIRECTORY = ROOT / "results/p1_exploration_seed_preseal_20261101_hypatia_run01"
INPUTS = {
    REQUEST: "bcc5d39dcb85b0f7fba1c66f718ec00acf1909871568023ee809338a10814b53",
    DIRECTORY / "sourcepack.json": "5312e3eec05c91698d3689ce4dcbe08732ca3c76800e7cbb5fe7d5016c1465ab",
    DIRECTORY / "journal.jsonl": "eadbb66295647e3ff97c17428f0f521ce27082d2aeed9f6d65c7af8a39955800",
    DIRECTORY / "source_0.npz": "1163eb2b5efa1897c5ed106351e5f8e65ada27041a3eacbb3c9d5f1232d92b19",
    DIRECTORY / "source_1.npz": "356fe60e6499ee5a1957a3c01189826607ea5cf72afd630186ca71569cb7c0f1",
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def check_binding(request, report, events):
    p, e = request["plan"], request["exclusions"]
    need(runner.digest(p) == request["plan_sha256"] and runner.digest(e) == request["exclusion_sha256"]
         and report["plan"] == p and report["exclusion_id"] == request["exclusion_sha256"],
         "request binding")
    need(report["status"] == "completed_calibration_source_only" and report["durable_journal_complete"] is True
         and report["first_error"] is None and report["secondary_errors"] == []
         and report["P1_admission"] is False and report["new_public_steps"] == 0, "run status")
    need(events[0]["kind"] == "run_started" and events[0]["plan"] == p and events[0]["exclusions"] == e
         and events[-1]["kind"] == "run_finished" and events[-1]["status"] == report["status"]
         and events[-1]["resources"] == report["resources"], "journal boundary")
    need([x["receipt"] for x in events if x["kind"] == "slot_finished"] == report["slots"],
         "slot receipt mismatch")
    attempts = [a for slot in report["slots"] for proposal in slot["proposals"] for a in proposal["attempts"]]
    need([x["receipt"] for x in events if x["kind"] == "attempt_finished"] == attempts
         and len(attempts) == 8, "attempt receipt mismatch")
    pending = None
    counts = {"reset": {"attempted": 0, "completed": 0}, "render": {"attempted": 0, "completed": 0}}
    keys = ("operation", "candidate_id", "proposal_index", "attempt_index", "random_reset_seed", "placement_reset_seed")
    for event in events:
        if event["kind"] == "operation_started":
            need(pending is None and event["operation"] in ("random_reset", "placement_reset", "render"),
                 "unexpected or overlapping operation")
            pending = event
            counts["render" if event["operation"] == "render" else "reset"]["attempted"] += 1
        elif event["kind"] == "operation_completed":
            need(pending is not None and all(pending[k] == event[k] for k in keys), "operation pairing")
            counts["render" if event["operation"] == "render" else "reset"]["completed"] += 1
            pending = None
    need(pending is None and counts == report["resources"]
         and counts == {"reset": {"attempted": 16, "completed": 16},
                        "render": {"attempted": 8, "completed": 8}}, "operation counts")
    return attempts


def build():
    raw = {p: prior.base.read_bound(p, digest) for p, digest in INPUTS.items()}
    request = prior.base.decode(raw[REQUEST])
    source = prior.base.decode(raw[DIRECTORY / "sourcepack.json"])
    events = runner.read_journal(DIRECTORY / "journal.jsonl")
    attempts = check_binding(request, source, events)
    extracted = inventory.extract(raw[DIRECTORY / "sourcepack.json"], raw[DIRECTORY / "journal.jsonl"],
                                  str(DIRECTORY / "sourcepack.json"), str(DIRECTORY / "journal.jsonl"))
    need(len(extracted["events"]) == 16 and all(r["state"] is not None for r in extracted["events"]),
         "missing reset state")
    selected = []
    for slot in source["slots"]:
        slot_id = slot["slot"]["candidate_id"]
        chosen = [a for p in slot["proposals"] for a in p["attempts"] if a["selected"]]
        need(slot["status"] == "completed" and len(chosen) == 1, "selected source missing")
        path = DIRECTORY / f"source_{slot_id}.npz"
        need(slot["raw_source_file"] == path.name and slot["raw_source_sha256"] == INPUTS[path], "NPZ binding")
        packet = runner.load_raw_source(path, request["plan"])
        need(packet["metadata"] == slot["metadata"] and packet["state"].tolist() == chosen[0]["state"]
             and packet["goal"].tolist() == chosen[0]["goal"]
             and packet["metadata"]["source"]["proposal_index"] == chosen[0]["proposal_index"]
             and packet["metadata"]["selected_attempt"]["attempt_index"] == chosen[0]["attempt_index"]
             and packet["metadata"]["selected_attempt"]["random_reset_seed"] == chosen[0]["random_reset_seed"],
             "NPZ selected state binding")
        selected.append({"candidate_id": slot_id, "proposal_index": chosen[0]["proposal_index"],
                         "attempt_index": chosen[0]["attempt_index"],
                         "state_identity": prior.base.identity(chosen[0]["state"])})
    rows = []
    for i, event in enumerate(extracted["events"]):
        rows.append({"candidate_id": f"event_{i}", "root_seed": None, "reset_seed": None,
                     "state7": event["state"]})
    candidates = {"schema": prior.NAME + ".candidates.v1", "candidates": rows}
    history = prior.base.decode(prior.base.read_bound(prior.HISTORY_PATH, prior.HISTORY_SHA))
    coherent = prior.base.decode(prior.base.read_bound(prior.COHERENT_PATH, prior.COHERENT_SHA))
    historical_journal = prior.base.decode(prior.base.read_bound(journal.INVENTORY, journal.INVENTORY_SHA))
    comparisons = prior.compare(candidates, history, coherent)
    supplemental = journal.supplement(candidates, historical_journal)
    for comparison, extra, event in zip(comparisons, supplemental, extracted["events"]):
        comparison["historical_journal_hits"] = extra["journal_hits"]
        comparison["event_reference"] = {
            "operation": event["operation"], "coordinates": event["coordinates"],
            "start": event["start"], "completion": event["completion"],
            "source_attempt": event["source_attempt"],
        }
    for path, expected in INPUTS.items():
        prior.base.read_bound(path, expected)
    modules = [__file__, prior.__file__, prior.base.__file__, prior.coherent.__file__,
               journal.__file__, runner.__file__, inventory.__file__]
    return {
        "schema": NAME + ".v1", "inputs": {str(p): h for p, h in INPUTS.items()},
        "implementation_sha256": {str(Path(p).resolve()): runner.file_sha(Path(p)) for p in modules},
        "history_sha256": prior.HISTORY_SHA, "coherent_sha256": prior.COHERENT_SHA,
        "historical_journal_sha256": journal.INVENTORY_SHA,
        "run_status": source["status"], "resources": source["resources"],
        "selected_sources": selected, "reset_comparisons": comparisons,
        "limitations": ["recorded reset states are not a complete historical ledger",
                        "selected state and pose comparison does not prove source or training independence",
                        "RGB history is one frame repeated with zero action padding",
                        "native runtime closure and unrecorded restarts remain unknown"],
        "history_complete": False, "development_disjoint": "unknown", "training_disjoint": "unknown",
        "freshness": "unknown", "freshness_certified": False, "P1_admission": False,
        "new_environment_or_gpu_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    need(out.is_absolute() and out.parent.resolve() == out.parent and out.name.startswith(NAME + "_")
         and out.suffix == ".json", "output namespace")
    data = prior.base.encode(build())
    with out.open("xb") as stream:
        need(stream.write(data) == len(data), "short write")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(out), "sha256": prior.base.sha(data),
                      "freshness_certified": False, "P1_admission": False}))


if __name__ == "__main__":
    main()
