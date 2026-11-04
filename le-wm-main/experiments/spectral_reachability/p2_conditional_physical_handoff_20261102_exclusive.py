"""CPU-only conditional winner handoff; physical dispatch deliberately blocked."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p2_same_bank_host_development_20261102_exclusive as host
from . import p2_common_objective_ledger_20261102_exclusive as ledger
from . import p2_same_bank_winner_ledger_crosscheck_20261102_exclusive as cross
from . import run_pc_p2_conditional_host as model
from . import p2_shared_rgb_physical_bridge_20261102_exclusive as old
from . import p2_b0_isolated_physical_consume_20261102_exclusive as isolation

NAME = "p2_conditional_physical_handoff_20261102_exclusive"
ROOT = host.ROOT
RESULTS = host.RESULTS
MODEL_PLAN = RESULTS / "pc_p2_conditional_host_20261102_PLAN_v1.json"
MODEL_PLAN_SHA = "a0d05017806eb70d2b8514cc5179a3a3026d5eb45cdb829e6ff8783a3a88064f"
MODEL_RUN = RESULTS / "pc_p2_conditional_host_20261102_RUN_v1"
REPORT_SHA = "4f7fe258941ac4157671143dd025db51ffa41a70dc75c8bea7b22efc6b48e02a"
JOURNAL_SHA = "6b0c6cb65f007081d15e039af4805319fdb628e406a8c79f4342c3f7fa2dcdf9"
BINDING_SHA = "d9684d35f81ca90c33745d4a3bd8bf0c5a5c627c3953f7266b13946d04268fa3"
PHYSICAL_PROOF_SHA = "449d1a6edb73917eb95c17693eb9d259f9a578fd4a6cb1dad29227235a7d67ae"
ARMS = ("composition", "no_composition_sra")


def checked(path, expected):
    if host.file_sha(path) != expected:
        raise ValueError("SHA drift: " + str(path))
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("changed during read: " + str(path))
    return data


def winners(report, events):
    result = report["result"]
    rows = result["rows"]
    if [x["candidate_index"] for x in rows] != list(range(16)):
        raise ValueError("candidate order")
    compatible = {"rows": [{**x, "index": x["candidate_index"]} for x in rows]}
    cross.check_result_events(compatible, events)
    out = []
    for arm in ARMS:
        options = []
        for row in rows:
            if [x["arm"] for x in row["arms"]] != list(ARMS):
                raise ValueError("arm order")
            entry = next(x for x in row["arms"] if x["arm"] == arm)
            options.append((entry["model_score"], row["candidate_index"], entry))
        score, index, chosen = min(options, key=lambda x: (x[0], x[1]))
        raw = bytes.fromhex(chosen["raw_float32_hex"])
        host.native.optimizer.raw_actions(np.frombuffer(raw, dtype="<f4").reshape(10, 2))
        out.append({"arm": arm, "candidate_index": index, "iteration": chosen["winner_iteration"],
                    "model_score": score, "suffix_raw_hex": raw.hex(),
                    "suffix_sha256": hashlib.sha256(raw).hexdigest()})
    return out


def build():
    plan = model.verify(MODEL_PLAN, MODEL_PLAN_SHA)
    binding = json.loads(checked(MODEL_RUN / "binding.json", BINDING_SHA))
    report = json.loads(checked(MODEL_RUN / "report.json", REPORT_SHA))
    journal_raw = checked(MODEL_RUN / "journal.jsonl", JOURNAL_SHA)
    audit = ledger.verify_ledger(MODEL_RUN / "journal.jsonl")
    if (binding != {"plan": str(MODEL_PLAN), "sha256": MODEL_PLAN_SHA}
            or report["status"] != "completed_development" or report["failure"] is not None
            or not report["frozen_parameters_unchanged"] or audit["outstanding"]
            or report["journal_sha256"] != JOURNAL_SHA or report["plan_sha256"] != MODEL_PLAN_SHA):
        raise ValueError("conditional model report/binding/ledger")
    events = [json.loads(line) for line in journal_raw.splitlines()]
    result = report["result"]
    if any(result[key] != plan["input"][key]
           for key in ("context_sha256", "goal_sha256", "handoff_sha256", "pair_sha256")):
        raise ValueError("model input binding")
    selected = winners(report, events)
    proof = json.loads(checked(isolation.OUTPUT, PHYSICAL_PROOF_SHA))
    if (proof["status"] != "consumer_returned_success" or
            proof["consumer_result"]["status"] != "verified_known_root_physical_not_admitted" or
            proof["contract_sha256"] != isolation.CONTRACT_SHA or
            proof["physical_report_sha256"] != isolation.REPORT_SHA):
        raise ValueError("historical isolated physical proof")
    p = json.loads(checked(old.PREP, old.PREP_SHA))
    fixed = host.native.verified_input()
    context = json.loads(fixed["context_json"])
    observed = p["observed_context"]
    if any(context[k] != observed[k] for k in ("layout", "latent_history", "action_history")):
        raise ValueError("physical observed context differs")
    bundle, _, _, actions = old.launch.wire.load_capture(old.launch.CAPTURE / "capture.json",
                                                         old.launch.CAPTURE_SHA)
    prefix = np.asarray(actions, dtype="<f4").tobytes()
    if (bundle["base"] != 0 or bundle["source"]["fresh_root"] is not False or
            p["job_index"] != 0 or len(prefix) != 120 or len(p["prefix_reference"]) != 106):
        raise ValueError("B0 prefix contract")
    physical_events = [old.physical.native.decode(line) for line in
                       (old.capture.RUN / "journal.jsonl").read_bytes().splitlines()]
    old_report = json.loads(checked(old.capture.RUN / "report.json", old.capture.REPORT_SHA))
    if host.file_sha(old.capture.RUN / "journal.jsonl") != old_report["journal_sha256"]:
        raise ValueError("historical physical journal SHA")
    refs = [old.physical.normalized_prefix_event(e) for e in physical_events
            if e.get("branch") == 0 and old.physical.is_prefix_event(e)]
    if refs != p["prefix_reference"] or len(refs) != 106:
        raise ValueError("historical full prefix reference")
    for row in p["plans"]:
        if (np.asarray(row["commands"], dtype="<f4").tobytes()[:120] != prefix or
                row["native_root_sha256"] != bundle["source"]["root_sha256"] or
                row["task_id"] != bundle["source"]["task_id"]):
            raise ValueError("physical root/task/prefix")
    for item in selected:
        full = prefix + bytes.fromhex(item["suffix_raw_hex"])
        item["full25_raw_hex"] = full.hex()
        item["full25_sha256"] = hashlib.sha256(full).hexdigest()
    sources = [Path(__file__), Path(cross.__file__), Path(old.physical.__file__),
               Path(model.__file__), Path(isolation.__file__)]
    return {"schema": NAME + ".plan_v1", "status": "prepared_execution_blocked",
            "source_sha256": {str(f): host.file_sha(f) for f in sources},
            "model_plan_sha256": MODEL_PLAN_SHA, "model_report_sha256": REPORT_SHA,
            "model_journal_sha256": JOURNAL_SHA, "physical_prepare_sha256": old.PREP_SHA,
            "isolated_prior_physical_proof_sha256": PHYSICAL_PROOF_SHA,
            "capture_sha256": old.launch.CAPTURE_SHA, "root_sha256": bundle["source"]["root_sha256"],
            "task_id": bundle["source"]["task_id"],
            "context_sha256": result["context_sha256"], "goal_sha256": result["goal_sha256"],
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
            "prefix_reference_sha256": old.physical.digest(refs), "arms": selected,
            "selection": "minimum saved model score of 16 per arm; earliest candidate on tie; no physical outcomes",
            "physical_protocol": "fresh native env per arm, warm10 + exact guarded prefix15 + suffix10, no padding",
            "physical_dispatch_blockers": ["historical source SHA drift in active worktree",
                "new winner-aware execution and saved-output consumer not yet implemented"],
            "independent_confirmation": False, "equal_total_compute": False,
            "P1_admission": False, "P2_admission": False}


def output(path):
    path = Path(path)
    if (not path.is_absolute() or path.parent != RESULTS or path.parent.resolve() != RESULTS
            or not path.name.startswith(NAME + "_PLAN_")):
        raise ValueError("absolute exclusive plan path required")
    return path


def verify(path, digest):
    path = output(path)
    saved = json.loads(checked(path, digest))
    if saved != build() or host.file_sha(path) != digest:
        raise ValueError("plan or source drift")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    for name in ("check", "run"):
        command = sub.add_parser(name)
        command.add_argument("--plan", required=True)
        command.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        path = output(args.output)
        print(json.dumps({"plan": str(path), "sha256": host.write_new(path, build())}))
    else:
        verify(args.plan, args.plan_sha256)
        if args.command == "run":
            raise RuntimeError("physical execution blocked: winner-aware native runner and consumer unavailable")
        print(json.dumps({"status": "handoff_verified_physical_execution_blocked"}))


if __name__ == "__main__":
    main()
