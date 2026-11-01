"""Recompute saved known-root common-budget pair; no neural or physics calls."""
from collections import Counter
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .codex_p1_two_root_saved_audit_20261101_exclusive import chain, digest, filehash, native, require
from . import codex_p1_variable_cem_root0_audit_20261101_exclusive as cem_audit
from . import codex_p1_variable_cem_host_v2_20261101_exclusive as anchor

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
REPORTS = {
    "structured": ("codex_p1_common_structured_cp10_host_20261101_exclusive_RUN_CODEX01",
                   "6c49744a051ab327ca91c22f0aa5c19261eacff51c472fdf0b1cd4915d80f4cd"),
    "cem": ("p1_common_budget_cem_host_20261101_leibniz_root0_RUN_LEIBNIZ01",
            "28d1df63a0aa883256229d1a7fb5a9d2b78e96086f79b29ef9205703dee8a6c2"),
}
BOUNDARIES = [0, 5, 10, 15]
AXES = ("action_encode_calls", "predict_calls", "predict_samples", "replay_calls",
        "selection_evaluations", "backward_calls", "vjp_rows", "jacobian_calls", "svd_calls")


def load():
    reports, prepares, sources = {}, {}, {}
    for arm, (name, expected) in REPORTS.items():
        path = RESULTS / name / "report.json"
        require(filehash(path) == expected, f"{arm} report hash")
        report = json.loads(path.read_bytes())
        prep_path = Path(report["prepared"])
        require(prep_path.parent == RESULTS and filehash(prep_path) == report["prepare_sha256"],
                f"{arm} prepare hash")
        prepared = json.loads(prep_path.read_bytes())
        require(prepared["protocol"] == report["protocol"], f"{arm} protocol")
        for name, sha in prepared["sources"].items():
            require(name not in sources or sources[name] == sha, "source hash conflict")
            sources[name] = sha
        reports[arm], prepares[arm] = report, prepared
    for name, sha in sources.items():
        require(filehash(name) == sha, f"source drift: {name}")
    return reports, prepares, len(sources)


def verify_pair(reports, prepares):
    s, c = reports["structured"], reports["cem"]
    binding = s["cp10_goal_binding"]
    require(binding == c["cp10_goal_binding"] == prepares["structured"]["cp10_identity"]
            == prepares["cem"]["cem_source_identity"]["rebound_goal_binding"], "root/goal binding")
    require(s["model_before"] == s["model_after"] == c["model_before"] == c["model_after"], "model binding")
    require(s["protocol"]["common_node_caps"] == c["protocol"]["common_node_caps"], "common caps")
    require(s["protocol"]["boundaries"] == c["protocol"]["boundaries"] == BOUNDARIES
            and s["protocol"]["packet_steps"] == c["protocol"]["packet_steps"] == 5
            and s["protocol"]["terminal"] == c["protocol"]["terminal"] == 20, "feedback schedule")
    st, ct = prepares["structured"]["source"]["tasks"][0], prepares["cem"]["source"]["tasks"][0]
    require(all(st[k] == ct[k] for k in ("action_mean", "action_std", "projection")), "normalizer/projection")
    require(digest(st["goal"]) == binding["goal_latent_sha256"]
            and st["binding"]["root_sha256"] == binding["root_sha256"], "structured task target")
    return binding


def ledger_check(node, caps, binding, model, folder):
    receipt, account = node["receipt"], node["receipt"]["account"]
    ledger = Path(receipt["ledger"]["journal"])
    require(ledger.is_relative_to(folder) and filehash(ledger) == receipt["ledger"]["sha256"]
            == account["journal_sha256"], "ledger file hash")
    events = chain(ledger, "ledger")
    header, terminal = events[0]["payload"], events[-1]
    contract = header["contract"]
    require(header["contract_sha256"] == digest(contract) == account["contract_sha256"]
            and contract["levels"][0]["caps"] == caps, "ledger contract/caps")
    identity = contract["identity"]
    require(identity["root"] == binding["root_sha256"] and identity["goal"] == binding["goal_latent_sha256"]
            and identity["model"] == model, "ledger identity")
    inner = events[1:-1]
    require(len(inner) % 2 == 0 and [x["kind"] for x in inner] == ["intent", "return"] * (len(inner)//2),
            "ledger completed sequence")
    intents = inner[::2]
    total = Counter()
    for event in intents:
        total.update(event["payload"]["reservation"])
    charge = dict(total)
    require(charge == account["charged_upper_bounds"] == terminal["payload"]["charged_upper_bounds"]
            and terminal["kind"] == "terminal" and terminal["payload"]["status"] == "closed_unverified"
            and len(intents) == account["attempted_events"] == account["completed_events"]
            and not account["actual_usage_verified"] and not account["coverage_verified"]
            and all(charge[k] <= caps[k] for k in caps), "ledger settlement")
    return {k: charge[k] for k in AXES}


def structured_selection(node, previous, rgb_sha=None):
    receipt = node["receipt"]
    path = Path(receipt["trace"])
    require(filehash(path) == receipt["trace_sha256"], "structured trace hash")
    events = chain(path, "ledger")
    require(events[-1]["kind"] == "terminal" and events[-1]["payload"]["status"] == "complete_node",
            "structured trace terminal")
    selected = [x["payload"] for x in events if x["kind"] == "selected_full_plan"]
    require(len(selected) == 1, "structured selected plan")
    selected = selected[0]
    plan = np.asarray(selected["raw"], dtype=np.float32)
    require(plan.ndim == 3 and plan.shape[1:] == (5, 2) and np.isfinite(plan).all()
            and (np.abs(plan) <= 1).all(), "plan shape")
    sha = hashlib.sha256(plan.tobytes()).hexdigest()
    require(sha == selected["plan_sha256"] == receipt["selected_full_plan_sha256"]
            and plan.tolist() == receipt["selected_full_plan"]
            and plan[0].tolist() == receipt["commands"], "structured plan binding")
    final = [x["payload"] for x in events if x["kind"] == "final_selection"]
    require(len(final) == 1, "structured original selection")
    if node["boundary"]:
        feedback = [x["payload"] for x in events if x["kind"] == "first_feedback_selection"]
        require(len(feedback) == 1 and previous is not None, "carry selection")
        feedback = feedback[0]
        bound = feedback["feedback"]
        require(bound["previous_plan_sha256"] == hashlib.sha256(previous.tobytes()).hexdigest()
                and bound["issued_packet_sha256"] == hashlib.sha256(previous[0].tobytes()).hexdigest()
                and bound["rgb_sha256"] == rgb_sha and bound["boundary"] == node["boundary"],
                "actual carry/RGB provenance")
        scores = feedback["scores"]
        require(all(math.isfinite(x) for x in scores)
                and feedback["selected_source"] in ("new", "carry")
                and receipt["selected_index"] == min(range(len(scores)), key=lambda i: (scores[i], i)),
                "feedback ranking")
        if feedback["selected_source"] == "carry":
            require(np.array_equal(plan, previous[1:]), "carry plan")
        else:
            optimized = next(x["payload"]["raw"] for x in events if x["kind"] == "optimized_candidates")
            require(np.array_equal(plan, np.asarray(optimized, np.float32)[receipt["selected_index"], :len(plan)]),
                    "fresh plan")
    else:
        require(final[0]["index"] == receipt["selected_index"]
                == min(range(len(final[0]["scores"])), key=lambda i: (final[0]["scores"][i], i)), "initial ranking")
        optimized = next(x["payload"]["raw"] for x in events if x["kind"] == "optimized_candidates")
        require(np.array_equal(plan, np.asarray(optimized, np.float32)[receipt["selected_index"]]),
                "initial selected plan")
    return plan


def physical_check(report, arm, folder, selections):
    path = folder / "journal.jsonl"
    events = chain(path, "physical")
    require(filehash(path) == report["journal_sha256"] and events[-1]["event_sha256"]
            == report["journal_chain_tail"], "physical journal")
    row = report["rows"][0]
    obs_list = [x["receipt"] for x in events if x["kind"] == "fresh_observation" and x["receipt"]["phase"] == "plan"]
    observations = {x["step"]: x for x in obs_list}
    n = row["control_returned"]
    require(len(obs_list) == n + 1 and set(observations) == set(range(n+1)), "observation roster")
    dispatch = [x for x in events if x["kind"] == "authorized_dispatch"]
    returns = [x for x in events if x["kind"] == "control_return"]
    require(len(dispatch) == len(returns) == len(row["executed_raw"]) == n, "step roster")
    goal, metrics = observations[0]["goal"], []
    for step in range(n+1):
        obs = observations[step]
        image_path = Path(obs["path"])
        require(image_path.is_relative_to(folder) and filehash(image_path) == obs["file_sha256"]
                and obs["state"] == obs["snapshot"]["state"] and obs["goal"] == goal
                and obs["goal_rgb_sha256"] == report["cp10_goal_binding"]["goal_pixel_sha256"], "observation binding")
        with Image.open(image_path) as image:
            rgb = np.asarray(image)
        require(rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
                and hashlib.sha256(rgb.tobytes()).hexdigest() == obs["pixel_sha256"], "RGB pixel binding")
        metric = native(obs["state"], goal)
        metrics.append(metric)
        if step:
            d, r = dispatch[step-1], returns[step-1]
            boundary = (step-1)//5*5
            require(d["step"] == r["step"] == step and d["index"] < obs["observed_event_index"] < r["index"]
                    and d["command"] == r["command"] == row["executed_raw"][step-1]
                    == selections[boundary][(step-1)%5], "actual action/order")
            require(math.isclose(r["reward"], -metric[2], rel_tol=1e-12, abs_tol=1e-10)
                    and r["terminated"] == metric[3] and not r["truncated"], "physical return")
    require(not any(x[3] for x in metrics[:-1]) and metrics[-1][3] and row["native_success"]
            and row["terminated"] and not row["truncated"] and row["final_state"] == observations[n]["state"],
            "first native success")
    return {"steps": n, "first_native_success": n, "final_xy": metrics[-1][0],
            "final_angle": metrics[-1][1], "initial_state": observations[0]["state"],
            "initial_pixel_sha256": observations[0]["pixel_sha256"], "goal_state": goal,
            "physical_events": len(events), "rgb_encodes": len(row["encodings"])}


def audit_arm(arm, report, binding):
    folder = RESULTS / REPORTS[arm][0]
    require(report["status"] == "recorded_known_root_not_admitted" and not report["errors"]
            and not report["secondary_errors"] and not report["pending"] and not report["P1_admission"],
            "report status")
    row = report["rows"][0]
    nodes = report["nodes"][row["episode_id"]]
    require([x["boundary"] for x in nodes] == BOUNDARIES, "node roster")
    caps = report["protocol"]["common_node_caps"]
    charges, selections, previous = [], {}, None
    physical_events = chain(folder / "journal.jsonl", "physical")
    observations = {e["receipt"]["step"]: e["receipt"] for e in physical_events
                    if e["kind"] == "fresh_observation" and e["receipt"]["phase"] == "plan"}
    if arm == "cem":
        anchor.verify_anchor(folder / "full_cem_events.jsonl", report["cem_journal"])
        events = [x["payload"]["event"] for x in chain(folder / "full_cem_events.jsonl", "cem")]
    for node in nodes:
        if node["status"] == "not_started":
            continue
        require(node["status"] == "sealed", "unsealed node")
        b = node["boundary"]
        charges.append({"boundary": b, "charge": ledger_check(
            node, caps, binding, report["model_before"]["original_state_dict"], folder)})
        receipt = node["receipt"]
        if arm == "structured":
            previous = structured_selection(node, previous, observations[b]["pixel_sha256"])
            selections[b] = receipt["commands"]
        else:
            cem_audit.verify_cem_node(events, node, b)
            selections[b] = receipt["commands"]
        seals = [e for e in physical_events
                 if e["kind"] == "boundary_sealed" and e["boundary"] == b]
        seal_value = receipt if arm == "structured" else receipt["commands"]
        require(len(seals) == 1 and seals[0]["plan_sha256"] == digest(seal_value), "packet seal")
    require(list(selections) == list(range(0, row["control_returned"], 5)), "actual node chronology")
    output = physical_check(report, arm, folder, selections)
    output.update(node_charges=charges, unstarted_boundaries=[x["boundary"] for x in nodes if x["status"] == "not_started"])
    output["totals"] = {k: sum(x["charge"][k] for x in charges) for k in AXES}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="exclusive audit JSON in results")
    args = parser.parse_args()
    reports, prepares, sources = load()
    binding = verify_pair(reports, prepares)
    rows = {arm: audit_arm(arm, reports[arm], binding) for arm in ("structured", "cem")}
    require(all(rows["structured"][k] == rows["cem"][k]
                for k in ("initial_state", "initial_pixel_sha256", "goal_state")), "actual initial state/goal")
    result = {"schema": "p1_common_cp10_pair_audit_20261101_leibniz.v1",
                      "binding": binding, "sources_verified": sources, "arms": rows,
                      "report_sha256": {arm: meta[1] for arm, meta in REPORTS.items()},
                      "audit_source_sha256": {str(Path(p).resolve()): filehash(Path(p).resolve()) for p in
                                              (__file__, cem_audit.__file__, anchor.__file__,
                                               __import__(chain.__module__, fromlist=["x"]).__file__)},
                      "neural_calls": 0, "physical_execution": False,
                      "limitations": ["known root and observed outcome", "dispatch reservations, not actual GPU cost",
                                      "no independent model or physics re-execution", "not equal total compute",
                                      "structured optimizer H3 at all nodes; CEM horizon varies"]}
    serialized = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        path = Path(args.output)
        require(path.is_absolute() and path.parent == RESULTS and path.parent.resolve() == path.parent
                and path.name.startswith("p1_common_cp10_pair_audit_20261101_leibniz_")
                and path.suffix == ".json" and not path.exists() and not path.is_symlink(), "exclusive output")
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"path": str(path), "sha256": filehash(path), "status": "saved_audit_complete"}))
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
