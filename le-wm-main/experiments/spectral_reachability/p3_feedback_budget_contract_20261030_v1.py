"""Freeze and inspect a measured-work feedback comparison; never execute a model."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.p3_matched_order_feedback_numeric_bridge_codex_20261029 import reconstruct

NAME = "p3_feedback_budget_contract_20261030_v1"
PREPARE_SHA = "b6c8dcf650d4e752f6306ed4b6f9434861c224ced51c7c8de25c0e7ca6d323f2"
ROSTER = [(0, 0, "fixed_order"), (1, 0, "signal_order"), (2, 1, "fixed_order"), (3, 1, "signal_order")]
EXPECTED = {
    "VJP_directions": 768, "action_encoder_calls": 52, "action_encoder_context_tokens": 156,
    "action_encoder_samples": 52, "array_SVD_calls": 6, "encoder_calls": 0,
    "encoder_samples": 0, "grad_API": 768, "jacobian_API": 4, "metric_calls": 8,
    "model_SVD_calls": 0, "pred_proj_calls": 52, "pred_proj_rows": 156, "predict_API": 52,
    "predict_context_tokens": 156, "predict_samples": 52, "predictor_calls": 52,
    "predictor_context_tokens": 156, "predictor_samples": 52, "program_compile_calls": 2,
    "projector_calls": 0, "projector_samples": 0, "raw_backward_callbacks": 768,
    "replay_calls": 18, "schedule_decisions": 2, "signal_compile_calls": 4, "solve_calls": 2,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_pinned(path, expected):
    p = Path(path)
    if not p.is_file() or p.is_symlink() or sha(p) != expected:
        raise ValueError("input hash or regular file mismatch")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    return json.loads(p.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def contract():
    return {
        "schema": NAME + ".contract", "host_prepare_sha256": PREPARE_SHA,
        "verifier_sha256": sha(__file__), "roster": [list(x) for x in ROSTER],
        "budget_definition": "complete two blocks H1+H3, original paid signal, four candidates and required replay",
        "numerical_attempted_and_completed": EXPECTED,
        "observation_encoding_boundaries": [5, 10, 15, 20],
        "plan_steps": 20, "warmup_steps": 10, "checkpoints": [0, 5, 10, 15, 20],
        "primary_endpoint": "native7_distance at plan step 20; signal minus fixed",
        "secondary": ["native_success", "joint_xy", "object_xy", "velocity_l2", "native7_distance at all checkpoints"],
        "comparison": "measured counts and hook shape multiset separately; fixed same-root roster",
        "failure_rule": "retain four slots and attempted work; no retries, padding, or outcome-based selection",
        "timing": "same control step, not same wall time; no equal FLOPs claim",
        "fresh_roots": 0, "scientific_pass": False,
    }


def save_new(path, obj):
    raw = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


def metrics(state, goal):
    s, g = np.asarray(state, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    if s.shape != (7,) or g.shape != (7,) or not np.isfinite(s).all() or not np.isfinite(g).all():
        raise ValueError("invalid state or goal")
    d = s - g
    xy, angle = float(np.linalg.norm(d[:4])), abs(float(d[4]))
    return {"native7_distance": float(np.linalg.norm(d)), "joint_xy": xy,
            "object_xy": float(np.linalg.norm(d[2:4])), "velocity_l2": float(np.linalg.norm(d[5:])),
            "native_success": bool(xy < 20 and min(angle, 2 * math.pi - angle) < math.pi / 9)}


def shapes(events):
    return Counter(json.dumps([e["resource"], e["shape"], e["directions"], e["status"]],
                              sort_keys=True, separators=(",", ":")) for e in events)


def compare_pair(a, b):
    if a is None or b is None:
        return {"measured_vector_match": None, "hook_shape_multiset_match": None, "checkpoints": None}
    if a["root_sha256"] != b["root_sha256"] or a["goal"] != b["goal"] or a["cp0_identity"] != b["cp0_identity"]:
        raise ValueError("same-root identity mismatch")
    if a["attempted"] != EXPECTED or a["completed"] != EXPECTED or b["attempted"] != EXPECTED or b["completed"] != EXPECTED:
        raise ValueError("complete slot numerical budget mismatch")
    out = {}
    for t in ("0", "5", "10", "15", "20"):
        x, y = a["checkpoints"][t], b["checkpoints"][t]
        out[t] = {"signal_minus_fixed": {k: y[k] - x[k] for k in x if k != "native_success"},
                  "fixed_success": x["native_success"], "signal_success": y["native_success"]}
    return {"measured_vector_match": True, "hook_shape_multiset_match": a["shape_counts"] == b["shape_counts"],
            "checkpoints": out}


def inspect(run, receipt_path, receipt_sha, prepared_path, contract_path, contract_sha):
    frozen = read_pinned(contract_path, contract_sha)
    if frozen != contract():
        raise ValueError("contract or verifier drift")
    read_pinned(prepared_path, PREPARE_SHA)
    run = Path(run)
    receipt = read_pinned(receipt_path, receipt_sha)
    report_sha = receipt.get("report_sha256")
    report = read_pinned(run / "report.json", report_sha)
    completion = read_pinned(run / "completion.json", receipt.get("completion_sha256"))
    if (report.get("prepare_sha256") != PREPARE_SHA or completion.get("report_sha256") != report_sha
            or report.get("scientific_pass") is not False or receipt.get("scientific_pass") is not False
            or completion.get("scientific_pass") is not False):
        raise ValueError("source, completion, or scientific flag mismatch")
    rows = report["rows"]
    if [(r["slot"], r["root"], r["policy"]) for r in rows] != ROSTER:
        raise ValueError("roster mismatch")
    summary = {"schema": NAME + ".saved", "contract_sha256": contract_sha, "report_sha256": report_sha,
               "receipt_sha256": receipt_sha, "slots": [None] * 4, "pairs": [],
               "slot_statuses": [r["status"] for r in rows], "scientific_pass": False,
               "pre_run_external_timestamp_verified": False, "equal_FLOPs": False,
               "limitations": ["strict receipt is inherited, not re-executed", "known roots only",
                               "chronological preregistration requires external immutable timestamp",
                               "no equal total compute or equal wall-time claim"]}
    if not receipt.get("full_saved_strict"):
        summary["pairs"] = [compare_pair(None, None), compare_pair(None, None)]
        summary["status"] = "partial_unmatched_unknown"
        return summary
    if (receipt.get("status") != "four_saved_feedback_engineering_verified_known_roots"
            or report.get("status") != "four_complete_candidate" or completion.get("complete") is not True
            or len(receipt["slots"]) != 4):
        raise ValueError("strict receipt status mismatch")
    journal = run / "events.jsonl"
    if sha(journal) != report["journal_sha256"]:
        raise ValueError("journal hash mismatch")
    chunks, obs, ret, roots, enc = [], [{} for _ in rows], [{} for _ in rows], {}, [[] for _ in rows]
    slot = None
    with journal.open() as f:
        for line in f:
            e = json.loads(line)
            r, d = e["resource"], e["detail"]
            if r == "matched_slot_attempt":
                slot = d["slot"]
            elif r == "matched_slot_return":
                slot = None
            elif r == "matched_numeric_chunk":
                chunks.append(e)
            elif slot is not None and r == "native_matched_root_accepted":
                roots[slot] = d
            elif slot is not None and r == "native_fresh_observation" and d["receipt"]["phase"] == "plan":
                o = d["receipt"]
                if o["step"] in obs[slot]:
                    raise ValueError("duplicate observation")
                obs[slot][o["step"]] = o
            elif slot is not None and r == "native_control_return":
                if d["step"] in ret[slot]:
                    raise ValueError("duplicate return")
                ret[slot][d["step"]] = d
            elif slot is not None and r == "encoding_complete":
                enc[slot].append(d["boundary"])
    for i in range(4):
        authority = receipt["slots"][i]
        if (authority.get("slot") != i or authority.get("physical_steps") != 20
                or authority.get("encoding_boundaries") != 4 or authority.get("native_calls") != 22
                or authority.get("observed_context_rebuilt") is not True):
            raise ValueError("strict slot counts mismatch")
        wrapped = reconstruct(chunks, i)
        finals = [v["payload"]["result"] for v in wrapped if v["payload"].get("kind") == "slot_final"]
        if len(finals) != 1 or set(obs[i]) != set(range(21)) or set(ret[i]) != set(range(1, 21)) or enc[i] != [5, 10, 15, 20]:
            raise ValueError("missing saved observations or boundaries")
        final, goal = finals[0], roots[i]["goal"]
        if final["policy"] != ROSTER[i][2] or sorted(b["horizon"] for b in final["blocks"]) != [1, 3]:
            raise ValueError("policy or horizon mismatch")
        ledger = final["resource_receipt"]
        cp = {}
        for t in (0, 5, 10, 15, 20):
            if obs[i][t]["goal"] != goal:
                raise ValueError("goal mismatch")
            m = metrics(obs[i][t]["state"], goal)
            if t and (not math.isclose(-m["native7_distance"], ret[i][t]["reward"], abs_tol=1e-9)
                      or m["native_success"] != ret[i][t]["terminated"]):
                raise ValueError("native return mismatch")
            cp[str(t)] = m
        summary["slots"][i] = {
            "root_sha256": roots[i]["root_sha256"], "goal": goal,
            "cp0_identity": [obs[i][0]["snapshot"], obs[i][0]["pixel_sha256"]],
            "horizon_order": [b["horizon"] for b in final["blocks"]],
            "attempted": ledger["attempted"], "completed": ledger["completed"],
            "shape_counts": dict(sorted(shapes(ledger["events"]).items())), "checkpoints": cp,
            "numeric_ledger_wall_ns": ledger["wall_ns"], "cuda_ns": ledger["cuda_ns"],
            "FLOPs": ledger["FLOPs"], "slot_end_to_end_wall_ns": None,
            "observation_encoding_boundaries": enc[i], "physical_steps": 20,
        }
    summary["pairs"] = [compare_pair(summary["slots"][0], summary["slots"][1]),
                        compare_pair(summary["slots"][2], summary["slots"][3])]
    summary["status"] = "complete_saved_measured_work_only"
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--prepared", required=True)
    f.add_argument("--output", required=True)
    v = sub.add_parser("inspect")
    for field in ("run", "receipt", "receipt-sha", "prepared", "contract", "contract-sha", "output"):
        v.add_argument("--" + field, required=True)
    a = p.parse_args()
    if a.command == "freeze":
        read_pinned(a.prepared, PREPARE_SHA)
        result = contract()
    else:
        result = inspect(a.run, a.receipt, a.receipt_sha, a.prepared, a.contract, a.contract_sha)
    print(save_new(a.output, result))


if __name__ == "__main__":
    main()
