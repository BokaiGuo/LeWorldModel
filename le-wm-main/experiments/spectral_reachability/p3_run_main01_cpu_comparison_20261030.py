"""Saved-only comparison for the consumed four-slot P3 run. No model or simulator."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.p3_matched_order_feedback_numeric_bridge_codex_20261029 import reconstruct

REPORT_SHA = "02b5f2094de56faf807bbb552b43492b05021dc7caac79f7966ce4734da0c8cd"
RECEIPT_SHA = "0820addb1025ae79b37172767ce34e43ca76350825b90903752d2a6f710d6b23"
ROSTER = [(0, "fixed_order"), (0, "signal_order"), (1, "fixed_order"), (1, "signal_order")]
CHECKPOINTS = (0, 5, 10, 15, 20)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(state, goal):
    s, g = np.asarray(state, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    if s.shape != (7,) or g.shape != (7,) or not np.isfinite(s).all() or not np.isfinite(g).all():
        raise ValueError("invalid native state/goal")
    d = s - g
    angle = abs(float(d[4]))
    native_angle = min(angle, 2 * math.pi - angle)
    xy = float(np.linalg.norm(d[:4]))
    distance = float(np.linalg.norm(d))
    return {"agent_xy": float(np.linalg.norm(d[:2])), "object_xy": float(np.linalg.norm(d[2:4])),
            "joint_xy": xy, "native_angle_term": native_angle,
            "velocity_l2": float(np.linalg.norm(d[5:])), "native7_distance": distance,
            "native7_reward": -distance, "native_success": bool(xy < 20 and native_angle < math.pi / 9)}


def analyze(directory, receipt_path):
    directory, receipt_path = Path(directory), Path(receipt_path)
    report_path = directory / "report.json"
    if digest(report_path) != REPORT_SHA or digest(receipt_path) != RECEIPT_SHA:
        raise ValueError("untrusted report or receipt")
    report, receipt = json.loads(report_path.read_text()), json.loads(receipt_path.read_text())
    if not receipt["full_saved_strict"] or receipt["report_sha256"] != REPORT_SHA:
        raise ValueError("receipt does not authorize saved input")
    journal = directory / "events.jsonl"
    if digest(journal) != report["journal_sha256"]:
        raise ValueError("journal hash mismatch")
    rows = report["rows"]
    if [(r["root"], r["policy"]) for r in rows] != ROSTER:
        raise ValueError("unexpected roster")
    chunks, observations, returns, roots, meters, encoding = [], [{} for _ in rows], [{} for _ in rows], {}, [{} for _ in rows], [0] * 4
    slot = None
    for line in journal.open():
        event = json.loads(line)
        r, d, kind = event["resource"], event["detail"], event["kind"]
        if r == "matched_slot_attempt":
            slot = d["slot"]
        elif r == "matched_slot_return":
            slot = None
        elif r == "matched_numeric_chunk":
            chunks.append(event)
        elif slot is not None and r == "native_matched_root_accepted":
            roots[slot] = d
        elif slot is not None and r == "native_fresh_observation" and d["receipt"]["phase"] == "plan":
            obs = d["receipt"]
            if obs["step"] in observations[slot]:
                raise ValueError("duplicate observation")
            observations[slot][obs["step"]] = obs
        elif slot is not None and r == "native_control_return":
            if d["step"] in returns[slot]:
                raise ValueError("duplicate return")
            returns[slot][d["step"]] = d
        elif slot is not None and r.startswith("native_meter/") and kind == "intent":
            name = r.removeprefix("native_meter/")
            meters[slot][name] = meters[slot].get(name, 0) + d["units"]
        elif slot is not None and r == "encoding_complete":
            encoding[slot] += 1
    output = []
    for i, row in enumerate(rows):
        wrapped = reconstruct(chunks, i)
        final = [v["payload"]["result"] for v in wrapped if v["payload"].get("kind") == "slot_final"]
        if len(final) != 1 or set(observations[i]) != set(range(21)) or set(returns[i]) != set(range(1, 21)):
            raise ValueError("incomplete fixed slot")
        goal = roots[i]["goal"]
        cp = {}
        for t in CHECKPOINTS:
            obs = observations[i][t]
            if obs["goal"] != goal:
                raise ValueError("goal mismatch")
            m = metrics(obs["state"], goal)
            if t and (not math.isclose(m["native7_reward"], returns[i][t]["reward"], abs_tol=1e-9)
                      or m["native_success"] != returns[i][t]["terminated"]):
                raise ValueError("native metric disagrees with saved return")
            cp[str(t)] = m
        receipt_counts = receipt["slots"][i]
        output.append({"slot": i, "root": row["root"], "policy": row["policy"],
                       "root_sha256": roots[i]["root_sha256"], "goal": goal,
                       "checkpoints": cp, "horizon_order": [b["horizon"] for b in final[0]["blocks"]],
                       "resource_completed": final[0]["resource_receipt"]["completed"],
                       "physical_meter_intents": meters[i], "encoding_boundaries": encoding[i],
                       "native_calls": receipt_counts["native_calls"], "slot_wall_seconds": None,
                       "numerical_ledger_wall_ns": final[0]["resource_receipt"]["wall_ns"],
                       "numerical_inclusive_operation_wall_ns": final[0]["resource_receipt"]["inclusive_operation_wall_ns"],
                       "numerical_wall_semantics": final[0]["resource_receipt"]["wall_semantics"],
                       "cuda_ns": final[0]["resource_receipt"]["cuda_ns"],
                       "FLOPs": final[0]["resource_receipt"]["FLOPs"]})
    pairs = []
    for a, b in ((0, 1), (2, 3)):
        if (output[a]["root_sha256"] != output[b]["root_sha256"]
                or output[a]["goal"] != output[b]["goal"]
                or observations[a][0]["snapshot"] != observations[b][0]["snapshot"]
                or observations[a][0]["pixel_sha256"] != observations[b][0]["pixel_sha256"]):
            raise ValueError("paired root identity mismatch")
        delta = {}
        for t in CHECKPOINTS:
            left, right = output[a]["checkpoints"][str(t)], output[b]["checkpoints"][str(t)]
            delta[str(t)] = {k: right[k] - left[k] for k in left if k != "native_success"}
        pairs.append({"root": output[a]["root"], "signal_minus_fixed": delta})
    return {"schema": "p3_run_main01_cpu_comparison_v1", "report_sha256": REPORT_SHA,
            "receipt_sha256": RECEIPT_SHA, "journal_sha256": report["journal_sha256"],
            "report_wall_seconds_all_slots": report["elapsed_seconds"], "slots": output, "pairs": pairs,
            "limitations": ["two known roots, not fresh", "same control step is not equal wall or compute",
                            "native call counters exclude separately reported observation encoding",
                            "no full per-slot wall, CUDA time or FLOPs; numerical ledger wall is partial",
                            "descriptive only; no generalization"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.receipt)
    with open(args.output, "x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
