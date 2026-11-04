"""Saved-data audit: selector and action chart, no model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup

SOURCE_SHA = "14ed522959434b1c936960606b99103e32145842705a63b24b6f770c66cbc1b4"
MARGIN = 1e-6


def audit_row(row, mean, std):
    if row["status"] != "complete":
        raise ValueError("incomplete root")
    receipt = row["compiler_receipt"]
    rows = receipt["proposal_rows"]
    if len(rows) != 21 or [r["proposal_index"] for r in rows] != list(range(21)):
        raise ValueError("candidate roster mismatch")
    bank = np.asarray(row["bank_params"], dtype=np.float32)
    raw = np.asarray(row["prefix_raw"], dtype=np.float32)
    normalized = np.asarray(row["prefix_normalized"], dtype=np.float32)
    tail_raw = np.asarray(row["tail_raw"], dtype=np.float32)
    tail_normalized = np.asarray(row["tail_normalized"], dtype=np.float32)
    endpoints = np.asarray(row["predicted_endpoints"], dtype=np.float64)
    if (bank.shape != (21, 1, 10) or raw.shape != (21, 5, 2) or normalized.shape != (21, 1, 10)
            or tail_raw.shape != (5, 10, 2) or tail_normalized.shape != (5, 2, 10)
            or endpoints.shape != (21, 5, 192)):
        raise ValueError("saved array shape mismatch")
    for array in (bank, raw, normalized, tail_raw, tail_normalized, endpoints):
        if not np.isfinite(array).all():
            raise ValueError("nonfinite saved array")
    for actions, chart in ((raw, normalized), (tail_raw, tail_normalized)):
        if np.any(np.abs(actions) > 1.0 + 1e-6):
            raise ValueError("out-of-bounds raw action")
        for a, n in zip(actions, chart, strict=True):
            decoded = setup.p0._normalized_blocks_to_raw_actions(n, mean, std, action_dim=2)
            if not np.array_equal(decoded, a):
                raise ValueError("raw action binding mismatch")
    baseline = rows[0]
    base_f = np.asarray(baseline["final_costs"], dtype=np.float64)
    if base_f.shape != (5,) or not np.isfinite(base_f).all():
        raise ValueError("invalid baseline scores")
    candidates = []
    for i, r in enumerate(rows):
        final = np.asarray(r["final_costs"], dtype=np.float64)
        recalculated = np.asarray(row["recalculated_final_costs"][i], dtype=np.float64)
        delta = final - base_f
        adjusted = delta[1:] - delta[0]
        if (final.shape != (5,) or not np.isfinite(final).all()
                or not np.allclose(final, recalculated, atol=1e-7, rtol=1e-5)
                or not np.allclose(delta, r["per_tail_delta"], atol=1e-12, rtol=1e-12)
                or not np.allclose(adjusted, r["active_tail_neutral_adjusted_delta"], atol=1e-12, rtol=1e-12)):
            raise ValueError("saved score or delta mismatch")
        eligible = (i > 0 and float(r["immediate_cost"]) >= float(baseline["immediate_cost"]) + MARGIN
                    and float(delta.mean()) <= -MARGIN and bool((delta <= 0).all())
                    and float(adjusted.mean()) <= -MARGIN and bool((adjusted <= 0).all()))
        if bool(r["eligible"]) != eligible:
            raise ValueError("selector flag mismatch")
        if eligible:
            candidates.append((float(delta.sum()), i))
    selected = min(candidates)[1] if candidates else None
    if selected != receipt["selected_proposal_index"]:
        raise ValueError("winner mismatch")
    digest = lambda array: hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return {"seed": row["seed"], "candidates": len(rows), "eligible": len(candidates),
            "selected": selected, "prefix_raw_sha256": digest(raw), "tail_raw_sha256": digest(tail_raw),
            "saved_endpoint_score_independent_recomputation": "unavailable_goal_latent_not_saved"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if setup._sha256_file(args.report) != SOURCE_SHA:
        raise RuntimeError("source report SHA mismatch")
    data = json.loads(args.report.read_text())
    pf = json.loads(args.preflight.read_text())
    stats = Path(pf["action_stats_path"])
    if setup._sha256_file(stats) != pf["action_stats_sha256"]:
        raise RuntimeError("normalizer hash mismatch")
    if not data["plan_postcheck"] or not data["frozen_parameters_unchanged"]:
        raise RuntimeError("producer postchecks failed")
    mean, std = setup._load_action_stats(stats, "swm/ReacherDMControl-v0")
    rows = [audit_row(row, mean, std) for row in data["rows"]]
    if [r["seed"] for r in rows] != [r["seed"] for r in pf["roots"]]:
        raise RuntimeError("root roster mismatch")
    result = {"status": "selector_and_action_chart_verified_goal_q_not_independently_recomputed",
              "source_sha256": SOURCE_SHA, "preflight_sha256": setup._sha256_file(args.preflight),
              "runner_sha256": setup._sha256_file(Path(__file__)), "rows": rows,
              "model_inference": False, "physical_execution": False, "P1_admission": False}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"status": result["status"], "sha256": setup._sha256_file(args.output)}))


if __name__ == "__main__":
    main()
