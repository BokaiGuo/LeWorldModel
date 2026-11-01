"""Independent full-coordinate calibration arithmetic; never selects actions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

METRIC_GATES = {"all_selected_prefixes_physically_no_more_goal_greedy_than_raw_zero",
                "all_selected_prefixes_actual_rgb_no_more_goal_greedy_than_zero"}
INTEGRITY_GATES = {"cuda_only", "frozen_parameter_hash_unchanged", "sealed_prefixes_reconstructed",
    "same_root_raw_zero_controls", "all_observed_successors_recompiled", "enough_model_eligible_roots",
    "independent_replay_exact", "root_disjoint_from_physical_calibration", "cohort_binding_valid",
    "prerun_implementation_and_upstream_unchanged", "same_root_shared_suffix_projection"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows: list[dict]) -> dict[str, dict]:
    result = {r["root_sha256"]: r for r in rows}
    if not rows or len(result) != len(rows):
        raise ValueError("nonempty duplicate-free root identities required")
    return result


def vector(value) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (192,) or not np.isfinite(result).all():
        raise ValueError("all 192 finite coordinates required")
    return result


def summarize(report: dict, upstream: dict, previous: dict | None = None) -> dict:
    if report.get("schema") != "pusht_d2p_p1_physical_calibration.v3":
        raise ValueError("corrected physical contract required")
    gates = report.get("gates", {})
    if (set(gates) != INTEGRITY_GATES | METRIC_GATES
            or any(gates[k] is not True for k in INTEGRITY_GATES)
            or any(type(v) is not bool for v in gates.values())
            or report.get("status") != ("pass" if all(gates.values()) else "fail")):
        raise ValueError("complete passing integrity gates and honest scientific status required")
    if report["protocol"]["outcome_roots_opened"] is not False:
        raise ValueError("this diagnostic must not open outcome")
    physical, model = indexed(report["rows"]), indexed(upstream["rows"])
    if physical.keys() != model.keys():
        raise ValueError("model/physical root sets differ")
    prior = None if previous is None else indexed(previous["rows"])
    if prior is not None and prior.keys() != physical.keys():
        raise ValueError("previous physical root set differs")
    rows = []
    for key in sorted(physical):
        row, sealed = physical[key], model[key]
        if row.get("model_eligible") is not True:
            raise ValueError("this all-witness calibration audit requires all fixed candidates")
        margins = {}
        for name in ("selected", "raw_zero_control"):
            branch = row[name]
            forecast, observed = branch["prefix_prediction_before_execution"], branch["suffix_recompile"]
            goal, actual = vector(observed["goal_vector"]), vector(observed["observed_successor_vector"])
            np.testing.assert_array_equal(goal, vector(forecast["goal_vector"]))
            predicted = vector(forecast["predicted_successor_vector"])
            pcost, ocost = float(np.mean((predicted - goal)**2)), float(np.mean((actual - goal)**2))
            if (abs(pcost-forecast["predicted_goal_cost"]) > 1e-6
                    or abs(ocost-observed["observed_successor_goal_cost"]) > 1e-6
                    or forecast["live_action_roundtrip_max_abs"] > 1e-6):
                raise ValueError("stored RGB arithmetic/action differs from complete vectors")
            if (branch["warmup_rgb_sha256"] != sealed["history_rgb_sha256"]
                    or branch["goal_rgb_sha256"] != sealed["goal_rgb_sha256"]):
                raise ValueError("actual/model RGB contexts differ")
            margins[name] = {"predicted_rgb_cost": pcost, "actual_rgb_cost": ocost,
                "pose_cost": branch["after_prefix_label"]["pose_cost"],
                "final_pose_cost": branch["final_label"]["pose_cost"],
                "observed_reachability_value": observed["value"]}
        s, z = (margins[k] for k in ("selected", "raw_zero_control"))
        predicted_margin = s["predicted_rgb_cost"]-z["predicted_rgb_cost"]
        if abs(predicted_margin-sealed["compilation"]["model_margin_vs_zero"]) > 2e-6:
            raise ValueError("executed same-action forecast differs from sealed selection margin")
        out = {"root_sha256": key, "root_id": row["root_id"],
            "selected_index": sealed["compilation"]["selected_proposal_index"],
            "predicted_rgb_margin": predicted_margin,
            "actual_rgb_margin": s["actual_rgb_cost"]-z["actual_rgb_cost"],
            "actual_pose_margin": s["pose_cost"]-z["pose_cost"],
            "final_pose_margin": s["final_pose_cost"]-z["final_pose_cost"],
            "observed_reachability_improvement": z["observed_reachability_value"]-s["observed_reachability_value"],
            "branches": margins}
        if row["selected"]["suffix_recompile"]["projection_sha256"] != row["raw_zero_control"]["suffix_recompile"]["projection_sha256"]:
            raise ValueError("suffix projection differs within root")
        if prior is not None:
            before = prior[key]
            checks = {}
            for name in ("selected", "raw_zero_control"):
                now, old = row[name], before[name]
                checks[name] = {k: now[k] == old[k] for k in (
                    "prefix_action", "successor_rgb_sha256", "successor_state",
                    "suffix_action", "final_rgb_sha256", "final_state")}
            out["previous_physical_exact_checks"] = checks
        rows.append(out)
    rows.sort(key=lambda r: r["root_id"])
    rgb_ok = all(r["actual_rgb_margin"] >= -1e-8 for r in rows)
    pose_ok = all(r["actual_pose_margin"] >= -1e-8 for r in rows)
    if (rgb_ok != report["gates"]["all_selected_prefixes_actual_rgb_no_more_goal_greedy_than_zero"]
            or pose_ok != report["gates"]["all_selected_prefixes_physically_no_more_goal_greedy_than_raw_zero"]):
        raise ValueError("stored metric gates disagree with independent arithmetic")
    return {"schema": "pusht_p1_metric_audit.v1", "root_count": len(rows), "rows": rows,
        "rgb_non_greedy_count": sum(r["actual_rgb_margin"] >= -1e-8 for r in rows),
        "pose_non_greedy_count": sum(r["actual_pose_margin"] >= -1e-8 for r in rows),
        "rgb_pose_sign_disagreement_count": sum((r["actual_rgb_margin"] >= -1e-8) != (r["actual_pose_margin"] >= -1e-8) for r in rows),
        "p1_admitted": False, "outcome_opened": False,
        "scope": "Known calibration roots only; RGB and physical-pose metrics are distinct. No gate relaxation or outcome authorization."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    upstream = Path(report["upstream_p1"]["path"])
    seal = Path(report["prerun_seal"]["path"])
    if digest(upstream) != report["upstream_p1"]["sha256"] or digest(seal) != report["prerun_seal"]["sha256"]:
        raise ValueError("upstream/prerun seal changed")
    for item in json.loads(seal.read_text())["inputs"]:
        if digest(Path(item["snapshot"])) != item["sha256"]:
            raise ValueError("archived prerun input changed")
    result = summarize(report, json.loads(upstream.read_text()),
        None if args.previous is None else json.loads(args.previous.read_text()))
    result["report_sha256"] = digest(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/"summary.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    (args.output_dir/"notes.txt").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    (args.output_dir/"manifest.json").write_text(json.dumps({"analyzer_sha256": digest(Path(__file__)),
        "report_sha256": digest(args.report), "artifacts": {p.name: digest(p) for p in args.output_dir.iterdir()}}, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
