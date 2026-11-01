"""Saved-only comparison of selected q1 and historical full-hold at cp10."""
import hashlib
import json
import math
from pathlib import Path

from . import codex_p1_cp5_q1_feedback_verify_20261101_exclusive as new_verify
from . import codex_p1_diag32_fixed_physical_verify_20261101_exclusive as hold_verify

ROOT = hold_verify.ROOT
NEW_DIR = ROOT / "results/codex_p1_cp5_q1_feedback_runner_20261101_exclusive_run_CODEX01"
NEW_SHA = "518e74f19be37747fcfdaba4fbb1bea6b1e48c5819abbc4dfc3931353608a0dd"
HOLD_DIR = hold_verify.DIRECTORY
IDENTITY = ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256", "goal_pose")


def identity(a, b):
    result = {key: a["observations"][10][key] == b["observations"][10][key] for key in IDENTITY}
    result["first10_actions"] = (len(a["executed_raw"]) >= 10 and len(b["executed_raw"]) >= 10
                                 and a["executed_raw"][:10] == b["executed_raw"][:10])
    return result


def joint_xy(state, goal):
    return math.sqrt(sum((float(a)-float(b)) ** 2 for a, b in zip(state[:4], goal[:4], strict=True)))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(row, goal):
    state = row["transitions"][-1]["state"]
    metrics = row["final_metrics"]
    return {"status": row["status"], "returned": row["control_returned"],
            "unobserved": 15 - row["control_returned"], "native_success": metrics["native_success"],
            "wrapped_allxy_success": metrics["wrapped_allxy_success"],
            "joint_xy": joint_xy(state, goal),
            "pusher_xy": math.dist(state[:2], goal[:2]),
            "object_xy": metrics["object_position_error"],
            "wrapped_object_angle_error": metrics["wrapped_object_angle_error"],
            "final_state": state, "terminal": row["transitions"][-1]["terminated"],
            "truncated": row["transitions"][-1]["truncated"]}


def audit():
    new_check = new_verify.verify(NEW_DIR, NEW_SHA)
    old_check = hold_verify.verify()
    report = json.loads((NEW_DIR / "report.json").read_bytes())
    current = next(r for r in report["rows"] if r["role"] == "selected" and r["policy"] == "q1_feedback")
    historical = json.loads((HOLD_DIR / "report.json").read_bytes())["row"]
    matches = identity(current, historical)
    if not all(matches.values()):
        raise ValueError("cp10 identity mismatch: " + json.dumps(matches))
    activation_path = Path(json.loads((NEW_DIR / "activation_binding.json").read_bytes())["path"])
    plan = json.loads(activation_path.read_bytes())
    candidates = plan["candidates"]["selected"]
    if (current["executed_raw"][10:] != candidates["optimized"][1]
            or historical["executed_raw"][10:] != candidates["hold"][1]):
        raise ValueError("saved last-packet labels/actions do not match")
    decision = current["decisions"][1]
    if decision["boundary"] != 10 or decision["choice"] != "optimized":
        raise ValueError("unexpected saved cp10 choice")
    goal = current["observations"][10]["goal"]
    return {"status": "verified_saved_only", "development_only": True,
            "outcome_observed_before_audit": True, "independent_confirmation": False, "P1_admission": False,
            "identity": matches, "cp10_pixel_sha256": current["observations"][10]["pixel_sha256"],
            "cp10_state": current["observations"][10]["state"],
            "cp10_goal": goal, "cp10_goal_rgb_sha256": current["observations"][10]["goal_rgb_sha256"],
            "model_scores_goal_latent_mse_hold_then_optimized": decision["scores"],
            "model_scoring_source": "selected q1 cp10 actual RGB scoring; no recomputation",
            "hold": summarize(historical, goal), "optimized": summarize(current, goal),
            "comparison": "separate historical episodes, identical saved first10/cp10 identity; not a new rollout",
            "source_checks": {"new_status": new_check["status"], "hold_status": old_check["status"],
                              "new_report_sha256": NEW_SHA, "hold_report_sha256": hold_verify.REPORT_SHA,
                              "new_contract_sha256": report["activation_sha256"],
                              "hold_contract_sha256": old_check["contract_sha256"],
                              "audit_source_sha256": sha(Path(__file__))},
            "new_neural_calls": 0, "new_physical_calls": 0}


def main():
    out = ROOT / "results/codex_p1_selected_cp10_hold_audit_20261101_exclusive_CODEX01.json"
    result = audit()
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(out), "sha256": sha(out)}))


if __name__ == "__main__":
    main()
