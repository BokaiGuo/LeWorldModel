"""Saved cp10 timing/goal audit and fixed model-only local-risk scoring primitive."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RUN = RESULTS / "codex_p1_cp10_rgb_refresh_runner_20261101_exclusive_run_CODEX01"
RUN_SHA = "d22ce41df39c1a5514e31754f0b6cdeb8ab6e425d92385c426c69828d0df990d"
HIST = RESULTS / "p1_diag32_samebank_tail_20261101_codex_run_CODEX01"
HIST_SHA = "f87f9c6c207076108c158c6c2335f05b2afa688eff9debacab793f92096da0d3"
NAME = "codex_p1_cp10_risk_score_audit_20261101_exclusive"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variants(packet):
    raw = np.asarray(packet, dtype=np.float32)
    if raw.shape != (5, 2) or not np.isfinite(raw).all() or (np.abs(raw) > 1).any():
        raise ValueError("expected finite raw action packet in [-1, 1]")
    pattern = np.tile(np.array([1, -1], np.float32), (5, 1))
    perturbations = [np.zeros((5, 2), np.float32), np.ones((5, 2), np.float32),
                     -np.ones((5, 2), np.float32), pattern, -pattern]
    return [np.clip(raw + np.float32(0.05) * delta, -1, 1).astype("<f4")
            for delta in perturbations]


def evaluate(packet, score_model):
    """Callback returns one-step model MSE; never receives true future observations."""
    scores = [float(score_model(candidate)) for candidate in variants(packet)]
    if not np.isfinite(scores).all():
        raise ValueError("all five model scores must be finite")
    mean = float(np.mean(scores))
    spread = float(max(scores) - min(scores))
    return {"nominal_q": scores[0], "variant_q": scores, "mean_q": mean,
            "range_q": spread, "robust_score": mean + spread}


def select(candidates, score_model):
    if len(candidates) != 2:
        raise ValueError("exactly two preavailable candidates")
    scores = [evaluate(candidate, score_model) for candidate in candidates]
    chosen = min(range(2), key=lambda i: (scores[i]["robust_score"], i))
    return {"chosen_index": chosen, "scores": scores, "predictor_calls": 10}


def build():
    from . import codex_p1_cp10_rgb_refresh_verify_20261101_exclusive as verify
    from . import codex_p1_diag32_fixed_physical_20261101_exclusive as physical
    from . import e3_cp10_four_arm_root0_20261031 as target
    from . import p1_diag32_samebank_saved_20261101_codex as historical

    saved = verify.verify(RUN, RUN_SHA)
    historical.verify(HIST, HIST_SHA)
    entry, encoded, sources = target.inputs()
    prepared, data, expected_root, _, _, _, _ = physical.inputs()
    task, _, _, _ = physical.blind.load(0)
    model_goal = np.asarray(task["goal"], np.float32)
    encoded_goal = np.asarray(encoded["latent"], np.float32)
    ref = data["references"][0]
    if not (np.array_equal(model_goal, encoded_goal)
            and np.array_equal(np.asarray(entry["goal_state"]), ref["goal"])
            and np.array_equal(entry["pixels"], ref["goal_rgb"])
            and encoded["pixel_sha256"] == expected_root["goal_rgb_sha256"]
            and physical.engine.digest(task["goal"]) == prepared["cp10_identity"]["goal_latent_sha256"]):
        raise ValueError("goal image/state/latent provenance mismatch")
    report = json.loads((RUN / "report.json").read_bytes())
    old = json.loads((HIST / "report.json").read_bytes())
    old_selected = old["rows"][0]
    rows = []
    for row in report["rows"]:
        if row["policy"] != "rgb_refresh":
            continue
        cp10, last = row["observations"][10], row["observations"][-1]
        if cp10["goal_rgb_sha256"] != entry["pixel_sha256"] or cp10["goal"] != entry["goal_state"]:
            raise ValueError("live goal drift")
        n = row["control_returned"]
        decision = row["decisions"][1]
        rows.append({"role": row["role"], "cp10_pixel_sha256": cp10["pixel_sha256"],
                     "candidate_scores_hold_optimized": decision["scores"], "dispatched": decision["choice"],
                     "scored_prediction_target_step": 15, "physical_observed_terminal_step": n,
                     "terminal_is_same_time_as_score": n == 15,
                     "counterfactual_candidate_observed_from_same_cp10": False,
                     "physical_endpoint_native_success": row["final_metrics"]["native_success"],
                     "physical_endpoint_strict_success": row["final_metrics"]["wrapped_allxy_success"],
                     "physical_endpoint_joint_xy": float(np.linalg.norm(
                         np.asarray(last["state"][:4]) - np.asarray(last["goal"][:4]))),
                     "physical_endpoint_pixel_sha256": last["pixel_sha256"],
                     "actual_terminal_rgb_latent_q_available": False})
    current = next(r for r in report["rows"] if r["role"] == "selected" and r["policy"] == "rgb_refresh")
    current_cp10 = current["observations"][10]["pixel_sha256"]
    historical_cp10 = old_selected["observations"][10]["pixel_sha256"]
    if current_cp10 == historical_cp10:
        raise ValueError("expected different cp10 checkpoints; re-audit source")
    paths = {"run": RUN / "report.json", "historical": HIST / "report.json",
             "goal_target_receipt": target.TARGET, "source_code": Path(__file__).resolve(),
             "test_code": ROOT / "tests" / ("test_" + NAME + ".py")}
    return {
        "schema": NAME + ".v1", "files": {k: {"path": str(v), "sha256": sha(v)} for k, v in paths.items()},
        "saved_run_verification": saved["status"],
        "goal": {"pixel_sha256": entry["pixel_sha256"], "physical_goal_state": entry["goal_state"],
                 "goal_latent_sha256": physical.engine.digest(task["goal"]),
                 "source": "historical witness root0 step10 image and state, encoded with frozen model",
                 "image_goal_state_latent_binding_verified": True,
                 "semantic_boundary": "latent image MSE and native first-four-position/angle criteria are distinct"},
        "timing": {"prefix_selected_cp10_optimized_pixel_sha256": current_cp10,
                   "prefix_selected_historical_hold_cp10_pixel_sha256": historical_cp10,
                   "same_cp10": False,
                   "interpretation": "historical whole-hold outcome is cp5-paired, not cp10 counterfactual"},
        "rows": rows,
        "wiring_evidence": ["source code appends cp10 encoded RGB and actual prefix/committed packet history",
                            "journal checks input RGB hash, branch order, scoring-call counts and action seals",
                            "saved run does not include cp10 latent or numerical prediction tensors; no numerical replay performed"],
        "scheme": {"development_only": True, "name": "fixed_local_action_perturbation_risk_q",
                   "decision_boundary": 10, "candidate_order": ["own_hold_last_packet", "optimized_last_packet"],
                   "candidate_provenance": "both must be available at cp5; regenerate optimization in deployment",
                   "model_input": "own actual cp10 RGB encoding, actual action history, same goal latent",
                   "variants": "nominal, +/-0.05 both axes, +/-0.05 opposite axes; coordinate clamp [-1,1]",
                   "score": "mean(five one-packet model goal latent MSEs) + (max - min); tie first",
                   "budget_per_decision": {"action_encoder": 10, "predict": 10, "extra_rgb_encode": 0,
                                           "backward": 0},
                   "selection_excludes": ["future physical state", "future image", "reward", "physical success"],
                   "physical_runtime_implemented": False, "model_or_physical_execution_this_audit": False,
                   "rationale_limit": "fixed sensitivity heuristic, not calibrated uncertainty or evidence of improvement"},
        "interpretation": ["diagnostic uses prior outcomes; rule fixed after those outcomes, not preregistered confirmation",
                           "saved cp10 score is model prediction, not observed endpoint RGB latent error",
                           "source target has historical witness provenance; not an independent task or root",
                           "no gate change and no independently confirmed P1 claim"],
    }


if __name__ == "__main__":
    out = RESULTS / (NAME + "_CODEX01.json")
    value = build()
    with out.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(out), "sha256": sha(out)}))
