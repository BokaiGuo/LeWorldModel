"""Compare nonlinear model, linearized model and real candidate endpoints."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np

from . import p1_dev304_qgoal_real_rgb_20261102 as source

base, saved, physical, model_run = source.base, source.saved, source.physical, source.model_run
ROOT = source.ROOT/"results"
RUN = ROOT/f"{source.NAME}_RUN_DEV01"
SPEC_SHA = "a71bca0efa55352977bd3284650d49420ac5ac20060f9381e59bf86886e650d4"
REPORT_SHA = "bc6342f59b8b07b952ea388ab4f1b2ce50850b9156a7c2e01a3edf19ca18bd0d"
need = source.need


def analyze():
    spec = base.read(ROOT/f"{source.NAME}_PREP_DEV01.json", SPEC_SHA)
    need(spec == source.protocol(), "contract")
    report = base.read(RUN/"report.json", REPORT_SHA)
    need(report["contract_sha256"] == SPEC_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"]
         and report["counts"] == {"encoder": {"calls": 19, "samples": 19},
                                  "projector": {"calls": 19, "samples": 19}}, "encoding ledger")
    model = base.read(ROOT/f"{model_run.NAME}_RUN_DEV01/report.json", physical.MODEL_SHA)
    physical_data = saved.analyze()
    need(len(report["rows"]) == len(spec["candidates"]) == len(physical_data["rows"]) == 18, "roster")
    goal = np.asarray(report["goal_latent"], float)
    need(goal.shape == (192,) and np.isfinite(goal).all(), "goal latent")
    metric = np.asarray(model["metric"], float)
    need(metric.shape == (192, 192) and np.isfinite(metric).all()
         and np.allclose(goal, model["metric_info"]["goal_latent"], rtol=1e-5, atol=1e-6), "metric/goal")
    rows = []
    for encoded, ref, truth in zip(report["rows"], spec["candidates"], physical_data["rows"], strict=True):
        need(encoded["arm"] == ref["arm"] and encoded["prefix"] == truth["prefix"]
             and encoded["candidate"] == truth["candidate"]
             and base.read(RUN/(encoded["arm"]+".json")) == encoded, "row binding")
        model_row = next(r for r in model["rows"] if r["arm"] == encoded["prefix"])
        c = model_row["candidates"][encoded["candidate"]]
        actual = np.asarray(encoded["terminal_latent"], float)
        predicted = np.asarray(c["terminal"], float)
        anchor = np.asarray(model_row["anchor_terminal"], float)
        jac = np.asarray(model_row["jacobian"], float)
        delta_action = np.asarray(c["actions"], float).reshape(-1)-np.asarray(model_row["candidates"][0]["actions"], float).reshape(-1)
        need(actual.shape == predicted.shape == anchor.shape == (192,)
             and jac.shape == (192, 20) and np.isfinite(actual).all(), "latent shapes")
        delta_pred = predicted-anchor
        delta_linear = jac@delta_action
        denominator = float(np.linalg.norm(delta_pred))
        # The anchor has zero action displacement. A separate rollout can still
        # differ by floating-point noise, so its ratio has no meaningful scale.
        linear_error = None if not np.any(delta_action) or denominator < 1e-12 else float(np.linalg.norm(delta_pred-delta_linear)/denominator)
        predicted_q = float(np.mean((metric @ (predicted-goal))**2))
        actual_q = float(np.mean((metric @ (actual-goal))**2))
        need(np.isclose(predicted_q, c["score"], rtol=1e-5, atol=1e-7), "weighted score binding")
        rows.append({"arm": encoded["arm"], "prefix": encoded["prefix"], "candidate": encoded["candidate"],
                     "kind": c["kind"], "model_selected": truth["model_selected"],
                     "pred_goal_cost": float(np.mean((predicted-goal)**2)),
                     "real_goal_cost": float(np.mean((actual-goal)**2)),
                     "pred_q_cost": predicted_q, "real_q_cost": actual_q,
                     "prediction_mse": float(np.mean((predicted-actual)**2)),
                     "linear_relative_error": linear_error, "model_delta_norm": denominator,
                     "position_error": truth["terminal"]["object_position_error"],
                     "angle_error": truth["terminal"]["object_angle_error"], "reward": truth["reward"]})
    pairs = []
    for prefix in ("original", "coherent"):
        group = [r for r in rows if r["prefix"] == prefix]
        for a, b in combinations(group, 2):
            def sign(k):
                return int(np.sign(a[k]-b[k]))
            pairs.append({"prefix": prefix, "candidates": [a["candidate"], b["candidate"]],
                          "prediction": sign("pred_goal_cost"), "real_latent": sign("real_goal_cost"),
                          "pred_q": sign("pred_q_cost"), "real_q": sign("real_q_cost"),
                          "position": sign("position_error"), "angle": sign("angle_error"),
                          "negative_reward": -sign("reward")})
    def count(left, right):
        comparable = [p for p in pairs if p[left] and p[right]]
        return {"agreement": sum(p[left] == p[right] for p in comparable),
                "disagreement": sum(p[left] != p[right] for p in comparable),
                "ties": len(pairs)-len(comparable)}
    return {"schema": source.NAME+".analysis.v1", "rows": rows, "pairs": pairs,
            "pred_vs_real_latent": count("prediction", "real_latent"),
            "pred_q_vs_real_q": count("pred_q", "real_q"),
            "real_q_vs_position": count("real_q", "position"),
            "real_q_vs_angle": count("real_q", "angle"),
            "real_q_vs_reward": count("real_q", "negative_reward"),
            "real_latent_vs_position": count("real_latent", "position"),
            "real_latent_vs_angle": count("real_latent", "angle"),
            "real_latent_vs_reward": count("real_latent", "negative_reward"),
            "classification": "post_outcome_one_known_root", "independent_confirmation": False,
            "p1_admission": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(base.write_new(p.parse_args().output, analyze()))
