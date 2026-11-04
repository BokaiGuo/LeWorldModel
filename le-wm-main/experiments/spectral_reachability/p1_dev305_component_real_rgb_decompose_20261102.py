"""Separate model-to-real and real-latent-to-physical ranking on fixed candidates."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_component_tangent_vectors_20261102 as predicted
from . import p1_dev305_two_packet_actual_vectors_20261102 as actual
from . import p1_dev305_two_packet_metric_decompose_20261102 as previous

base, need, ROOT = predicted.base, predicted.need, predicted.ROOT
PRED_SHA = "b87b33fea647f7121c555d1521f2a4a57347e26189b0f0bbf0d9908d5b84c8fa"
PRED_PREP_SHA = "37f245aa3ea2d5b73f6de82d931d3ca3acd6268ab3b4fd7336c1bfc70cf3bebb"
ACTUAL_PREP_SHA = "1e7683056397b656d937088602edb8010b089b64d7aff1e48dcee3bf75ed759a"
OLD_ACTUAL_SHA = previous.REPORT_SHA


def analyze(actual_sha):
    p = base.read(ROOT / "results" / f"{predicted.NAME}_RUN_DEV01/report.json", PRED_SHA)
    p_spec = base.read(ROOT / "results" / f"{predicted.NAME}_PREP_DEV01.json", PRED_PREP_SHA)
    spec = base.read(ROOT / "results" / f"{actual.NAME}_PREP_DEV01.json", ACTUAL_PREP_SHA)
    r = base.read(ROOT / "results" / f"{actual.NAME}_RUN_DEV01/report.json", actual_sha)
    old = base.read(ROOT / "results" / f"{previous.actual.NAME}_RUN_DEV01/report.json", OLD_ACTUAL_SHA)
    physics = base.read(ROOT / "results" / f"{previous.actual.physical.source.NAME}_RUN_DEV01/report.json",
                        previous.actual.physical.saved.REPORT_SHA)
    capture = base.read(ROOT / "results/p1_dev305_diverse_cem_20261102_CAPTURE_DEV01/report.json",
                        previous.CAPTURE_SHA)
    need(spec == actual.protocol() and p_spec == predicted.protocol()
         and p["contract_sha256"] == PRED_PREP_SHA and p_spec["goal"] == spec["goal"]
         and p_spec["model_hash"] == spec["model_hash"] == p["model_before"]
         and r["contract_sha256"] == ACTUAL_PREP_SHA
         and r["model_before"] == r["model_after"] == p["model_before"] == p["model_after"]
         and len(r["rows"]) == len(old["rows"]) == len(physics["rows"]) == 867, "source")
    goal = np.asarray(capture["goal_state_evaluation_only"], dtype=float)
    out = []
    for entry in p["rows"]:
        arm = entry["arm"]
        real = [x for x in r["rows"] if x["arm"] == arm]
        prior = [x for x in old["rows"] if x["arm"] == arm]
        phys = [x for x in physics["rows"] if x["prefix"] == arm]
        need(len(real) == len(prior) == len(phys) == 289, "coverage")
        for k, (a, b, c) in enumerate(zip(real, prior, phys, strict=True)):
            ij = (k // 17, k % 17)
            need((a["first"], a["second"]) == (b["first"], b["second"])
                 == (c["first"], c["second"]) == ij and c["status"] == "fixed_horizon_completed", "index")
        d_real = np.asarray([x["actual_goal_delta"] for x in real], dtype=np.float64)
        d_pred = np.asarray(entry["terminal_goal_deltas"], dtype=np.float64).reshape(289, 192)
        need(d_real.shape == d_pred.shape and np.isfinite(d_real).all(), "latent")
        recomputed_mse = (d_real ** 2).mean(axis=1)
        need(np.allclose(recomputed_mse, [x["actual_goal_mse"] for x in real], atol=1e-6, rtol=0)
             and np.allclose(recomputed_mse, [x["actual_goal_mse"] for x in prior], atol=1e-6, rtol=0), "mse")
        pred_scores, _, _ = predicted.component_scores(d_pred, entry["jacobian"])
        real_scores, _, _ = predicted.component_scores(d_real, entry["jacobian"])
        rewards = -np.asarray([x["transitions"][-1]["reward"] for x in phys])
        states = np.asarray([x["transitions"][-1]["state"] for x in phys])
        positions = np.linalg.norm(states[:, 2:4] - goal[2:4], axis=1)
        angles = np.abs((states[:, 4] - goal[4] + np.pi) % (2 * np.pi) - np.pi)
        metrics = {}
        for name in ("object", "pusher", "joint"):
            ps, rs = pred_scores[name], real_scores[name]
            need(np.allclose(ps.reshape(17, 17), entry["scores"][name], atol=1e-12, rtol=1e-12), "scores")
            ip, ir = int(np.argmin(ps)), int(np.argmin(rs))
            def details(i):
                return {"pair": [i // 17, i % 17], "reward": float(-rewards[i]),
                        "object_position_error": float(positions[i]), "object_angle_error": float(angles[i])}
            metrics[name] = {
                "predicted_vs_real_score": previous.pair_counts(ps, rs, 1e-6),
                "real_score_vs_negative_reward": previous.pair_counts(rs, rewards, 1e-6),
                "real_score_vs_object_position": previous.pair_counts(rs, positions, 1e-6),
                "predicted_choice": details(ip), "actual_rgb_oracle_choice": details(ir)}
        out.append({"arm": arm, "metrics": metrics,
                    "actual_mse_max_abs_difference": float(np.max(np.abs(
                        recomputed_mse - np.asarray([x["actual_goal_mse"] for x in prior]))))})
    return {"schema": "p1_dev305_component_real_rgb_decompose_20261102.v1", "rows": out,
            "source_actual_sha256": actual_sha, "scope": "post-outcome same-root shared candidates; real RGB oracle is not deployable",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(base.write_new(args.output, analyze(args.actual_sha)))
