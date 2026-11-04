"""Join physical, predicted, and real-RGB objective on the same candidate bank."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_two_packet_actual_metric_20261102 as actual

base, need, run, ROOT = actual.base, actual.need, actual.run, actual.ROOT
PREP_SHA = "5973ea288519667af36f1b7ba1a4c64cec672f007b14ceeb322d78c1c58eb7c3"
REPORT_SHA = "48ccadd04c75865a8eb9f980f9ca578ebe14756a9336489ee11ce060b3c50b29"
CAPTURE_SHA = "104ce4f480a783b7456518a839af649b0e48d01b8fef5f7d1668e5c8312a53fb"


def pair_counts(left, right, tolerance):
    i, j = np.triu_indices(len(left), 1)
    a, b = left[i] - left[j], right[i] - right[j]
    ties = (np.abs(a) <= tolerance) | (np.abs(b) <= tolerance)
    return {"same": int(np.sum(~ties & (a * b > 0))),
            "reversed": int(np.sum(~ties & (a * b < 0))),
            "ties": int(np.sum(ties))}


def analyze():
    spec = base.read(ROOT / "results" / f"{actual.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT / "results" / f"{actual.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    model_spec = base.read(ROOT / "results" / f"{actual.model_saved.model.NAME}_PREP_DEV01.json",
                           actual.model_saved.CONTRACT_SHA)
    model = base.read(ROOT / "results" / f"{actual.model_saved.model.NAME}_RUN_DEV01/report.json",
                      actual.model_saved.REPORT_SHA)
    physics = base.read(ROOT / "results" / f"{actual.physical.source.NAME}_RUN_DEV01/report.json",
                        actual.physical.saved.REPORT_SHA)
    capture = base.read(ROOT / "results" / "p1_dev305_diverse_cem_20261102_CAPTURE_DEV01/report.json", CAPTURE_SHA)
    need(spec == actual.protocol() and report["contract_sha256"] == PREP_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"]
         and len(spec["frames"]) == len(report["rows"]) == len(physics["rows"]) == 867, "source")
    goal = np.asarray(capture["goal_state_evaluation_only"], dtype=float)
    results = []
    for role, model_row in zip(model_spec["roles"], model["rows"], strict=True):
        arm = role["arm"]
        ar = [r for r in report["rows"] if r["arm"] == arm]
        pr = [r for r in physics["rows"] if r["prefix"] == arm]
        fr = [r for r in spec["frames"] if r["arm"] == arm]
        need(len(ar) == len(pr) == len(fr) == 289 and model_row["arm"] == arm, "roster")
        for k, (a, p, f) in enumerate(zip(ar, pr, fr, strict=True)):
            ij = (k // 17, k % 17)
            need((a["first"], a["second"]) == (p["first"], p["second"]) ==
                 (f["first"], f["second"]) == ij and p["status"] == "fixed_horizon_completed", "index")
            run.coherent.base.frame(f["image"])
        pred = np.asarray(model_row["terminal_scores"], dtype=float).reshape(-1)
        real = np.asarray([r["actual_goal_mse"] for r in ar], dtype=float)
        rewards = np.asarray([r["transitions"][-1]["reward"] for r in pr], dtype=float)
        states = np.asarray([r["transitions"][-1]["state"] for r in pr], dtype=float)
        need(np.isfinite(pred).all() and np.isfinite(real).all() and np.isfinite(rewards).all(), "finite")
        picks = {"predicted_latent": int(np.argmin(pred)), "real_rgb_latent_oracle": int(np.argmin(real)),
                 "physical_reward_oracle": int(np.argmax(rewards))}
        selected = {}
        for name, index in picks.items():
            angle = abs(float((states[index, 4] - goal[4] + np.pi) % (2*np.pi) - np.pi))
            selected[name] = {"pair": [index // 17, index % 17], "physical_reward": float(rewards[index]),
                              "actual_goal_mse": float(real[index]), "predicted_goal_mse": float(pred[index]),
                              "block_position_error": float(np.linalg.norm(states[index, 2:4] - goal[2:4])),
                              "block_angle_error": angle}
        results.append({"arm": arm, "selected": selected,
                        "pair_counts_predicted_vs_actual_1e-6": pair_counts(pred, real, 1e-6),
                        "pair_counts_actual_vs_negative_reward_1e-6": pair_counts(real, -rewards, 1e-6)})
    return {"schema": "p1_dev305_two_packet_metric_decompose_20261102.v1", "rows": results,
            "scope": "known root, shared candidate bank, post-outcome real successor oracle; not deployable",
            "p1_admission": False, "independent_confirmation": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
