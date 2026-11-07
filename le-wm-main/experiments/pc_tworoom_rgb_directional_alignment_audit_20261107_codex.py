"""Post-seal same-action directional alignment summary and integrity check."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_rgb_directional_alignment_20261107_codex as run

OUT = run.source.ROOT / "results/pc_tworoom_rgb_directional_alignment_audit_20261107_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan, physical = json.loads(run.PLAN.read_text()), json.loads(run.PHYSICAL.read_text())
    if (physical["plan_sha256"] != run.file_hash(run.PLAN)
            or plan["protocol_sha256"] != run.file_hash(run.PROTOCOL)
            or plan["runner_sha256"] != run.file_hash(run.__file__)):
        raise ValueError("artifact drift")
    results = []
    for model, truth in zip(plan["rows"], physical["rows"], strict=True):
        if model["row"] != truth["row"] or model["root"] != truth["root"]:
            raise ValueError("task mismatch")
        if len(truth["indices"]) != 16 or len(truth["traces"]) != 16:
            raise ValueError("physical coverage")
        lookup = {index: pos for pos, index in enumerate(truth["indices"])}
        pred = np.asarray(model["predicted_features"])
        obs = np.asarray(truth["true_features"])
        target = np.asarray(truth["target_feature"])
        costs = np.asarray(truth["encoded_costs"])
        np.testing.assert_allclose(np.square(obs - target).sum(-1), costs, rtol=0, atol=2e-3)
        model_costs = np.asarray(model["predicted_costs"])
        pair_data = []
        for d in range(4):
            for h_idx in (1, 2):
                plus, minus = d * 6 + 2 * h_idx, d * 6 + 2 * h_idx + 1
                p, m = lookup[plus], lookup[minus]
                if len(truth["traces"][p]) != 25 or len(truth["traces"][m]) != 25:
                    raise ValueError("physical time mismatch")
                pred_delta, real_delta = pred[plus] - pred[minus], obs[p] - obs[m]
                pnorm, rnorm = np.linalg.norm(pred_delta), np.linalg.norm(real_delta)
                cosine = (float(pred_delta @ real_delta / (pnorm * rnorm))
                          if pnorm > 1e-8 and rnorm > 1e-8 else None)
                pair_data.append({"direction": d, "scale": run.SCALES[h_idx],
                                  "predicted_cost_diff": float(model_costs[d, h_idx, 0] - model_costs[d, h_idx, 1]),
                                  "real_encoded_cost_diff": float(costs[p] - costs[m]),
                                  "physical_distance_diff": float(truth["traces"][p][-1]["distance"] -
                                                                  truth["traces"][m][-1]["distance"]),
                                  "predicted_real_effect_cosine": cosine,
                                  "success_plus": any(t["terminated"] for t in truth["traces"][p]),
                                  "success_minus": any(t["terminated"] for t in truth["traces"][m])})
        ad, fd = np.asarray(model["ad"]), np.asarray(model["fd"])
        diff = np.abs(fd[:, 0] - ad)
        flagged = (diff > .1) | ((np.abs(ad) >= 1e-4) & (diff / np.maximum(np.abs(ad), 1e-30) > .1))
        results.append({"root": model["root"], "row": model["row"], "ad": ad.tolist(), "fd": fd.tolist(),
                        "small_scale_absolute_error": diff.tolist(), "small_scale_flagged": flagged.tolist(),
                        "pairs": pair_data})
    with OUT.open("x") as f:
        json.dump({"rows": results, "plan_sha256": run.file_hash(run.PLAN),
                   "physical_sha256": run.file_hash(run.PHYSICAL), "audit_sha256": run.file_hash(Path(__file__)),
                   "scope": "exposed roots; arithmetic and bindings, not independent replay"}, f, indent=2, allow_nan=False)
    small = [err for r in results for err in r["small_scale_absolute_error"]]
    print(json.dumps({"roots": len(results), "flagged_directions": sum(sum(r["small_scale_flagged"]) for r in results),
                      "small_scale_max_absolute_error": max(small)}, indent=2))


if __name__ == "__main__":
    main()
