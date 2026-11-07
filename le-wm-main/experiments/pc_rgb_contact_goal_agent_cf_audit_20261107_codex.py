"""Independent recomputation of counterfactual target scores and physical regret."""
import json

import numpy as np
import torch

from experiments import pc_geometric_contact_equivalence_dev_20261106_codex as source
from experiments import pc_rgb_contact_goal_agent_cf_20261107_codex as run
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_true_history_alignment_20261107_codex import model_and_scaler

OUTPUT = source.ROOT / "results/pc_rgb_contact_goal_agent_cf_audit_20261107_codex.json"


def summarize(rows, method):
    return {"conditions": len(rows),
            "selection_changed": sum(r["results"]["original"][method]["selected"]
                                     != r["results"]["alternative"][method]["selected"] for r in rows),
            "mean_regret_original": float(np.mean([r["results"]["original"][method]["physical_regret"] for r in rows])),
            "mean_regret_alternative": float(np.mean([r["results"]["alternative"][method]["physical_regret"]
                                                       for r in rows]))}


def count_improved(values):
    return int(sum(x < -1e-9 for x in values))


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(run.RESULT.read_text())
    render = json.loads(run.RENDER.read_text())
    physical = json.loads(source.PHYSICAL.read_text())
    if (result["render_sha256"] != source.sha(run.RENDER) or result["images_sha256"] != source.sha(run.IMAGES)
            or result["features_sha256"] != source.sha(source.FEATURES)
            or result["runner_sha256"] != source.sha(run.__file__)
            or render["physical_sha256"] != source.sha(source.PHYSICAL)
            or render["protocol_sha256"] != source.sha(run.PROTOCOL)):
        raise ValueError("artifact drift")
    model, _, transform, _ = model_and_scaler()
    if model_fingerprint(model) != result["model_sha256"]:
        raise ValueError("model drift")
    with np.load(run.IMAGES, allow_pickle=False) as images, np.load(source.IMAGES, allow_pickle=False) as original, \
            np.load(source.FEATURES, allow_pickle=False) as features, torch.inference_mode():
        for row, rendered in zip(result["rows"], render["rows"], strict=True):
            root, side = row["root"], row["side"]
            if (root, side) != (rendered["root"], rendered["side"]):
                raise ValueError("identity mismatch")
            goal = np.asarray(rendered["goal"])
            alternate = np.asarray(rendered["alternate"])
            np.testing.assert_array_equal(goal[2:], alternate[2:])
            np.testing.assert_array_equal(images[f"r{root}_s{side}_original"], original[f"r{root}_white_goal"])
            condition = next(c for c in physical["rows"][root]["conditions"]
                             if c["side"] == side and c["appearance"] == "white")
            np.testing.assert_allclose(alternate[:2], condition["requested_state"][:2], rtol=0, atol=1e-12)
            physical_cost = np.asarray([condition["actions"][n]["object_pose_cost"]
                                        for n in ("zero", "plus", "minus")])
            np.testing.assert_array_equal(row["physical_costs"], physical_cost)
            feats = np.stack([features[f"r{root}_white_{side}_{name}"] for name in ("zero", "plus", "minus")])
            for variant in ("original", "alternative"):
                image = transform(images[f"r{root}_s{side}_{variant}"]).cuda()
                embed = model.encode({"pixels": image[None, None]})["emb"][0, 0].cpu().numpy()
                for method, index in (("predicted", 1), ("real_encoded", 2)):
                    expected = np.square(feats[:, index] - embed).sum(1)
                    recorded = row["results"][variant][method]
                    np.testing.assert_allclose(expected, recorded["scores"], atol=1e-4, rtol=1e-5)
                    chosen = int(np.argmin(expected))
                    if chosen != recorded["selected"]:
                        raise ValueError("selection mismatch")
                    np.testing.assert_allclose(physical_cost[chosen] - physical_cost.min(),
                                               recorded["physical_regret"], atol=1e-12, rtol=0)
    summary = {}
    for name, selected in (("all", result["rows"]), ("nonoverlap", [r for r in result["rows"] if not r["overlap"]]),
                           ("overlap", [r for r in result["rows"] if r["overlap"]])):
        summary[name] = {m: summarize(selected, m) for m in ("predicted", "real_encoded")}
    summary["root_improved"] = {}
    for method in ("predicted", "real_encoded"):
        by_root = [np.mean([r["results"]["alternative"][method]["physical_regret"]
                            - r["results"]["original"][method]["physical_regret"] for r in result["rows"]
                            if r["root"] == root]) for root in range(16)]
        summary["root_improved"][method] = count_improved(by_root)
    payload = json.dumps({"summary": summary, "result_sha256": source.sha(run.RESULT),
                          "audit_sha256": source.sha(__file__),
                          "scope": "counterfactual target post-hoc; not deployment or gate evidence"},
                         indent=2, allow_nan=False)
    with OUTPUT.open("x") as stream:
        stream.write(payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
