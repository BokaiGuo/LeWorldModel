"""Post-seal separation of action-effect error and encoded-goal mismatch."""
import json

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import ROOT, load_model

PREFIX = "pc_pointmaze_causal_residual_replication_20261106_codex"
OUT = ROOT / "results/pc_pointmaze_effect_decomposition_20261106_codex.json"


def decompose(predicted, actual, goal, distances):
    """Return three-action comparisons; pair indices are fixed, ties retained."""
    predicted = np.asarray(predicted, dtype=np.float64).reshape(3, -1)
    actual = np.asarray(actual, dtype=np.float64).reshape(3, -1)
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    distances = np.asarray(distances, dtype=np.float64)
    if predicted.shape != actual.shape or predicted.shape[1] != goal.size or distances.shape != (3,):
        raise ValueError("inconsistent action or latent dimensions")
    if not all(np.isfinite(x).all() for x in (predicted, actual, goal, distances)):
        raise ValueError("non-finite input")
    pred_cost = np.mean((predicted - goal) ** 2, axis=1)
    real_cost = np.mean((actual - goal) ** 2, axis=1)
    pairs = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        dp, dr = predicted[a] - predicted[b], actual[a] - actual[b]
        normp, normr = np.linalg.norm(dp), np.linalg.norm(dr)
        pairs.append({"actions": [a, b],
                      "predicted_vs_real_effect_cosine": float(np.dot(dp, dr) / (normp * normr))
                      if normp > 0 and normr > 0 else None,
                      "effect_relative_l2_error": float(np.linalg.norm(dp-dr) / normr) if normr > 0 else None,
                      "predicted_cost_difference": float(pred_cost[a]-pred_cost[b]),
                      "real_encoded_cost_difference": float(real_cost[a]-real_cost[b]),
                      "physical_distance_difference": float(distances[a]-distances[b])})
    choices = {"predicted": int(np.argmin(pred_cost)), "real_encoded": int(np.argmin(real_cost)),
               "physical": int(np.argmin(distances))}
    return {"pairs": pairs, "choices": choices, "predicted_costs": pred_cost.tolist(),
            "real_encoded_costs": real_cost.tolist(), "physical_distances": distances.tolist(),
            "physical_regret": {k: float(distances[v]-distances.min()) for k, v in choices.items()}}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = {k: ROOT / "results" / (PREFIX + "_" + suffix)
             for k, suffix in (("selection", "selection.json"), ("audit", "audit.json"),
                               ("physical", "physical.npz"), ("public", "inputs.npz"),
                               ("features", "features.npz"))}
    selection = json.loads(paths["selection"].read_text())
    audit = json.loads(paths["audit"].read_text())
    for key, path in (("public_sha256", paths["public"]), ("features_sha256", paths["features"])):
        if selection[key] != file_hash(path):
            raise ValueError("sealed selection dependency changed")
    if audit["selection_sha256"] != file_hash(paths["selection"]) or audit["physical_sha256"] != file_hash(paths["physical"]):
        raise ValueError("sealed audit dependency changed")
    model, _, permission = load_model()
    if model_fingerprint(model) != selection["model_fingerprint"] or permission.fingerprint != selection["input_audit_hash"]:
        raise ValueError("model contract changed")
    rows = []
    print("Post-seal only: separate action-dependent prediction and visual-goal cost mismatch.", flush=True)
    with np.load(paths["physical"], allow_pickle=False) as physical, np.load(paths["public"], allow_pickle=False) as public, np.load(paths["features"], allow_pickle=False) as features, torch.inference_mode():
        for row in audit["rows"]:
            root, camera = row["root"], row["camera"]
            i = root - 1892
            goal = model.encode(rgb_tensor(public["goal"][i, camera]))[0].cpu().numpy()
            actual = np.stack([model.encode(rgb_tensor(physical["images"][3*i+1, a, camera, -1]))[0].cpu().numpy()
                               for a in range(3)])
            raw = features[f"r{root}_c{camera}_raw"]
            corrected = features[f"r{root}_c{camera}_corrected"]
            result = decompose(raw, actual, goal, row["physical_distances"])
            correction = decompose(corrected, actual, goal, row["physical_distances"])
            for k, expected in (("predicted", row["choices"]["raw"]),):
                if result["choices"][k] != expected:
                    raise ValueError("reconstructed choice mismatch")
            if correction["choices"]["predicted"] != row["choices"]["corrected"]:
                raise ValueError("corrected choice mismatch")
            result.update(root=root, camera=camera, corrected_choice=correction["choices"]["predicted"],
                          corrected_regret=correction["physical_regret"]["predicted"])
            rows.append(result)
    if len(rows) != 16 or len({(r["root"], r["camera"]) for r in rows}) != 16:
        raise ValueError("incorrect panel")
    per_root = []
    for root in range(1892, 1900):
        pair = [r for r in rows if r["root"] == root]
        per_root.append({"root": root,
                         "physical_regret": {k: float(np.mean([r["physical_regret"][k] for r in pair]))
                                             for k in ("predicted", "real_encoded", "physical")},
                         "predicted_agrees_real_encoded": sum(r["choices"]["predicted"] == r["choices"]["real_encoded"] for r in pair),
                         "real_encoded_agrees_physical": sum(r["choices"]["real_encoded"] == r["choices"]["physical"] for r in pair)})
    pairs = [p for r in rows for p in r["pairs"]]
    output = {"rows": rows, "per_root": per_root,
              "mean_physical_regret": {k: float(np.mean([r["physical_regret"][k] for r in per_root]))
                                       for k in ("predicted", "real_encoded", "physical")},
              "predicted_real_encoded_choice_agreement": sum(r["predicted_agrees_real_encoded"] for r in per_root),
              "real_encoded_physical_choice_agreement": sum(r["real_encoded_agrees_physical"] for r in per_root),
              "pairwise_effect_cosine_mean": float(np.mean([p["predicted_vs_real_effect_cosine"] for p in pairs if p["predicted_vs_real_effect_cosine"] is not None])),
              "pairwise_effect_relative_l2_error_mean": float(np.mean([p["effect_relative_l2_error"] for p in pairs if p["effect_relative_l2_error"] is not None])),
              "input_hashes": {k: file_hash(v) for k, v in paths.items()}, "runner_sha256": file_hash(__file__),
              "scope": "post-hoc state-assisted development; real encoded outcome is an oracle diagnostic only"}
    with OUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ("mean_physical_regret", "predicted_real_encoded_choice_agreement",
                                              "real_encoded_physical_choice_agreement", "pairwise_effect_cosine_mean",
                                              "pairwise_effect_relative_l2_error_mean")}, indent=2))


if __name__ == "__main__":
    main()
