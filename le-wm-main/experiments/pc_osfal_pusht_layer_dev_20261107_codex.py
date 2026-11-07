"""RGB PushT frozen layer audit on previously collected contact-ready trajectories."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eval import img_transform
from experiments.checkpoint_compat import _resolve_local, load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "results/pc_geometric_contact_equivalence_dev_20261106_codex"
IMAGES = PREFIX.with_name(PREFIX.name + "_images.npz")
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")
MODEL = PREFIX.with_name(PREFIX.name + "_model.json")
PROTOCOL = ROOT.parent / "docs/PC_OSFAL_PUSHT_LAYER_DEV_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_osfal_pusht_layer_dev_20261107_codex.json"
ACTIONS = ("zero", "plus", "minus")
COLORS = ("white", "gray")
SIDES = (-1, 1)


def layer_metrics(features, physical):
    """Features have axes color, side, [initial, three final, goal], channel."""
    features = np.asarray(features, dtype=np.float64)
    physical = np.asarray(physical, dtype=np.float64)
    if features.ndim != 4 or features.shape[:3] != (2, 2, 5) or physical.shape != (2, 2, 3):
        raise ValueError("invalid feature or physical layout")
    background = float(np.mean((features[0, :, :4] - features[1, :, :4]) ** 2))
    side_gap = float(np.mean((features[:, 0, 0] - features[:, 1, 0]) ** 2))
    action_gap = float(np.mean([np.mean((features[c, s, 1+i] - features[c, s, 1+j]) ** 2)
                                for c in range(2) for s in range(2)
                                for i, j in ((0, 1), (0, 2), (1, 2))]))
    conditions = []
    for c, color in enumerate(COLORS):
        for s, side in enumerate(SIDES):
            scores = ((features[c, s, 1:4] - features[c, s, 4]) ** 2).sum(axis=-1)
            chosen = int(np.argmin(scores))
            regret = float(physical[c, s, chosen] - physical[c, s].min())
            conditions.append({"appearance": color, "side": side, "scores": scores.tolist(),
                               "physical_costs": physical[c, s].tolist(), "selected": ACTIONS[chosen],
                               "regret": regret, "tie_optimal_1e-9": regret <= 1e-9})
    return {"background_mse": background, "side_initial_mse": side_gap,
            "action_final_mse": action_gap, "conditions": conditions}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(PHYSICAL.read_text())
    prior = json.loads(MODEL.read_text())
    if source["images_sha256"] != file_hash(IMAGES):
        raise ValueError("physical image hash mismatch")
    if len(source["rows"]) != 16 or not all(row["valid"] for row in source["rows"]):
        raise ValueError("unexpected invalid physical panel")
    weights, config = _resolve_local("pusht/lewm")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    captured = {}
    def hook(index):
        def capture(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise ValueError("unexpected ViT layer output")
            captured[index] = value[:, 0].detach()
        return capture
    if len(model.encoder.layers) != 12:
        raise ValueError("unexpected layer count")
    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(model.encoder.layers)]
    records = []
    max_cost_error = 0.0
    print("PushT RGB: frozen CLS layer audit of all 16 previously exposed roots.", flush=True)
    try:
        with np.load(IMAGES, allow_pickle=False) as images, torch.inference_mode():
            for row in source["rows"]:
                root = row["root"]
                pixels, costs = [], np.zeros((2, 2, 3))
                for ci, color in enumerate(COLORS):
                    for si, side in enumerate(SIDES):
                        cond = next(c for c in row["conditions"] if c["appearance"] == color and c["side"] == side)
                        key = f"r{root}_{color}_{side}"
                        keys = [f"{key}_zero_initial"] + [f"{key}_{a}_final" for a in ACTIONS] + [f"r{root}_{color}_goal"]
                        pixels.extend(transform(images[k]) for k in keys)
                        costs[ci, si] = [cond["actions"][a]["object_pose_cost"] for a in ACTIONS]
                batch = torch.stack(pixels).cuda()[:, None]
                official = model.encode({"pixels": batch})["emb"][:, 0]
                layer_features = {f"block_{i+1:02d}": captured[i].cpu().numpy().reshape(2, 2, 5, -1)
                                  for i in range(12)}
                layer_features["official_visual"] = official.cpu().numpy().reshape(2, 2, 5, -1)
                metrics = {name: layer_metrics(value, costs) for name, value in layer_features.items()}
                modelrow = prior["rows"][root]
                if modelrow["root"] != root:
                    raise ValueError("prior root order mismatch")
                for condition in metrics["official_visual"]["conditions"]:
                    old = next(c for c in modelrow["conditions"] if c["appearance"] == condition["appearance"]
                               and c["side"] == condition["side"])
                    saved = np.array([old["actions"][a]["real_encoded_cost"] for a in ACTIONS])
                    actual = np.array(condition["scores"])
                    np.testing.assert_allclose(actual, saved, atol=1e-3, rtol=1e-4)
                    max_cost_error = max(max_cost_error, float(np.max(np.abs(actual - saved))))
                    condition["prior_predicted_costs"] = [old["actions"][a]["predicted_cost"] for a in ACTIONS]
                records.append({"root": root, "layers": metrics})
                captured.clear()
                print(f"root {root} complete", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    summary = {}
    for name in records[0]["layers"]:
        layers = [r["layers"][name] for r in records]
        summary[name] = {"mean_background_mse": float(np.mean([r["background_mse"] for r in layers])),
                         "mean_side_initial_mse": float(np.mean([r["side_initial_mse"] for r in layers])),
                         "mean_action_final_mse": float(np.mean([r["action_final_mse"] for r in layers])),
                         "mean_physical_regret": float(np.mean([c["regret"] for r in layers for c in r["conditions"]])),
                         "tie_optimal_conditions": sum(c["tie_optimal_1e-9"] for r in layers for c in r["conditions"])}
    report = {"scope": "primary_rgb frozen observation diagnostic; no predictor intervention or gate promotion",
              "physical_sha256": file_hash(PHYSICAL), "images_sha256": file_hash(IMAGES),
              "prior_model_sha256": file_hash(MODEL), "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "checkpoint_sha256": file_hash(weights),
              "config_sha256": file_hash(config), "model_fingerprint": fingerprint,
              "max_official_cost_error": max_cost_error, "summary": summary, "rows": records}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(OUTPUT), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
