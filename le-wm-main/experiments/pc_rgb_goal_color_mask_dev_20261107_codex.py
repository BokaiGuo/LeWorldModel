"""RGB-only goal color masking on an exposed, fixed candidate bank."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_geometric_contact_equivalence_dev_20261106_codex as source
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_true_history_alignment_20261107_codex import model_and_scaler

PROTOCOL = source.ROOT.parent / "docs/PC_RGB_GOAL_COLOR_MASK_DEV_PROTOCOL_20261107_CODEX.md"
OUTPUT = source.ROOT / "results/pc_rgb_goal_color_mask_dev_20261107_codex.json"


def mask_goal(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("expected uint8 RGB image")
    rgb = image.astype(np.int16)
    r, g, b = np.moveaxis(rgb, -1, 0)
    mask = (b > 170) & (b - r > 45) & (b - g > 25) & (r < 150)
    if not mask.any():
        raise ValueError("no target agent pixels detected")
    modified = image.copy()
    modified[mask] = image[0, 0]
    ys, xs = np.nonzero(mask)
    return modified, mask, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = json.loads(source.MODEL.read_text())
    if prior["features_sha256"] != source.sha(source.FEATURES):
        raise ValueError("features changed")
    model, _, transform, action_hash = model_and_scaler()
    before = model_fingerprint(model)
    if before != prior["model_sha256"]:
        raise ValueError("model changed")
    rows = []
    with np.load(source.IMAGES, allow_pickle=False) as images, \
            np.load(source.FEATURES, allow_pickle=False) as features, torch.inference_mode():
        for root in range(16):
            goal = images[f"r{root}_white_goal"]
            modified, mask, bbox = mask_goal(goal)
            batch = torch.stack([transform(goal), transform(modified)]).cuda()
            embeddings = model.encode({"pixels": batch[:, None]})["emb"][:, 0].cpu().numpy()
            for side in (-1, 1):
                feats = np.stack([features[f"r{root}_white_{side}_{name}"]
                                  for name in ("zero", "plus", "minus")])
                np.testing.assert_allclose(feats[:, 3], np.broadcast_to(embeddings[0], feats[:, 3].shape),
                                           atol=1e-5, rtol=1e-5)
                scores = {}
                for variant, embedding in zip(("original", "masked"), embeddings, strict=True):
                    scores[variant] = {}
                    for method, index in (("predicted", 1), ("real_encoded", 2)):
                        costs = np.square(feats[:, index] - embedding).sum(axis=1)
                        scores[variant][method] = {"scores": costs.tolist(), "selected": int(costs.argmin())}
                rows.append({"root": root, "side": side, "mask_pixels": int(mask.sum()), "mask_bbox": bbox,
                             "goal_sha256": source.array_sha(goal), "masked_sha256": source.array_sha(modified),
                             "scores": scores})
    if model_fingerprint(model) != before:
        raise ValueError("model changed during run")
    # Evaluation labels are loaded only after every score and choice is fixed.
    physical = json.loads(source.PHYSICAL.read_text())
    if physical["images_sha256"] != source.sha(source.IMAGES):
        raise ValueError("physical source mismatch")
    for row in rows:
        condition = next(c for c in physical["rows"][row["root"]]["conditions"]
                         if c["side"] == row["side"] and c["appearance"] == "white")
        costs = np.asarray([condition["actions"][n]["object_pose_cost"] for n in ("zero", "plus", "minus")])
        row["physical_costs"] = costs.tolist()
        for variant in ("original", "masked"):
            for method in ("predicted", "real_encoded"):
                entry = row["scores"][variant][method]
                entry["regret"] = float(costs[entry["selected"]] - costs.min())
    summary = {}
    for method in ("predicted", "real_encoded"):
        deltas = [np.mean([r["scores"]["masked"][method]["regret"] -
                           r["scores"]["original"][method]["regret"]
                           for r in rows if r["root"] == root]) for root in range(16)]
        summary[method] = {
            "selection_changed": sum(r["scores"]["masked"][method]["selected"] !=
                                     r["scores"]["original"][method]["selected"] for r in rows),
            "original_mean_regret": float(np.mean([r["scores"]["original"][method]["regret"] for r in rows])),
            "masked_mean_regret": float(np.mean([r["scores"]["masked"][method]["regret"] for r in rows])),
            "roots_improved": int(sum(d < -1e-9 for d in deltas)),
            "roots_worsened": int(sum(d > 1e-9 for d in deltas)),
            "roots_unchanged": int(sum(abs(d) <= 1e-9 for d in deltas)),
        }
    record = {"protocol_sha256": source.sha(PROTOCOL), "runner_sha256": source.sha(Path(__file__)),
              "source_images_sha256": source.sha(source.IMAGES), "features_sha256": source.sha(source.FEATURES),
              "physical_sha256": source.sha(source.PHYSICAL), "model_sha256": before,
              "action_array_sha256": action_hash, "rows": rows, "summary": summary,
              "scope": "exposed development candidates; not closed-loop or independent gate evidence"}
    payload = json.dumps(record, indent=2, allow_nan=False)
    with OUTPUT.open("x") as stream:
        stream.write(payload)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
