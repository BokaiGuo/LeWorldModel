"""Audit latent/cost consistency and independently re-encode fixed frame subsets."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_block_b_reencode_alignment_20261107_codex as exp
from experiments.pc_input_permissions import file_hash

OUTPUT = Path(str(exp.PREFIX) + "_audit.json")
ROOTS = (0, 4, 9)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, prior_physical = exp.source_records()
    result = json.loads(exp.RESULT.read_text())
    physical = json.loads(exp.PHYSICAL.read_text())
    if (result["prediction_sha256"] != file_hash(exp.PRED)
            or result["features_sha256"] != file_hash(exp.FEATURES)
            or result["physical_sha256"] != file_hash(exp.PHYSICAL)
            or physical["frames_sha256"] != file_hash(exp.FRAMES)
            or result["runner_sha256"] != file_hash(exp.__file__)):
        raise ValueError("artifact binding mismatch")
    model, _, transform = exp.setup(plan)
    max_cost_error, max_feature_error = 0., 0.
    with (np.load(exp.PRED, allow_pickle=False) as pred,
          np.load(exp.FEATURES, allow_pickle=False) as features,
          np.load(exp.FRAMES, allow_pickle=False) as frames,
          np.load(exp.prior.source.block.prior.inputs.INPUT, allow_pickle=False) as inputs,
          torch.inference_mode()):
        for root in range(10):
            predicted = pred[f"r{root}_predicted"]
            actual = features[f"r{root}_actual"]
            goal = features[f"r{root}_goal"]
            costs = pred[f"r{root}_cost"]
            predicted_cost = np.square(predicted - goal).sum(axis=-1)
            actual_cost = np.square(actual - goal).sum(axis=-1)
            max_cost_error = max(max_cost_error, float(np.abs(predicted_cost - costs).max()))
            np.testing.assert_allclose(actual_cost, features[f"r{root}_encoded_cost"], atol=1e-5, rtol=0)
            row, _ = exp.summarize_root(predicted, actual, goal, costs, prior_physical["rows"][root]["outcomes"])
            for key, value in row.items():
                if result["rows"][root][key] != value:
                    raise ValueError(f"summary mismatch root={root} field={key}")
            if root in ROOTS:
                goal_check = model.encode({"pixels": transform(inputs["goal"][root])[None, None].cuda()})["emb"][0, 0].cpu().numpy()
                max_feature_error = max(max_feature_error, float(np.abs(goal_check - goal).max()))
                images = frames[f"r{root}_frames"]
                for offset in (0, 32):
                    batch = torch.stack([transform(x) for x in images[offset:offset+32]])[:, None].cuda()
                    replay = model.encode({"pixels": batch})["emb"][:, 0].cpu().numpy()
                    max_feature_error = max(max_feature_error, float(np.abs(replay - actual[offset:offset+32]).max()))
            print(f"root={root} verified", flush=True)
    if max_cost_error > 0.001 or max_feature_error > 0.0001:
        raise ValueError(f"replay mismatch cost={max_cost_error} feature={max_feature_error}")
    with OUTPUT.open("x") as f:
        json.dump({"roots": 10, "rescored_real_images": len(ROOTS) * 64,
                   "max_cost_error": max_cost_error, "max_feature_error": max_feature_error,
                   "result_sha256": file_hash(exp.RESULT), "features_sha256": file_hash(exp.FEATURES),
                   "scope": "same model/simulator replay, not independent verification"},
                  f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
