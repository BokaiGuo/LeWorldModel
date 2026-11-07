"""Audit five-packet alignment and re-encode a fixed subset of temporal frames."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments import pc_tworoom_block_b_temporal_alignment_20261107_codex as exp
from experiments.pc_input_permissions import file_hash

OUTPUT = Path(str(exp.PREFIX) + "_audit.json")
ROOTS = (0, 4, 9)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, prior = exp.base.source_records()
    physical, result = json.loads(exp.PHYSICAL.read_text()), json.loads(exp.RESULT.read_text())
    if (result["pred_sha256"] != file_hash(exp.PRED)
            or result["physical_sha256"] != file_hash(exp.PHYSICAL)
            or result["features_sha256"] != file_hash(exp.FEATURES)
            or physical["frames_sha256"] != file_hash(exp.FRAMES)
            or result["runner_sha256"] != file_hash(exp.__file__)):
        raise ValueError("artifact binding mismatch")
    model, _, transform = exp.base.setup(plan)
    max_error, checked = 0., 0
    with (np.load(exp.PRED, allow_pickle=False) as pred,
          np.load(exp.FEATURES, allow_pickle=False) as features,
          np.load(exp.FRAMES, allow_pickle=False) as frames,
          np.load(exp.base.FRAMES, allow_pickle=False) as final_frames,
          torch.inference_mode()):
        for root in range(10):
            images = frames[f"r{root}_frames"]
            actual = features[f"r{root}_actual"]
            predicted = pred[f"r{root}_predicted"]
            goal = features[f"r{root}_goal"]
            if images.shape[:2] != (64, 5) or predicted.shape[:2] != (64, 6):
                raise ValueError("time or candidate axis mismatch")
            np.testing.assert_array_equal(images[:, -1], final_frames[f"r{root}_frames"])
            recomputed, pred_cost, actual_cost = exp.temporal_summary(predicted[:, 1:], actual, goal,
                                                                        prior["rows"][root]["outcomes"])
            for key, value in recomputed.items():
                if result["rows"][root][key] != value:
                    raise ValueError(f"row mismatch root={root} key={key}")
            np.testing.assert_array_equal(pred_cost, features[f"r{root}_predicted_cost"])
            np.testing.assert_array_equal(actual_cost, features[f"r{root}_encoded_cost"])
            if root in ROOTS:
                flattened = images.reshape(320, *images.shape[2:])
                for offset in range(0, 320, 32):
                    batch = torch.stack([transform(x) for x in flattened[offset:offset+32]])[:, None].cuda()
                    encoded = model.encode({"pixels": batch})["emb"][:, 0].cpu().numpy()
                    expected = actual.reshape(320, -1)[offset:offset+32]
                    max_error = max(max_error, float(np.abs(encoded - expected).max()))
                    checked += len(encoded)
            print(f"root={root} temporal artifacts audited", flush=True)
    if max_error > 1e-4:
        raise ValueError(f"temporal re-encoding mismatch: {max_error}")
    with OUTPUT.open("x") as f:
        json.dump({"roots": 10, "reencoded_frames": checked, "max_feature_error": max_error,
                   "result_sha256": file_hash(exp.RESULT), "features_sha256": file_hash(exp.FEATURES),
                   "scope": "same model and simulator, not independent verification"}, f, indent=2)


if __name__ == "__main__":
    main()
