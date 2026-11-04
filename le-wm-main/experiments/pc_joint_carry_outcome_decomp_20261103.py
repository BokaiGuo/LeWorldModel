"""Compare model scores, same-action actual RGB scores, and physical outcomes."""
import argparse
import itertools
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from experiments import pc_causal_history_h10_suffix_20261103 as h

SOURCE = h.ROOT / "results/pc_joint_carry_all_roots_20261103_DEV01.json"
SOURCE_SHA = "f5977bd71e636a316935ecfce1045968adbcf6c70b100de5c825bfc20f2fe849"
PROTOCOL = h.ROOT.parent / "docs/PC_JOINT_CARRY_SAME_ACTION_OUTCOME_DECOMP_PROTOCOL_20261103.md"
OUTPUT = h.ROOT / "results/pc_joint_carry_outcome_decomp_20261103_DEV01.json"


def pairs(rows):
    results = []
    for i in range(0, len(rows), 3):
        group = rows[i:i+3]
        if len(group) != 3 or [r["mode"] for r in group] != ["old", "zero_init", "carry_init"]:
            raise ValueError("unexpected pair roster")
        if len({(r["root"], r["role"]) for r in group}) != 1:
            raise ValueError("cross-role comparison")
        for a, b in itertools.combinations(group, 2):
            entry = {"root": a["root"], "role": a["role"], "modes": [a["mode"], b["mode"]]}
            for first, second, label in (("predicted", "actual_rgb", "model_vs_rgb"),
                                         ("predicted", "physical", "model_vs_physical"),
                                         ("actual_rgb", "physical", "rgb_vs_physical")):
                x, y = a[first] - b[first], a[second] - b[second]
                entry[label] = "tie" if x == 0 or y == 0 else ("reversed" if x*y < 0 else "agree")
            results.append(entry)
    return results


def main():
    if OUTPUT.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (h.OBSTACLE, h.OBSTACLE_SHA),
                           (h.CAMERA, h.CAMERA_SHA), (h.OLD, h.OLD_SHA),
                           (h.DEFAULT_DATASET / "seq_lengths.pth", h.LENGTHS_SHA)):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    source, obstacle, camera, old = (json.loads(p.read_text()) for p in (SOURCE, h.OBSTACLE, h.CAMERA, h.OLD))
    for path, expected in ((SOURCE.with_suffix(".npz"), source["arrays_sha256"]),
                           (h.OBSTACLE.with_suffix(".npz"), obstacle["arrays_sha256"]),
                           (h.DEFAULT_CHECKPOINT, source["checkpoint_sha256"])):
        if h.sha256_file(path) != expected:
            raise ValueError(f"source changed: {path}")
    if len(source["rows"]) != 51:
        raise ValueError("invalid source count")
    args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                              config=h.DEFAULT_CONFIG, dinov2_source=h.ROOT / "external/local_dinov2")
    rows, started = [], monotonic()
    print("Encode saved cp17 RGB for fixed old/zero/carry actions; no new selection.", flush=True)
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as data, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle_arrays, \
            h.lock.gpu_lock(h.ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        model, dataset = h.load_model_and_dataset(args, device)
        before = h.parameter_hash(model)
        if before != source["model_before"] or before != source["model_after"]:
            raise ValueError("frozen model mismatch")
        if not (np.allclose(dataset.proprio_mean.cpu().numpy(), h.MEAN, rtol=0, atol=1e-7)
                and np.allclose(dataset.proprio_std.cpu().numpy(), h.STD, rtol=0, atol=1e-7)):
            raise ValueError("proprio normalizer mismatch")
        weight, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weight), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        metadata, offset = camera["metadata"], old["body_offset_from_static_scene"]
        goal_rgb = obstacle_arrays["goal_rgb"]
        goal_state, _ = h.estimate_state(goal_rgb, metadata, offset)
        target = h.encode_with_state(model, goal_rgb, goal_state, device)["proprio"].reshape(256, 16)
        for row in source["rows"]:
            name = f"r{row['root']}_{row['role']}_{row['mode']}"
            frames, actions = data[name+"_history"], data[name+"_actions"]
            if frames.shape != (18, 224, 224, 3) or actions.shape != (17, 2) or np.any(np.abs(actions) > 1):
                raise ValueError("trajectory input contract")
            state = h.causal_state(frames[16], frames[17], metadata, offset, .05)
            with torch.no_grad():
                actual = h.encode_with_state(model, frames[17], state, device)["proprio"]
                cost = float(h.projected_cost(actual.reshape(1, 256, 16), target, pinv, pos_std).item())
            prediction = row["refined_score"] if row["refined"] else row["prior_score"]
            if not np.isfinite([cost, prediction, row["cp17_distance"]]).all():
                raise ValueError("nonfinite score")
            rows.append({"root": row["root"], "role": row["role"], "mode": row["mode"],
                         "predicted": prediction, "actual_rgb": cost, "physical": row["cp17_distance"],
                         "causal_state": state.tolist()})
        torch.cuda.synchronize()
        after = h.parameter_hash(model)
        if before != after:
            raise ValueError("model changed")
    pair_rows = pairs(rows)
    summary = {}
    for label in ("model_vs_rgb", "model_vs_physical", "rgb_vs_physical"):
        summary[label] = {value: sum(p[label] == value for p in pair_rows) for value in ("agree", "reversed", "tie")}
    report = {"schema": "pc_joint_carry_outcome_decomp.v1", "rows": rows, "pairs": pair_rows,
              "summary": summary, "source_sha256": SOURCE_SHA, "source_arrays_sha256": source["arrays_sha256"],
              "protocol_sha256": h.sha256_file(PROTOCOL), "script_sha256": h.sha256_file(Path(__file__)),
              "checkpoint_sha256": h.sha256_file(h.DEFAULT_CHECKPOINT), "model_before": before, "model_after": after,
              "counts": {"goal_encodes": 1, "actual_rgb_encodes": 51, "fresh_prediction_samples": 0,
                         "physical_commands": 0}, "wall_seconds": monotonic()-started,
              "scope": "post-outcome dependent development, not a planner or independent confirmation"}
    with OUTPUT.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
