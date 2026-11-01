"""Full cross-goal roster with failed slots and actual RGB/physical separation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_cross_goal_carry_20261101_main as cross
from .p1_common_h10_analysis_20261101_main import metrics, delta

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("structured", "native", "random", "immediate_progress")


def source_directory(root, arm):
    if root == 0 and arm in ("structured", "native"):
        return ROOT / "results/p1_cross_goal_carry_20261101_main_root0_RUN_MAIN01"
    return ROOT / f"results/p1_cross_goal_shard_20261101_main_root{root}_{arm}_RUN_MAIN01"


def analyze():
    sources, rows = {}, []

    def read(path):
        blob = path.read_bytes()
        sources[str(path)] = hashlib.sha256(blob).hexdigest()
        return json.loads(blob)

    target_path = ROOT / "results/e3_cp10_target_cuda_20261031_RUN01.json"
    target = read(target_path)
    if sources[str(target_path)] != "a0a524ef256dc8737995ce2f0daac5c15ad7578e2234be42c19c19cfbc864aef":
        raise ValueError("fixed target source")
    for root in (0, 1):
        for arm in ARMS:
            directory = source_directory(root, arm)
            report = read(directory / "report.json")
            row = next(r for r in report["rows"] if r["arm"] == arm)
            if row["status"] == "not_started":
                raise ValueError("unattempted panel slot")
            cp = {p["step"]: p for p in row["checkpoints"]}
            goal_record = target["rows"][1-root]
            goal = np.asarray(goal_record["latent"], np.float32)
            points = []
            for step, point in sorted(cp.items()):
                if point["goal"] != goal_record["goal_state"]:
                    raise ValueError("physical target mismatch")
                image = Path(point["path"])
                if hashlib.sha256(image.read_bytes()).hexdigest() != point["file_sha256"]:
                    raise ValueError("physical PNG bytes")
                sources[str(image)] = point["file_sha256"]
                point_metrics = metrics(point["state"], point["goal"])
                encoded = directory / f"{row['episode_id']}_b{step:02}_encoding.json"
                score = None
                if encoded.exists():
                    encoding = read(encoded)
                    original = encoding["mapping"]["original_png"]
                    if original["pixel_sha256"] != point["pixel_sha256"] or encoding["actual_step"] != step:
                        raise ValueError("actual RGB/time binding")
                    latent = encoding["response"]["latent"]
                    path = Path(latent["path"])
                    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                    if sources[str(path)] != latent["file_sha256"]:
                        raise ValueError("latent file")
                    value = np.load(path, allow_pickle=False)
                    if value.shape != (192,) or value.dtype != np.float32 or not np.isfinite(value).all():
                        raise ValueError("saved latent")
                    score = float(np.square(value - goal).mean())
                points.append(dict(step=step, metrics=point_metrics, actual_RGB_latent_goal_MSE=score,
                                   object_state=point["state"][2:5]))
            immediate = (delta(metrics(cp[5]["state"], cp[0]["goal"]),
                               metrics(cp[0]["state"], cp[0]["goal"])) if 5 in cp else None)
            rows.append(dict(root=root, target=1-root, arm=arm, status=row["status"],
                             success=row["native_success"], steps=row["control_returned"],
                             error=row["first_error"], immediate_delta=immediate, points=points,
                             report_sha256=sources[str(directory / "report.json")]))
    return dict(rows=rows, sources=sources, attempted_denominator=8, P1_admission=False,
                independent_confirmation=False, task_reachability_within20="unknown",
                terminal_embeddings_not_imputed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(rows=len(result["rows"]), output=str(args.output))))
