"""Analyze already-encoded real frames with identical object pose and varying pusher."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from . import p1_backtrack_rgb_feedback_cuda_20261030 as feedback

ROOT = feedback.ROOT
ENCODED = ROOT / "results/p1_backtrack_rgb_feedback_cuda_20261030_MAIN01_report.json"
ENCODED_SHA = "57722787fb1a008440fdc0a0acbcca733913db9a231aa4d0cbc6f7a4012cc21e"
need = feedback.require


def analyze(encoded, physical):
    need(len(encoded["rows"]) == len(physical["rows"]) == 16, "fixed16")
    groups = {}
    for index, (encoded_row, physical_row) in enumerate(zip(encoded["rows"], physical["rows"], strict=True)):
        need(encoded_row["index"] == physical_row["index"] == index and
             all(encoded_row[k] == physical_row["plan"][k] for k in ("slot", "arm", "role"))
             and len(encoded_row["actual_latents"]) == len(physical_row["checkpoints"]) == 4,
             "row identity")
        for point, latent in zip(physical_row["checkpoints"], encoded_row["actual_latents"], strict=True):
            state = point["state"]
            need(len(state) == 7 and len(latent) == 192 and len(encoded_row["goal_latent"]) == 192,
                 "state/latent dimensions")
            group = (encoded_row["slot"], tuple(state[2:5]))
            record = dict(index=index, step=point["step"], pixel_sha=point["pixel_sha256"],
                          pusher=state[:2], goal=physical_row["goal"][:2],
                          latent=latent, goal_latent=encoded_row["goal_latent"])
            # Deduplicate exact repeated frames before forming pairs.
            key = (tuple(state[:2]), point["pixel_sha256"])
            groups.setdefault(group, {}).setdefault(key, record)
    pairs = []
    for (slot, object_pose), records in groups.items():
        for left, right in itertools.combinations(records.values(), 2):
            if left["pusher"] == right["pusher"]:
                continue
            need(left["goal"] == right["goal"] and left["goal_latent"] == right["goal_latent"], "goal identity")
            def euclidean(x, y):
                return math.sqrt(sum((float(a)-float(b)) ** 2 for a, b in zip(x, y, strict=True)))
            ldist = euclidean(left["latent"], left["goal_latent"])
            rdist = euclidean(right["latent"], right["goal_latent"])
            lp = euclidean(left["pusher"], left["goal"])
            rp = euclidean(right["pusher"], right["goal"])
            pairs.append(dict(slot=slot, object_pose=list(object_pose),
                              left={k: left[k] for k in ("index", "step", "pixel_sha", "pusher")},
                              right={k: right[k] for k in ("index", "step", "pixel_sha", "pusher")},
                              pusher_distance=[lp, rp], goal_latent_distance=[ldist, rdist],
                              pair_latent_distance=euclidean(left["latent"], right["latent"])))
    return dict(schema="p1_pusher_latent_saved_analysis.v1", pairs=pairs, groups=len(groups),
                new_neural_calls=0, P1_admission=False, independent_confirmation=False,
                limitations="known roots; exact object pose and same slot; repeated-pair dependence; no causal guarantee")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    need(out.is_absolute() and out.parent == ROOT / "results"
         and out.name.startswith("codex_p1_pusher_latent_saved_analysis_20261030_")
         and not out.exists() and not out.is_symlink(), "exclusive output")
    need(feedback.physical.sha(ENCODED) == ENCODED_SHA, "encoded source hash")
    source = feedback.inputs()
    report = feedback.physical.native.read(ENCODED, cap=64 * 1024**2)
    prepared_path = Path(report["prepared"])
    need(prepared_path.is_file() and not prepared_path.is_symlink()
         and feedback.physical.sha(prepared_path) == report["prepare_sha256"]
         and feedback.physical.native.read(prepared_path, cap=64 * 1024**2) == source
         and len(source["rows"]) == 16 and report["model_before"] == report["model_after"]
         and report["model_before"] == source["model_hash"], "frozen source model and prepared inputs")
    physical = feedback.physical.native.read(feedback.REPORT, cap=64 * 1024**2)
    result = analyze(report, physical)
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with out.open("xb") as stream:
        need(stream.write(data) == len(data), "full write")
    print(json.dumps(dict(output=str(out), sha256=hashlib.sha256(data).hexdigest(), pairs=len(result["pairs"]))))


if __name__ == "__main__":
    main()
