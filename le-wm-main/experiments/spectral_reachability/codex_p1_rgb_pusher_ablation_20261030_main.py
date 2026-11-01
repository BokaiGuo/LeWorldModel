"""Fixed known-root RGB pusher correction ablation with existing fail-closed runner."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import codex_p1_rgb_feedback_launcher_20261030_v1 as launch
from . import codex_p1_rgb_feedback_saved_20261030_v1 as saved

NAME = "codex_p1_rgb_pusher_ablation_20261030_main"
ORIGINAL_BUILD = launch.build
POLICY = {**launch.rgb_candidate.POLICY, "schema": NAME,
          "object_shape_chamfer_max_pixels": None,
          "meaning": "exploratory unconditional pusher correction on valid RGB; not a safety policy"}


def decide(current_rgb, goal_rgb, model_first_packet):
    if (not isinstance(model_first_packet, np.ndarray) or model_first_packet.dtype != np.dtype("<f4")
            or model_first_packet.shape != (5, 2) or not np.isfinite(model_first_packet).all()
            or not (np.abs(model_first_packet) <= 1).all()):
        raise ValueError("model packet ABI")
    try:
        features = launch.rgb_candidate.rgb.score(current_rgb, goal_rgb)
    except ValueError as exc:
        return {"decision": "rgb_rejected", "reason": str(exc), "packet": None, "features": None}
    packet = model_first_packet.copy()
    decision = "model_packet"
    if features["pusher_centroid_distance_pixels"] > POLICY["pusher_distance_min_pixels"]:
        delta = (np.asarray(features["goal_pusher"]["centroid_xy"], np.float64)
                 - np.asarray(features["current_pusher"]["centroid_xy"], np.float64))
        norm = float(np.linalg.norm(delta))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("pusher direction")
        packet = np.zeros((5, 2), dtype="<f4")
        packet[0] = (delta / norm * POLICY["correction_raw_magnitude"]).astype("<f4")
        decision = "rgb_pusher_correction"
    return {"decision": decision, "reason": None, "packet": packet,
            "packet_sha256": hashlib.sha256(packet.tobytes()).hexdigest(), "features": features}


def configure():
    launch.NAME = NAME
    launch.rgb_candidate.POLICY = POLICY
    launch.rgb_candidate.decide = decide

    def build():
        result = ORIGINAL_BUILD()
        path = Path(__file__).resolve()
        result["sources"][str(path)] = launch.sha(path)
        result["variant"] = NAME
        result["interpretation"] = ("fixed 24 known rows, post-outcome ablation; "
                                    "same visual rejection; not independent or equal total budget")
        return result

    launch.build = build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--directory")
    parser.add_argument("--report-sha")
    args = parser.parse_args()
    configure()
    print("Execute fixed RGB correction ablation; segmentation rejection remains fail-closed.", flush=True)
    if args.command == "prepare":
        print(json.dumps({"sha256": launch.prepare(args.output)}), flush=True)
        return 0
    if args.command == "run":
        launch.need(args.prepared and args.prepare_sha, "prepared contract")
        return launch.run(args.prepared, args.prepare_sha, args.output)
    out = Path(args.output)
    launch.need(args.prepared and args.prepare_sha and args.directory and args.report_sha
                and out.is_absolute() and out.parent == launch.ROOT / "results"
                and out.name.startswith(NAME + "_saved_") and not out.exists() and not out.is_symlink(),
                "exclusive saved output")
    result = saved.consume(args.directory, args.report_sha, args.prepared, args.prepare_sha)
    result["variant"] = NAME
    print(json.dumps({"sha256": launch.h10._write_new(out, result), "rows": len(result["rows"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
