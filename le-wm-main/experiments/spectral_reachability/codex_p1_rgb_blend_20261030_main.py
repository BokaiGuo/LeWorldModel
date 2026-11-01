"""Fixed RGB/model convex blend on the known P1 development panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_rgb_pusher_ablation_20261030_main as old

NAME = "codex_p1_rgb_blend_20261030_main"
launch, saved = old.launch, old.saved
POLICY = {**old.POLICY, "schema": NAME, "correction_schedule": "half_model_half_five_pulse",
          "model_weight": 0.5, "meaning": "known-root exploratory blend, not a safety policy"}


def decide(current_rgb, goal_rgb, model_first_packet):
    result = old.decide(current_rgb, goal_rgb, model_first_packet)
    if result["decision"] == "rgb_pusher_correction":
        correction = np.repeat(result["packet"][:1], 5, axis=0).astype("<f4", copy=False)
        packet = (np.float32(0.5) * model_first_packet + np.float32(0.5) * correction).astype("<f4")
        launch.need(np.isfinite(packet).all() and (np.abs(packet) <= 1).all(), "blend action bounds")
        result["packet"] = packet
        result["packet_sha256"] = hashlib.sha256(packet.tobytes()).hexdigest()
    return result


def configure():
    old.configure()
    previous_build = launch.build
    launch.NAME = NAME
    launch.rgb_candidate.POLICY = POLICY
    launch.rgb_candidate.decide = decide

    def build():
        result = previous_build()
        path = Path(__file__).resolve()
        result["sources"][str(path)] = launch.sha(path)
        result["variant"] = NAME
        result["interpretation"] = "fixed convex blend; known roots; post-outcome exploratory"
        return result
    launch.build = build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    for arg in ("output", "prepared", "prepare-sha", "directory", "report-sha"):
        parser.add_argument("--" + arg, required=arg == "output")
    args = parser.parse_args()
    configure()
    print("Fixed half-model half-RGB action blend; retain all rejection and stop states.", flush=True)
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
