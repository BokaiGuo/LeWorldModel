"""Post-outcome five-pulse ablation of the fixed RGB pusher correction."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import codex_p1_rgb_pusher_ablation_20261030_main as old

NAME = "codex_p1_rgb_pusher_fivepulse_20261030_main"
launch, saved = old.launch, old.saved
POLICY = {**old.POLICY, "schema": NAME, "correction_schedule": "five_identical_relative_actions",
          "meaning": "post-outcome fixed five-pulse RGB correction; not a safety policy"}


def decide(current_rgb, goal_rgb, model_first_packet):
    result = old.decide(current_rgb, goal_rgb, model_first_packet)
    if result["decision"] == "rgb_pusher_correction":
        packet = np.repeat(result["packet"][:1], 5, axis=0).astype("<f4", copy=False)
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
        result["interpretation"] = ("five identical pulses; known fixed roots, post-outcome exploration; "
                                    "same fail-closed visual checks")
        return result
    launch.build = build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    for arg in ("output", "prepared", "prepare-sha", "directory", "report-sha"):
        parser.add_argument("--" + arg, required=arg == "output")
    args = parser.parse_args()
    configure()
    print("Five identical correction pulses; retain RGB rejection and all stops.", flush=True)
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
