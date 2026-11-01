"""Post-outcome image-scale guard for the five-pulse RGB feedback experiment."""
import argparse
import hashlib
import json
from pathlib import Path

from . import codex_p1_rgb_pusher_fivepulse_20261030_main as five

NAME = "codex_p1_rgb_shape_guard_20261030_main"
launch, saved = five.launch, five.saved
# Scale conversion only, not a claim that Chamfer equals object position.
SHAPE_LIMIT_PIXELS = 20.0 * 224.0 / 512.0
POLICY = {**five.POLICY, "schema": NAME,
          "object_shape_chamfer_max_pixels": SHAPE_LIMIT_PIXELS,
          "meaning": "post-outcome image-scale guard, not a native success classifier or safety guarantee"}


def decide(current_rgb, goal_rgb, model_first_packet):
    result = five.decide(current_rgb, goal_rgb, model_first_packet)
    if (result["decision"] == "rgb_pusher_correction"
            and result["features"]["object_shape_chamfer_pixels"] > SHAPE_LIMIT_PIXELS):
        packet = model_first_packet.copy()
        result.update(decision="model_packet", packet=packet,
                      packet_sha256=hashlib.sha256(packet.tobytes()).hexdigest())
    return result


def configure():
    five.configure()
    previous_build = launch.build
    launch.NAME = NAME
    launch.rgb_candidate.POLICY = POLICY
    launch.rgb_candidate.decide = decide

    def build():
        result = previous_build()
        path = Path(__file__).resolve()
        result["sources"][str(path)] = launch.sha(path)
        result["variant"] = NAME
        result["interpretation"] = "shape guard is image-only and post-outcome; no native success inference"
        return result
    launch.build = build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    for arg in ("output", "prepared", "prepare-sha", "directory", "report-sha"):
        parser.add_argument("--" + arg, required=arg == "output")
    args = parser.parse_args()
    configure()
    print("Fixed image-shape guard; original segmentation rejection remains fail-closed.", flush=True)
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
