"""Known-root feedback exploration with 10 optimizer updates per phase.

Matches the open 20-update run only on backward count, not on encoding,
forward calls, wall time, or complete end-to-end resources.
"""
import argparse
import json
from pathlib import Path

from . import codex_p1_two_stage_launcher_20261030_v1 as launcher
from . import codex_p1_two_stage_saved_20261030_v1 as saved

NAME = "codex_p1_feedback_10x2_20261030_main"
ORIGINAL_BUILD = launcher.build


def configure():
    # All mutations are process-local; sealed historical source files are intact.
    launcher.episode.feedback.CONFIG = {"steps": 10, "learning_rate": 0.05, "history_size": 3}
    caps = dict.fromkeys(launcher.base.spec.AXES, 0)
    caps.update(action_encode_calls=22, predict_calls=22, predict_samples=22,
                backward_calls=10, vjp_rows=10, replay_calls=1, selection_evaluations=1)
    launcher.stage.CAPS = caps

    def build():
        result = ORIGINAL_BUILD()
        result["sources"][str(Path(__file__).resolve())] = launcher.sha(Path(__file__).resolve())
        result["variant"] = NAME
        result["comparison"] = ("20 total backward calls when both phases complete; "
                                "not equal total cost, known roots and post-outcome exploration")
        return result

    launcher.build = build


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
    print("Fixed known-root 2x10 feedback; backward-matched only, not equal total budget.", flush=True)
    if args.command == "prepare":
        print(json.dumps({"sha256": launcher.prepare(args.output)}))
    elif args.command == "run":
        launcher.need(args.prepared and args.prepare_sha, "prepare required")
        return launcher.run(args.prepared, args.prepare_sha, args.output)
    else:
        launcher.need(args.prepared and args.prepare_sha and args.directory and args.report_sha,
                      "saved verification inputs required")
        out = Path(args.output)
        launcher.need(out.is_absolute() and out.parent == launcher.ROOT / "results"
                      and out.name.startswith(NAME + "_saved_"), "exclusive saved output")
        result = saved.consume(args.directory, args.report_sha, args.prepared, args.prepare_sha)
        result["variant"] = NAME
        print(json.dumps({"sha256": launcher.h10._write_new(out, result), "rows": len(result["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
