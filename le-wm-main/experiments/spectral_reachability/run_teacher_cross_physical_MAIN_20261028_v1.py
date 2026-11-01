"""MAIN launcher: preserve native reports; print only bounded summaries."""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p1_teacher_cross_library_physical_20261028_v1 as physical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "consume"))
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-sha")
    parser.add_argument("--receipt")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        physical.require(args.authorize_physics and args.prepared and args.prepare_sha,
                         "explicit MAIN physics authorization and sealed prepare required")

        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                            with_target=False)

        result = physical.run(args.prepared, args.prepare_sha, args.output, factory,
                              authorization=physical.AUTH)
        artifact = Path(args.output) / "report.json"
    else:
        physical.require(args.report_sha and args.receipt, "actual report SHA and new receipt required")
        result = physical.consume(args.output, args.report_sha)
        artifact = Path(args.receipt)
        payload = physical.canonical(result)
        with artifact.open("xb") as stream:
            physical.require(stream.write(payload) == len(payload), "complete receipt write")
            stream.flush()
            physical.os.fsync(stream.fileno())
        physical.adapter.paired.fsync_dir(artifact.parent)
    statuses = {}
    for row in result["rows"]:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    print(json.dumps(dict(status=result["status"], artifact=str(artifact.absolute()),
                          sha256=physical.sha(artifact), rows=statuses,
                          counts=result["counts"], errors=result["errors"],
                          scientific_admission=False), sort_keys=True, allow_nan=False))
    return int("partial" in result["status"])


if __name__ == "__main__":
    raise SystemExit(main())
