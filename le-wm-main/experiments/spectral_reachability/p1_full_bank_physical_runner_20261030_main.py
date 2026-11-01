"""Run the fixed full-bank roster through the existing audited physical engine."""
import argparse
import json
from pathlib import Path

from . import p1_full_bank_physical_source_20261030_main as full_source
from . import p1_diverse_physical_runner_20261030_main as diverse

NAME = "p1_full_bank_physical_runner_20261030_main"
ROOT = diverse.ROOT
JOIN = ROOT / "results/p1_full_bank_physical_source_20261030_main_RUN01.json"
PROTOCOL = ROOT.parent / "docs/D2P_P1_FULL_BANK_PHYSICAL_PROTOCOL_20261030_MAIN.md"


def material(join_sha):
    full_source.configure()
    diverse.NAME, diverse.JOIN = NAME, JOIN
    p, data = diverse.material(join_sha)
    joined = diverse.runner.native.read(JOIN, cap=32 * 1024**2)
    diverse.runner.require(joined["full_bank_selection_sha256"] == full_source.full.SELECTED_SHA
                           and joined["full_bank_source_code_sha256"] ==
                           diverse.runner.sha(Path(full_source.__file__)), "full-bank source closure")
    for path in (Path(__file__), Path(full_source.__file__), PROTOCOL):
        path = str(path.resolve())
        digest = diverse.runner.sha(path)
        diverse.runner.require(path not in p["bindings"] or p["bindings"][path] == digest, "binding collision")
        p["bindings"][path] = digest
    return p, data


def verify(path, digest, join_sha):
    runner = diverse.runner
    path = Path(path)
    runner.require(path.is_absolute() and path.is_file() and not path.is_symlink()
                   and runner.sha(path) == digest, "prepare SHA")
    p = runner.native.read(path, cap=32 * 1024**2)
    current, data = material(join_sha)
    runner.require(p == current and runner.sha(path) == digest, "prepare drift")
    return p, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--join-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--report-sha")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    runner = diverse.runner
    if args.command == "prepare":
        p, _ = material(args.join_sha)
        path = Path(args.output)
        runner.require(path.is_absolute() and path.parent == ROOT / "results"
                       and path.name.startswith(NAME + "_prepare_") and not path.exists(), "exclusive prepare")
        result = {"sha256": full_source.full.replay.prior.one.saved.write(path, p), "physical_authorized": False}
    elif args.command == "run":
        runner.require(args.authorize_physics, "explicit physical authorization")
        p, data = verify(args.prepared, args.prepare_sha, args.join_sha)
        runner.native.constructor_contract()

        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

        print("Execute all 16 fixed full-bank and incumbent actions; no P1 admission.", flush=True)
        report = runner.execute(p, data, runner.Store(args.output), factory, args.prepared,
                                args.prepare_sha, authorization=runner.AUTH)
        result = {"status": report["status"], "rows": len(report["rows"]), "errors": report["errors"],
                  "report_sha256": runner.sha(Path(args.output) / "report.json")}
    else:
        directory = Path(args.output)
        runner.require(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(), "directory")
        path = directory / "report.json"
        runner.require(runner.sha(path) == args.report_sha, "report SHA")
        report = runner.native.read(path, cap=64 * 1024**2)
        binding = report["execution_binding"]
        p, _ = verify(binding["prepared"], binding["prepare_sha256"], args.join_sha)
        runner.require(binding == runner.native.read(directory / "execution_binding.json")
                       and binding["join_sha256"] == args.join_sha, "execution binding")
        result = diverse.consumer.check(directory, report, p)
    print(json.dumps(result, sort_keys=True))
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
