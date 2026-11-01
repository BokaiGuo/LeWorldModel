"""Run the fixed diverse roster using the existing instrumented physical engine."""
import argparse
import json
from pathlib import Path

from . import p1_backtrack_physical_runner_20261030 as runner
from . import p1_backtrack_physical_consume_20261030 as consumer
from . import p1_diverse_physical_source_join_20261030_main as source

NAME = "p1_diverse_physical_runner_20261030_main"
ROOT = runner.ROOT
JOIN = ROOT / "results/p1_diverse_physical_source_join_20261030_main_RUN01.json"


def configure(join_sha):
    runner.native.actual_hash(join_sha)
    runner.NAME, runner.JOIN, runner.JOIN_SHA = NAME, JOIN, join_sha


def material(join_sha):
    configure(join_sha)
    p, data = runner.material()
    joined = runner.native.read(JOIN, cap=32 * 1024**2)
    expected_roles = ["incumbent", "diverse_selected"] * 8
    runner.require(joined.get("source_replay_sha256") == source.RESULT_SHA
                   and joined.get("diverse_replay_prepare_sha256") == source.PREPARE_SHA
                   and joined.get("diverse_join_sha256") == source.replay.JOIN_SHA
                   and joined.get("diverse_source_roles") == expected_roles
                   and joined.get("diverse_source_join_code_sha256") == runner.sha(Path(source.__file__))
                   and joined.get("role_mapping") == {"diverse_selected": "model_selected", "incumbent": "incumbent"},
                   "diverse source lineage")
    manifest = runner.native.read(source.PREPARE, cap=32 * 1024**2)
    runner.require(runner.sha(source.PREPARE) == source.PREPARE_SHA
                   and runner.sha(source.RESULT) == source.RESULT_SHA
                   and manifest["files"] == source.replay.bindings(), "replay sources")
    extra = [Path(__file__), Path(source.__file__), Path(source.consumer.__file__),
             Path(consumer.__file__), source.RESULT, source.PREPARE,
             ROOT.parent / "docs/D2P_P1_DIVERSE_PHYSICAL_PROTOCOL_20261030_MAIN.md"]
    for path in extra:
        path = str(path.resolve())
        digest = runner.sha(path)
        runner.require(path not in p["bindings"] or p["bindings"][path] == digest, "binding collision")
        p["bindings"][path] = digest
    for group in manifest["files"].values():
        for path, digest in group.items():
            runner.require(runner.sha(path) == digest and
                           (path not in p["bindings"] or p["bindings"][path] == digest), "replay binding drift")
            p["bindings"][path] = digest
    p["diverse_source_replay_sha256"] = source.RESULT_SHA
    p["diverse_source_roles"] = expected_roles
    return p, data


def prepare(path, join_sha):
    p, _ = material(join_sha)
    path = Path(path)
    runner.require(path.is_absolute() and path.parent == ROOT / "results" and
                   path.name.startswith(NAME + "_prepare_") and not path.exists() and
                   not path.is_symlink(), "exclusive prepare path")
    digest = source.replay.prior.one.saved.write(path, p)
    return {"path": str(path), "sha256": digest, "physical_authorized": False}


def verify(path, expected, join_sha):
    runner.native.actual_hash(expected)
    path = Path(path)
    runner.require(path.is_absolute() and path.is_file() and not path.is_symlink()
                   and runner.sha(path) == expected, "prepare SHA")
    saved = runner.native.read(path, cap=32 * 1024**2)
    current, data = material(join_sha)
    runner.require(saved == current and runner.sha(path) == expected, "prepare/source/code drift")
    return saved, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--join-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--report-sha")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.join_sha)
    elif args.command == "run":
        runner.require(args.authorize_physics, "explicit physics flag")
        p, data = verify(args.prepared, args.prepare_sha, args.join_sha)
        runner.native.constructor_contract()

        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

        print("Execute all 16 known-root diverse and incumbent actions; no P1 admission.", flush=True)
        result = runner.execute(p, data, runner.Store(args.output), factory, args.prepared,
                                args.prepare_sha, authorization=runner.AUTH)
        result = {"status": result["status"], "rows": len(result["rows"]),
                  "report_sha256": runner.sha(Path(args.output) / "report.json"),
                  "errors": result["errors"]}
    else:
        directory = Path(args.output)
        runner.require(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(), "directory")
        path = directory / "report.json"
        runner.require(runner.sha(path) == args.report_sha, "report SHA")
        report = runner.native.read(path, cap=64 * 1024**2)
        binding = report["execution_binding"]
        prepared, _ = verify(binding["prepared"], binding["prepare_sha256"], args.join_sha)
        runner.require(binding == runner.native.read(directory / "execution_binding.json")
                       and binding["join_sha256"] == args.join_sha, "execution binding")
        result = consumer.check(directory, report, prepared)
    print(json.dumps(result, sort_keys=True))
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
