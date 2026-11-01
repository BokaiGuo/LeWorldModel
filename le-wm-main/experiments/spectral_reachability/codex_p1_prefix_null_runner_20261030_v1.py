"""Known-root prefix-null execution adapter; never itself authorizes physics."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

from . import codex_p1_prefix_null_prepare_20261030_v1 as roster
from . import p1_backtrack_physical_runner_20261030 as base

NAME = "codex_p1_prefix_null_runner_20261030_v1"
AUTH = "PREFIX_NULL_FIXED16_KNOWN_ROOT_EXPLORATION"
PREPARED = roster.ROOT / "results/codex_p1_prefix_null_20261030_v1_prepare_CODEX03.json"
PREPARED_SHA = "589d728541fc575db3d5b97edda0e2b75505982f6da59f26e6c3cfc3980e798a"
CONSUMER = Path(__file__).with_name("codex_p1_prefix_null_saved_consumer_20261030_v1.py")
require = base.require


def material():
    frozen = roster.verify(PREPARED, PREPARED_SHA)
    upstream, data = base.material()
    require(upstream["join_sha256"] == frozen["join_sha256"] and
            upstream["contract"] == frozen["contract"], "source or contract mismatch")
    # Independently rebuild from the join; no outcome-based row selection.
    require(roster.build(roster.load()) == frozen, "fixed roster mismatch")
    bindings = copy.deepcopy(upstream["bindings"])
    for path in (Path(__file__).resolve(), CONSUMER.resolve(), Path(roster.__file__).resolve(), PREPARED):
        require(path.is_file() and not path.is_symlink(), "regular bound source")
        digest = base.sha(path)
        require(str(path) not in bindings or bindings[str(path)] == digest, "binding collision")
        bindings[str(path)] = digest
    require(bindings[str(PREPARED)] == PREPARED_SHA, "roster SHA")
    return {"schema": NAME + ".prepare", "fixed_roster_sha256": PREPARED_SHA,
            "plans": frozen["plans"], "task_rows": frozen["task_rows"],
            "contract": frozen["contract"], "bindings": bindings,
            "deployment_status": frozen["deployment_status"],
            "P1_admission": False, "physical_authorized": False,
            "independent_confirmation": False, "exploratory": True}, data


def prepare(path):
    path = Path(path)
    require(path.is_absolute() and path.parent == roster.ROOT / "results" and
            path.name.startswith(NAME + "_prepare_") and not path.is_symlink(), "exclusive prepare path")
    result, _ = material()
    raw = base.native.raw(result)
    with path.open("xb") as stream:
        require(stream.write(raw) == len(raw), "short write")
        stream.flush()
        os.fsync(stream.fileno())
    base.native.paired.fsync_dir(path.parent)
    return {"path": str(path), "sha256": base.sha(path), "physical_authorized": False}


def verify(path, digest):
    base.native.actual_hash(digest)
    path = Path(path)
    require(path.is_absolute() and path.parent == roster.ROOT / "results" and
            path.name.startswith(NAME + "_prepare_") and path.is_file() and
            not path.is_symlink() and base.sha(path) == digest, "prepare identity")
    saved = base.native.read(path, cap=32 * 1024**2)
    current, data = material()
    require(saved == current and base.sha(path) == digest, "source or code drift")
    return saved, data


class Store(base.Store):
    def __init__(self, path):
        self.path = Path(path)
        c = base.contract.contract()
        require(self.path.is_absolute() and self.path.parent == roster.ROOT / "results" and
                self.path.name.startswith(NAME + "_run_") and not self.path.exists() and
                not self.path.is_symlink() and
                shutil.disk_usage(self.path.parent).free >= c["disk_floor_bytes"] + c["output_cap_bytes"],
                "exclusive bounded output and disk reserve")
        self.path.mkdir()
        base.native.paired.fsync_dir(self.path.parent)
        self.previous, self.index = None, 0


def execute(prepared_path, expected, output, authorization):
    require(authorization == AUTH, "explicit exploration authorization required")
    prepared, data = verify(prepared_path, expected)
    require(prepared["deployment_status"] == "certified_for_known_root_exploration",
            "frozen roster remains deployment-blocked")
    base.native.constructor_contract()

    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

    # The adapter retains all 16 row shells, writes operation intents before
    # dispatch, aborts on engineering failures, and leaves remaining rows unstarted.
    return base.execute(prepared, data, Store(output), factory, prepared_path, expected,
                        authorization=base.AUTH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--authorization")
    args = parser.parse_args()
    try:
        value = prepare(args.output) if args.command == "prepare" else execute(
            args.prepared, args.prepare_sha, args.output, args.authorization)
        print(json.dumps(value, sort_keys=True, allow_nan=False))
        return 0 if value.get("status") != "partial_or_failed" else 3
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "P1_admission": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
