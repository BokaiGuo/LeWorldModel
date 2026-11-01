"""Known-root prefix-null v2. Preparation is read-only; run performs physics."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from . import codex_p1_prefix_null_runner_20261030_v1 as v1

NAME = "codex_p1_prefix_null_runner_20261030_v2"
ROOT = v1.roster.ROOT
PREVIOUS = ROOT / "results/codex_p1_prefix_null_runner_20261030_v1_prepare_CODEX03.json"
PREVIOUS_SHA = "ea27d8e80e1408d9690221195b027c9d4e372c38ac240d430dacec10576ada2d"
CONSUMER = Path(__file__).with_name("codex_p1_prefix_null_saved_consumer_20261030_v2.py")
TEST = ROOT / "tests/test_codex_p1_prefix_null_v2_20261030.py"
DESIGN = ROOT.parent / "docs/D2P_P1_PREFIX_NULL_V2_CODEX_20261030.md"
need = v1.require


def check_roster(p):
    need(p["contract"] == v1.base.contract.contract() and p["contract"]["fixed_rows"] == 16 and
         p["P1_admission"] is False and p["independent_confirmation"] is False and
         len(p["plans"]) == 16, "fixed original contract")
    groups = {}
    for i in range(8):
        a, b = p["plans"][2*i:2*i+2]
        need(a["index"] == 2*i and b["index"] == 2*i+1 and
             a["logical_plan"] == b["logical_plan"] == i and
             (a["role"], b["role"]) == ("selected_prefix", "zero_prefix") and
             all(a[k] == b[k] for k in ("slot", "root_slot", "arm", "condition", "task_id",
                                         "physical_root_sha256")), "fixed pair identity")
        x, y = bytes.fromhex(a["raw15_float32_hex"]), bytes.fromhex(b["raw15_float32_hex"])
        need(len(x) == len(y) == 120 and y[:40] == bytes(40) and x[40:] == y[40:] and
             a["raw15_sha256"] == hashlib.sha256(x).hexdigest() and
             b["raw15_sha256"] == hashlib.sha256(y).hexdigest() and
             a["suffix_sha256"] == b["suffix_sha256"] == hashlib.sha256(x[40:]).hexdigest(),
             "fixed pair action bytes")
        for row, raw in ((a, x), (b, y)):
            key = (row["task_id"], row["physical_root_sha256"], row["condition"], raw)
            group = row["duplicate_action_task_group"]
            need(type(group) is int and group == groups.setdefault(key, len(groups)), "duplicate identity")


def material():
    previous, data = v1.verify(PREVIOUS, PREVIOUS_SHA)
    check_roster(previous)
    v1.base.native.constructor_contract()  # static source verification; no environment
    bindings = copy.deepcopy(previous["bindings"])
    for path in (Path(__file__).resolve(), CONSUMER, TEST, DESIGN, PREVIOUS):
        need(path.is_file() and not path.is_symlink(), "regular source required")
        digest = v1.base.sha(path)
        need(str(path) not in bindings or bindings[str(path)] == digest, "source binding collision")
        bindings[str(path)] = digest
    need(bindings[str(PREVIOUS)] == PREVIOUS_SHA, "v1 seal drift")
    return {"schema": NAME + ".prepare", "previous_sha256": PREVIOUS_SHA,
            "plans": previous["plans"], "task_rows": previous["task_rows"],
            "contract": previous["contract"], "bindings": bindings,
            "physical_authorized": False, "P1_admission": False,
            "independent_confirmation": False, "equal_budget_certified": False,
            "exploratory": True}, data


def prepare(path):
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         path.name.startswith(NAME + "_prepare_") and not path.is_symlink(), "exclusive prepare path")
    p, _ = material()
    raw = v1.base.native.raw(p)
    with path.open("xb") as stream:
        need(stream.write(raw) == len(raw), "short prepare write")
        stream.flush()
        os.fsync(stream.fileno())
    v1.base.native.paired.fsync_dir(path.parent)
    return {"path": str(path), "sha256": v1.base.sha(path), "P1_admission": False}


def verify(path, digest):
    v1.base.native.actual_hash(digest)
    path = Path(path)
    need(path.is_absolute() and path.parent == ROOT / "results" and
         path.name.startswith(NAME + "_prepare_") and path.is_file() and
         not path.is_symlink() and v1.base.sha(path) == digest, "prepare identity")
    saved = v1.base.native.read(path, cap=32 * 1024**2)
    current, data = material()
    need(saved == current and v1.base.sha(path) == digest, "source or code drift")
    return saved, data


class Store(v1.base.Store):
    def __init__(self, path):
        self.path = Path(path)
        c = v1.base.contract.contract()
        need(self.path.is_absolute() and self.path.parent == ROOT / "results" and
             self.path.name.startswith(NAME + "_run_") and not self.path.exists() and
             not self.path.is_symlink() and
             shutil.disk_usage(self.path.parent).free >= c["disk_floor_bytes"] + c["output_cap_bytes"],
             "exclusive bounded output and disk reserve")
        self.path.mkdir()
        v1.base.native.paired.fsync_dir(self.path.parent)
        self.previous, self.index = None, 0


def execute(path, digest, output):
    prepared, data = verify(path, digest)
    v1.base.native.constructor_contract()
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    return v1.base.execute(prepared, data, Store(output), factory, path, digest, authorization=v1.base.AUTH)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    args = parser.parse_args()
    try:
        result = prepare(args.output) if args.command == "prepare" else execute(
            args.prepared, args.prepare_sha, args.output)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 3 if result.get("status") == "partial_or_failed" else 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc), "P1_admission": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
