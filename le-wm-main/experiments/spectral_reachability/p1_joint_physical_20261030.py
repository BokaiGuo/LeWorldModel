"""Bind and execute the fixed CUDA-replayed joint roster in known roots."""
import argparse
import json
from pathlib import Path

import numpy as np
from . import p1_joint_cuda_replay_20261030 as replay
from . import p1_joint_cuda_replay_consume_20261030 as consume
from . import p1_backtrack_physical_source_join_20261030 as source
from . import p1_backtrack_physical_runner_20261030 as runner
from . import p1_backtrack_physical_consume_20261030 as consumer

ROOT = replay.ROOT
NAME = "p1_joint_physical_20261030"
MANIFEST = ROOT / "results/p1_joint_cuda_replay_20261030_prepare_RUN01.json"
MANIFEST_SHA = "f367458185b5092e0991c58d9c8d6085397dae468485f8dcc52bd3804809ff43"
RESULT = ROOT / "results/p1_joint_cuda_replay_20261030_RUN01.json"
RESULT_SHA = "9e762379b8ba3e4cf44caa1eb01ea06f8b645b6cdda4f591a84f063eed37e554"
JOIN = ROOT / "results/p1_joint_physical_20261030_source_RUN01.json"
PROTOCOL = ROOT.parent / "docs/D2P_P1_JOINT_PHYSICAL_PROTOCOL_20261030_MAIN.md"


def build_join():
    saved = replay.old.one.saved
    manifest = saved.read_verified(MANIFEST, MANIFEST_SHA)
    result = saved.read_verified(RESULT, RESULT_SHA)
    runner.require(result["manifest"] == str(MANIFEST) and result["manifest_sha256"] == MANIFEST_SHA,
                   "replay manifest")
    rows, joint, backtrack, report, prepared = replay.sources()
    consume.verify(result, manifest, rows, joint, backtrack, report, prepared)
    actions, comparisons = [], []
    for row, observed in zip(rows, result["rows"], strict=True):
        role = {"incumbent": "incumbent", "joint_selected": "model_selected"}[row["role"]]
        raw = np.frombuffer(bytes.fromhex(row["raw15_le_float32_hex"]), dtype="<f4").reshape(15, 2)
        identity = {k: row[k] for k in ("slot", "arm", "candidate_index")}
        actions.append({**identity, "role": role, "raw15": raw.tolist(),
                        "raw15_le_float32_hex": raw.tobytes().hex()})
        comparisons.append({**identity, "role": role, "exact": observed["exact"]})
    data = source.physical.source_material()
    joined = source.assemble({"rows": actions}, {"rows": comparisons}, prepared, data)
    for i, row in enumerate(joined["rows"]):
        runner.require(row["task_id"] == rows[i]["task_binding"]["task_id"]
                       and row["raw15_float32_hex"] == rows[i]["raw15_le_float32_hex"], "task/action")
    for path, digest in data["bindings"].items():
        runner.require(runner.sha(path) == digest, "native source drift")
    joined.update(source_replay_sha256=RESULT_SHA, joint_prepare_sha256=MANIFEST_SHA,
                  joint_source_sha256=replay.check.REPORT_SHA)
    return joined


def material(join_sha):
    runner.NAME, runner.JOIN, runner.JOIN_SHA = NAME, JOIN, join_sha
    p, data = runner.material()
    joined = runner.native.read(JOIN, cap=32 * 1024**2)
    runner.require(joined["source_replay_sha256"] == RESULT_SHA
                   and joined["joint_prepare_sha256"] == MANIFEST_SHA
                   and joined["joint_source_sha256"] == replay.check.REPORT_SHA, "lineage")
    manifest = replay.old.one.saved.read_verified(MANIFEST, MANIFEST_SHA)
    runner.require(runner.sha(RESULT) == RESULT_SHA and manifest["files"] == replay.files(), "replay drift")
    paths = [Path(__file__), Path(consume.__file__), Path(consumer.__file__), RESULT, MANIFEST, PROTOCOL]
    for path in paths:
        p["bindings"][str(path.resolve())] = runner.sha(path)
    for path, digest in manifest["files"].items():
        runner.require(runner.sha(path) == digest, "bound input drift")
        p["bindings"][path] = digest
    return p, data


def verify(path, digest, join_sha):
    path = Path(path)
    runner.require(path.is_file() and not path.is_symlink() and runner.sha(path) == digest, "prepare SHA")
    p = runner.native.read(path, cap=32 * 1024**2)
    current, data = material(join_sha)
    runner.require(p == current and runner.sha(path) == digest, "prepare drift")
    return p, data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("source", "prepare", "run", "consume"))
    p.add_argument("--output", required=True)
    p.add_argument("--join-sha")
    p.add_argument("--prepared")
    p.add_argument("--prepare-sha")
    p.add_argument("--report-sha")
    p.add_argument("--authorize-physics", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    if a.command == "source":
        runner.require(out == JOIN and not out.exists(), "exclusive source")
        result = {"sha256": replay.old.one.saved.write(out, build_join())}
    elif a.command == "prepare":
        data, _ = material(a.join_sha)
        runner.require(out.is_absolute() and out.parent == ROOT / "results"
                       and out.name.startswith(NAME + "_prepare_") and not out.exists(), "exclusive prepare")
        result = {"sha256": replay.old.one.saved.write(out, data)}
    elif a.command == "run":
        runner.require(a.authorize_physics, "explicit physics authorization")
        prepared, data = verify(a.prepared, a.prepare_sha, a.join_sha)
        runner.native.constructor_contract()
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
        print("Execute all 16 joint and incumbent actions, known roots only.", flush=True)
        report = runner.execute(prepared, data, runner.Store(out), factory, a.prepared,
                                a.prepare_sha, authorization=runner.AUTH)
        result = {"status": report["status"], "rows": len(report["rows"]), "errors": report["errors"],
                  "report_sha256": runner.sha(out / "report.json")}
    else:
        runner.require(out.is_absolute() and out.is_dir() and not out.is_symlink(), "directory")
        path = out / "report.json"
        runner.require(runner.sha(path) == a.report_sha, "report SHA")
        report = runner.native.read(path, cap=64 * 1024**2)
        binding = report["execution_binding"]
        prepared, _ = verify(binding["prepared"], binding["prepare_sha256"], a.join_sha)
        runner.require(binding == runner.native.read(out / "execution_binding.json")
                       and binding["join_sha256"] == a.join_sha, "execution binding")
        result = consumer.check(out, report, prepared)
    print(json.dumps(result, sort_keys=True))
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
