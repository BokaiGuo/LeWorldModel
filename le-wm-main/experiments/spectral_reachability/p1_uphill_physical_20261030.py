"""Physical exploration of the fixed paired CUDA-replayed action roster."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_cuda_replay_20261030 as replay
from . import p1_backtrack_physical_source_join_20261030 as source
from . import p1_backtrack_physical_runner_20261030 as runner
from . import p1_backtrack_physical_consume_20261030 as consumer

ROOT = runner.ROOT
NAME = "p1_uphill_physical_20261030"
JOIN = ROOT / "results/p1_uphill_physical_20261030_source_RUN01.json"
MANIFEST = ROOT / "results/p1_uphill_cuda_replay_20261030_prepare_RUN01.json"
MANIFEST_SHA = "94a4faf335f8411a5d0af05a563e77d52e3627a44d190a8d148ce524b6da4ab8"
RESULT = ROOT / "results/p1_uphill_cuda_replay_20261030_RUN01.json"
RESULT_SHA = "67fb9fe7be0acce5392685be7edba52b170bbd7fb46a73ad87f905343dd3843f"
RECEIPT = ROOT / "results/p1_uphill_cuda_replay_20261030_consume_RUN01.json"
RECEIPT_SHA = "26a07821cb9cf9b9ef9c89d0fe2e4b4f2f14c0bafd310bd3070b89fe628d12f4"
PROTOCOL = ROOT.parent / "docs/D2P_P1_UPHILL_PHYSICAL_PROTOCOL_20261030_MAIN.md"
need, sha = runner.require, runner.sha


def build_join():
    replay.configure()
    read = replay.engine.old.one.saved.read_verified
    manifest, value, receipt = read(MANIFEST, MANIFEST_SHA), read(RESULT, RESULT_SHA), read(RECEIPT, RECEIPT_SHA)
    need(value["manifest"] == str(MANIFEST) and value["manifest_sha256"] == MANIFEST_SHA
         and receipt["result_sha256"] == RESULT_SHA and receipt["prepare_sha256"] == MANIFEST_SHA,
         "replay identity")
    rows, joint, backtrack, report, prepared = replay.sources()
    checked = replay.consumer.verify(value, manifest, rows, joint, backtrack, report, prepared)
    need(checked["all_exact"] and checked["normalization_and_context_verified"], "saved verification")
    actions, comparisons = [], []
    for i, (row, observed) in enumerate(zip(rows, value["rows"], strict=True)):
        role = ("incumbent", "model_selected")[i % 2]
        raw = np.frombuffer(bytes.fromhex(row["raw15_le_float32_hex"]), dtype="<f4").reshape(15, 2)
        identity = {k: row[k] for k in ("slot", "arm", "candidate_index")}
        actions.append({**identity, "role": role, "raw15": raw.tolist(),
                        "raw15_le_float32_hex": raw.tobytes().hex()})
        comparisons.append({**identity, "role": role, "exact": observed["exact"]})
    data = source.physical.source_material()
    joined = source.assemble({"rows": actions}, {"rows": comparisons}, prepared, data)
    for i, row in enumerate(joined["rows"]):
        need(row["task_id"] == rows[i]["task_binding"]["task_id"]
             and row["raw15_float32_hex"] == rows[i]["raw15_le_float32_hex"], "task/action")
    joined.update(source_replay_sha256=RESULT_SHA, replay_manifest_sha256=MANIFEST_SHA,
                  replay_receipt_sha256=RECEIPT_SHA,
                  role_meanings={"incumbent": "original_joint_terminal_winner",
                                 "model_selected": "latent_nonimprovement_rerank"})
    return joined


def material(join_sha):
    runner.NAME, runner.JOIN, runner.JOIN_SHA = NAME, JOIN, join_sha
    prepared, data = runner.material()
    joined = runner.native.read(JOIN, cap=32 * 1024**2)
    need(joined == build_join() and sha(JOIN) == join_sha, "join rebuild")
    replay.configure()
    manifest = replay.engine.old.one.saved.read_verified(MANIFEST, MANIFEST_SHA)
    need(manifest["files"] == replay.files(), "replay sources")
    for path in (Path(__file__), Path(consumer.__file__), MANIFEST, RESULT, RECEIPT, PROTOCOL):
        prepared["bindings"][str(path.resolve())] = sha(path)
    for path, digest in manifest["files"].items():
        need(sha(path) == digest, "source drift")
        need(path not in prepared["bindings"] or prepared["bindings"][path] == digest, "binding collision")
        prepared["bindings"][path] = digest
    need(all(sha(path) == digest for path, digest in prepared["bindings"].items()), "binding drift")
    return prepared, data


def verify(path, digest, join_sha):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and sha(path) == digest, "prepare sha")
    saved = runner.native.read(path, cap=32*1024**2)
    current, data = material(join_sha)
    need(saved == current and sha(path) == digest, "prepare/source drift")
    return saved, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("source", "prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--join-sha")
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--report-sha")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    write = replay.engine.old.one.saved.write
    if args.command == "source":
        need(out == JOIN and not out.exists(), "exclusive source")
        result = {"sha256": write(out, build_join())}
    elif args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_prepare_") and not out.exists(), "exclusive prepare")
        prepared, _ = material(args.join_sha)
        result = {"sha256": write(out, prepared)}
    elif args.command == "run":
        need(args.authorize_physics, "explicit physics authorization")
        prepared, data = verify(args.prepared, args.prepare_sha, args.join_sha)
        runner.native.constructor_contract()
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
        print("Execute all 16 fixed paired known-root physical trajectories.", flush=True)
        report = runner.execute(prepared, data, runner.Store(out), factory, args.prepared,
                                args.prepare_sha, authorization=runner.AUTH)
        result = {"status": report["status"], "rows": len(report["rows"]), "errors": report["errors"],
                  "report_sha256": sha(out / "report.json")}
    else:
        need(out.is_absolute() and out.is_dir() and not out.is_symlink(), "directory")
        need(sha(out / "report.json") == args.report_sha, "report sha")
        report = runner.native.read(out / "report.json", cap=64*1024**2)
        binding = report["execution_binding"]
        prepared, _ = verify(binding["prepared"], binding["prepare_sha256"], args.join_sha)
        need(binding == runner.native.read(out / "execution_binding.json")
             and binding["join_sha256"] == args.join_sha, "execution binding")
        result = consumer.check(out, report, prepared)
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
