"""Collect fixed response-diverse prefixes and real cp5 RGB on known roots."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_uphill_physical_20261030 as uphill
from . import p1_response_common_suffix20_replay_20261030_main as replay

runner = uphill.runner
ROOT = runner.ROOT
NAME = "p1_response_prefix_pilot_20261031"
JOIN = ROOT / f"results/{NAME}_source_RUN01.json"
REPLAY = ROOT / "results/p1_response_common_suffix20_replay_20261030_main_RUN01.json"
REPLAY_SHA = "4f0c5fd74436c74001fc3fa800b1ae523cc43a47ad38576125ae9b6c2a9310dc"
PROTOCOL = ROOT.parent / "docs/D2P_P1_RESPONSE_PREFIX_PILOT_PROTOCOL_20261031.md"
need, sha = runner.require, runner.sha


def assemble(joined, model, confirmation):
    need(confirmation["status"] == "complete_cuda_replay"
         and confirmation["source_outcome_sha256"] == replay.OUTCOME_SHA
         and len(confirmation["rows"]) == 54, "CUDA replay contract")
    need(len(model["rows"]) == 8 and len(joined["rows"]) == 16, "fixed denominator")
    for logical, arm in enumerate(model["rows"]):
        slot = logical // 2
        need(arm["slot"] == slot and arm["arm"] == runner.contract.ARMS[logical % 2],
             "slot and arm order")
        indices = arm["candidate_sets"]["diverse"][:2]
        need(len(indices) == 2 and indices[0] != indices[1], "fixed diversity roster")
        for offset, candidate in enumerate(indices):
            row = joined["rows"][2 * logical + offset]
            need(row["slot"] == slot and row["arm"] == arm["arm"]
                 and row["role"] == runner.contract.ROLES[offset], "physical slot identity")
            matches = [v for v in confirmation["rows"]
                       if v["slot"] == slot and v["arm"] == arm["arm"] and v["candidate"] == candidate]
            need(len(matches) == 1, "CUDA replay candidate")
            raw = bytes.fromhex(arm["candidates"][str(candidate)]["prefix_float32_hex"])
            actions = np.frombuffer(raw, dtype="<f4")
            need(len(raw) == 40 and np.isfinite(actions).all() and (np.abs(actions) <= 1).all(),
                 "literal bounded prefix")
            row["candidate_index"] = candidate
            row["raw15_float32_hex"] = (raw + bytes(80)).hex()
    joined["role_meanings"] = {"incumbent": "diversity_list_first_immediate_seed",
                               "model_selected": "diversity_list_second_not_a_selected_winner"}
    joined["pilot_protocol_sha256"] = sha(PROTOCOL)
    joined["model_panel_sha256"] = replay.OUTCOME_SHA
    joined["model_replay_sha256"] = REPLAY_SHA
    return joined


def build_join():
    report, _, selection = replay.source.inputs()
    model = replay.source.panel.source.saved.read_verified(replay.OUTCOME, replay.OUTCOME_SHA)
    replay.validate_result(model, report, selection)
    confirmation = replay.source.panel.source.saved.read_verified(REPLAY, REPLAY_SHA)
    return assemble(uphill.build_join(), model, confirmation)


def configure(join_sha):
    runner.NAME, runner.JOIN, runner.JOIN_SHA = NAME, JOIN, join_sha


def material(join_sha):
    configure(join_sha)
    prepared, data = runner.material()
    need(runner.native.read(JOIN, cap=32 * 1024**2) == build_join(), "rebuild join")
    for path in (Path(__file__).resolve(), PROTOCOL, replay.OUTCOME, REPLAY):
        prepared["bindings"][str(path)] = sha(path)
    need(all(sha(p) == d for p, d in prepared["bindings"].items()), "binding drift")
    return prepared, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("source", "prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--join-sha")
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha")
    parser.add_argument("--report-sha")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    write = replay.source.base.write
    if args.command == "source":
        need(out == JOIN and not out.exists(), "exclusive source")
        result = {"sha256": write(out, build_join())}
    elif args.command == "prepare":
        need(out.is_absolute() and out.parent == ROOT / "results"
             and out.name.startswith(NAME + "_prepare_") and not out.exists(), "exclusive prepare")
        prepared, _ = material(args.join_sha)
        result = {"sha256": write(out, prepared)}
    else:
        need(args.join_sha and args.prepared and args.prepare_sha, "source identity required")
        need(sha(args.prepared) == args.prepare_sha, "prepare hash")
        prepared, data = material(args.join_sha)
        need(prepared == runner.native.read(args.prepared, cap=32 * 1024**2), "prepare drift")
        if args.command == "run":
            need(args.authorize_physics, "physical authorization")
            runner.native.constructor_contract()
            def factory():
                import gymnasium as gym
                import stable_worldmodel  # noqa: F401
                return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
            print("Execute all 16 predetermined prefixes on known roots; no neural inference.", flush=True)
            report = runner.execute(prepared, data, runner.Store(out), factory, args.prepared,
                                    args.prepare_sha, authorization=runner.AUTH)
            result = {"status": report["status"], "rows": len(report["rows"]),
                      "errors": report["errors"], "report_sha256": sha(out / "report.json")}
        else:
            need(out.is_dir() and not out.is_symlink() and sha(out / "report.json") == args.report_sha,
                 "physical report")
            report = runner.native.read(out / "report.json", cap=64 * 1024**2)
            need(report["execution_binding"]["prepare_sha256"] == args.prepare_sha
                 and report["execution_binding"]["join_sha256"] == args.join_sha,
                 "physical binding")
            result = uphill.consumer.check(out, report, prepared)
    print(json.dumps(result, sort_keys=True), flush=True)
    return int(result.get("status") == "partial_or_failed")


if __name__ == "__main__":
    raise SystemExit(main())
