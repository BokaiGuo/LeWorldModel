"""Bind and rerun the identical sealed B0 physical commands for shared-RGB pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p2_shared_capture_launch_20261102_exclusive as launch
from . import p2_cp15_capture_builder_20261102_codexexclusive as capture
from . import codex_p2_multistart_joint_physical_20261030_v1 as physical

NAME = "p2_shared_rgb_physical_bridge_20261102_exclusive"
ROOT = launch.ROOT
RUN = ROOT / "results/p2_shared_rgb_pair_wiring_20261102_codexexclusive_RUN_20261102_p2continuation_host01"
MODEL_REPORT_SHA = "d39f8e4f5f292034331b8a60b3fe0682b04d81206fd904200db2295a38e3ef2d"
PREP = ROOT / "results/codex_p2_multistart_joint_physical_20261030_v1_B0_PREP_01.json"
PREP_SHA = "ceb38405762f7631d367a490e55d378c655dfe0463fcc648ee88ebade683a23c"
need = physical.require


def action_and_source_binding(bundle, actions, pair, receipt, events):
    """Authenticate externally first, then match literal prefix, RGB, and actions."""
    need(bundle["base"] == receipt["job_index"] == 0 and len(receipt["rows"]) == 2
         and bundle["source"]["fresh_root"] is False, "fixed known B0")
    need([x["arm"] for x in pair["branches"]] == ["joint_multistart", "prefix_chain"],
         "fixed model arms")
    prefix = np.asarray(actions, dtype="<f4").tobytes()
    need(len(prefix) == 120, "15 action prefix")
    rows = []
    references = []
    for i, (arm, branch, row) in enumerate(zip(
            ("joint_multistart", "prefix_chain"), pair["branches"], receipt["rows"], strict=True)):
        plan, guard = row["plan"], row["feedback_guard"]
        raw = np.asarray(plan["commands"], dtype="<f4")
        encoded = bytes.fromhex(plan["raw_float32_hex"])
        need(row["policy"] == arm and row["initial_exact"] is True
             and row["status"] == "horizon_completed"
             and row["warmup_returned"] == 10 and row["own_attempted"] == row["own_returned"] == 25
             and row["terminated"] is False and row["truncated"] is False,
             "complete prior physical trajectory")
        need(raw.shape == (25, 2) and np.isfinite(raw).all()
             and (np.abs(raw) <= 1).all() and raw.tobytes() == encoded
             and hashlib.sha256(encoded).hexdigest() == plan["raw_float32_sha256"]
             and encoded[:120] == prefix == bytes.fromhex(guard["prefix_raw15_bits"])
             and encoded[120:].hex() == launch.wire.reuse.winner_action_hex(branch),
             "exact prefix and model winner action bytes")
        need(row["root_sha256"] == bundle["source"]["root_sha256"]
             and row["task_id"] == bundle["source"]["task_id"]
             and row["goal_pixel_sha256"] == bundle["goal"]["pixel_sha256"]
             and guard["no_state_restore"] is True
             and guard["actual_observed_context_sha256"] == guard["compile_input_context_sha256"],
             "root, goal and prior guard")
        refs = [physical.normalized_prefix_event(e) for e in events
                if e.get("branch") == i and physical.is_prefix_event(e)]
        need(len(refs) == 106 and physical.digest(refs) == guard["prefix_reference_sha256"],
             "full guarded native/RGB prefix")
        frame_hashes = []
        for step in (5, 10, 15):
            matches = [e for e in events if e["kind"] == "fresh_observation"
                       and e.get("branch") == i and e["receipt"]["phase"] == "plan"
                       and e["receipt"]["step"] == step and e["receipt"]["overlay_index"] == 0]
            need(len(matches) == 1, "unique CP RGB")
            frame_hashes.append(matches[0]["receipt"]["pixel_sha256"])
        need(frame_hashes == [f["pixel_sha256"] for f in bundle["frames"]],
             "actual shared feedback RGB")
        references.append(refs)
        rows.append({"arm": arm, "full25_sha256": hashlib.sha256(encoded).hexdigest(),
                     "winner_suffix_sha256": hashlib.sha256(encoded[120:]).hexdigest(),
                     "status": row["status"], "final_metrics": row["final_metrics"]})
    need(references[0] == references[1], "identical complete guarded pair prefix")
    return rows


def build():
    launch.audit(RUN, MODEL_REPORT_SHA)
    bundle, _, _, actions = launch.wire.load_capture(launch.CAPTURE / "capture.json", launch.CAPTURE_SHA)
    pair = json.loads((RUN / "pair.json").read_bytes())
    need(physical.sha(capture.RECEIPT) == capture.RECEIPT_SHA, "prior physical receipt SHA")
    receipt = physical.consume(capture.RUN, capture.REPORT_SHA, capture.COMPLETION_SHA)
    need(receipt == json.loads(capture.RECEIPT.read_bytes()), "prior receipt reconstruction")
    events = [physical.native.decode(x) for x in (capture.RUN / "journal.jsonl").read_bytes().splitlines()]
    rows = action_and_source_binding(bundle, actions, pair, receipt, events)
    need(physical.sha(PREP) == PREP_SHA, "old physical preparation SHA")
    prepared, _ = physical.verify(PREP, PREP_SHA)
    for plan, row in zip(prepared["plans"], receipt["rows"], strict=True):
        need(plan == row["plan"], "old prepared plan equals executed prior plan")
    return {"schema": NAME + ".contract.v1", "model_report_sha256": MODEL_REPORT_SHA,
            "capture_sha256": launch.CAPTURE_SHA, "physical_prepare": str(PREP),
            "physical_prepare_sha256": PREP_SHA, "prior_report_sha256": capture.REPORT_SHA,
            "prior_completion_sha256": capture.COMPLETION_SHA,
            "bridge_source_sha256": physical.sha(__file__),
            "arms": rows, "known_root": True, "independent_confirmation": False,
            "P1_admission": False, "P2_admission": False,
            "execution": "existing physical runner, exact old prepared action bytes, two fresh env instances",
            "qualification": "new model source linked by identical pixels, actual prefix and winner action bytes; old planner witness remains old"}


def verify_contract(path, sha):
    need(physical.sha(path) == sha, "bridge SHA")
    value = json.loads(Path(path).read_bytes())
    need(value == build() and physical.sha(path) == sha, "bridge reconstruction")
    return value


def audit_output(contract, output, report_sha, completion_sha):
    receipt = physical.consume(output, report_sha, completion_sha)
    bundle, _, _, actions = launch.wire.load_capture(launch.CAPTURE / "capture.json", launch.CAPTURE_SHA)
    pair = json.loads((RUN / "pair.json").read_bytes())
    events = [physical.native.decode(x) for x in (Path(output) / "journal.jsonl").read_bytes().splitlines()]
    rows = action_and_source_binding(bundle, actions, pair, receipt, events)
    need([r["full25_sha256"] for r in rows] == [r["full25_sha256"] for r in contract["arms"]],
         "new physical commands exactly contract")
    return {"schema": NAME + ".audit.v1", "status": "verified_known_root_physical_not_admitted",
            "physical_report_sha256": report_sha, "completion_sha256": completion_sha,
            "model_report_sha256": MODEL_REPORT_SHA, "capture_sha256": launch.CAPTURE_SHA,
            "arms": rows, "independent_confirmation": False,
            "P1_admission": False, "P2_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    for command in ("check", "run", "audit"):
        p = sub.add_parser(command)
        p.add_argument("--contract", required=True)
        p.add_argument("--contract-sha256", required=True)
        if command in ("run", "audit"):
            p.add_argument("--output", required=True)
        if command == "audit":
            p.add_argument("--report-sha256", required=True)
            p.add_argument("--completion-sha256", required=True)
        if command == "run":
            p.add_argument("--authorize-known-root-physics", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        path = Path(args.output).absolute()
        need(path.parent == ROOT / "results" and path.name.startswith(NAME + "_CONTRACT_"), "contract location")
        print(json.dumps({"path": str(path), "sha256": launch.write_new(path, build())}))
        return
    contract = verify_contract(args.contract, args.contract_sha256)
    if args.command == "check":
        print(json.dumps({"status": "verified_no_execution"}))
        return
    output = Path(args.output).absolute()
    if args.command == "run":
        need(output.parent == ROOT / "results" and output.name.startswith(physical.NAME + "_")
             and not os.path.lexists(output), "exclusive physical output")
        with launch.wire.store_api.locking.gpu_lock(launch.wire.LOCK):
            verify_contract(args.contract, args.contract_sha256)
            def factory():
                import gymnasium as gym
                import stable_worldmodel.envs
                return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                                with_target=False, render_action=False)
            result = physical.run(PREP, PREP_SHA, output, factory, authorization=physical.AUTH)
        report_sha, completion_sha = physical.sha(output / "report.json"), physical.sha(output / "completion.json")
        print(json.dumps({"status": result["status"], "output": str(output),
                          "report_sha256": report_sha, "completion_sha256": completion_sha}))
        if result["status"] != physical.RECORDED:
            raise SystemExit(1)
    else:
        print(json.dumps(audit_output(contract, output, args.report_sha256, args.completion_sha256),
                         sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
