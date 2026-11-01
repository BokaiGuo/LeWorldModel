"""CPU preparation and explicitly authorized CUDA launch for one known-root pair."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat

import numpy as np

from . import p2_cp15_capture_builder_20261102_codexexclusive as builder
from . import p2_shared_rgb_pair_wiring_20261102_codexexclusive as wire
from . import p2_shared_pair_saved_20261102_codexexclusive as saved

NAME = "p2_shared_capture_launch_20261102_exclusive"
ROOT = wire.ROOT
RESULTS = ROOT / "results"
CAPTURE = RESULTS / "p2_cp15_capture_builder_20261102_codexexclusive_RUN_20261102_p2continuation"
CAPTURE_SHA = "0a8c4c8ff7798fa082a168f3b770e75238ce556627cf4ba56a058c305f331706"
need = wire.need


def source_bindings():
    directory = ROOT / "experiments/spectral_reachability"
    names = (
        NAME, builder.NAME, wire.NAME, saved.NAME,
        "p2_matched_program_kernel_20261101_parallel",
        "p2_matched_program_saved_20261101_parallel",
        "p2_matched_program_physical_reuse_20261101_parallel",
        "p2_matched_program_host_20261101_parallel",
        "codex_p2_program_library_pair_20261030_v1",
        "codex_p2_multistart_joint_cuda_20261030_v1",
        "codex_p2_prefix_recompile_cuda_20261030_v1",
        "codex_p2_program_model_scope_20261030_v1",
        "p1_constructive_blind_encode_cuda_20261027_v1",
        "p1_dev301_blind_plan_physical_20261101_codex",
        "preflight_pusht_d2p_p0",
        "e3_p2_cooperative_gpu_lock_codex_20261030_v1",
    )
    paths = [directory / (name + ".py") for name in names]
    paths += [ROOT / "experiments/checkpoint_compat.py", ROOT / "experiments/gpu_preflight.py"]
    return {str(p): wire.checkpoint.sha(p) for p in paths}


def build():
    source = builder.verify(CAPTURE, CAPTURE_SHA)
    bundle, frames, goal, actions = wire.load_capture(CAPTURE / "capture.json", CAPTURE_SHA)
    metadata = wire.checkpoint.native_metadata()
    need(source["fresh_root"] is False and bundle["base"] == 0, "known root only")
    lock = wire.LOCK
    lockstat = lock.lstat()
    need(stat.S_ISREG(lockstat.st_mode) and lockstat.st_nlink == 1
         and lock.parent == lock.parent.resolve(), "stable cooperative lock file")
    need(shutil.disk_usage(RESULTS).free >= wire.store_api.FLOOR + wire.store_api.CAP,
         "existing output reserve")
    return {
        "schema": NAME + ".contract.v1",
        "capture": str(CAPTURE / "capture.json"), "capture_sha256": CAPTURE_SHA,
        "source_verification": source,
        "rgb_pixel_sha256": [wire.pixel_sha(x) for x in [*frames, goal]],
        "action_sha256": wire.hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest(),
        "normalization": metadata, "model_files": dict(wire.checkpoint.MODELS),
        "model_tensor_sha256": wire.checkpoint.MODEL_TENSOR_SHA,
        "kernel_contract": wire.kernel.contract(), "source_bindings": source_bindings(),
        "execution": {"shared_encodes": 4, "arm_order": list(wire.kernel.ARMS),
                      "physical_actions": 0, "lock": str(lock), "automatic_retry": False,
                      "cuda_authorization_required": True},
        "development_only": True, "fresh_root": False, "independent_confirmation": False,
        "equal_total_compute": False, "P1_admission": False, "P2_admission": False,
    }


def write_new(path, value):
    path = Path(path).absolute()
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    need(not os.path.lexists(path), "exclusive output")
    with path.open("xb") as stream:
        need(stream.write(raw) == len(raw), "complete write")
        stream.flush()
        os.fsync(stream.fileno())
    return wire.checkpoint.sha(path)


def verify_contract(path, expected):
    path = Path(path).absolute()
    need(wire.checkpoint.sha(path) == expected, "contract SHA")
    contract = json.loads(path.read_bytes())
    need(contract == build(), "contract/source drift")
    need(wire.checkpoint.sha(path) == expected, "contract drift")
    return contract


def verify_branch_events(pair, events, base):
    records = [event["record"] for event in events if event.get("kind") == "branch"]
    need([b["arm"] for b in pair["branches"]] == list(wire.kernel.ARMS), "arm order")
    need(all(row.get("base") == base and row.get("arm") in wire.kernel.ARMS
             for row in records), "branch event routing")
    for arm, branch in zip(wire.kernel.ARMS, pair["branches"], strict=True):
        scoped = [row for row in records if row["arm"] == arm]
        wire.saved.validate_branch(
            {"base": base, "context": pair["context"], "goal": pair["goal"]},
            branch, scoped,
        )


def audit(run_dir, report_sha):
    source = builder.verify(CAPTURE, CAPTURE_SHA)
    bundle, _, _, actions = wire.load_capture(CAPTURE / "capture.json", CAPTURE_SHA)
    run_dir = Path(run_dir).absolute()
    report = saved.verify_report(run_dir / "report.json", report_sha, CAPTURE_SHA)
    pair_path = run_dir / "pair.json"
    need(saved.sha(pair_path) == report["pair_sha256"], "pair SHA")
    pair = json.loads(pair_path.read_bytes())
    need(pair["schema"] == wire.NAME + ".pair.v1"
         and pair["P1_admission"] is False and pair["P2_admission"] is False
         and pair["independent_confirmation"] is False and pair["equal_total_compute"] is False
         and pair["source_authentication"] == "capture_bundle_producer_not_authenticated"
         and pair["shared_setup"] == {"encode_attempted": 4, "encode_returned": 4, "encode_failed": 0}
         and pair["resource_axes"] == wire.kernel.CAPS, "pair boundary")
    rows = saved._read_journal(run_dir / "journal.jsonl", report["journal_sha256"], report["journal_tail"])
    need(rows[0]["kind"] == "run_started" and rows[-1]["kind"] == "run_finished"
         and rows[-1].get("status") == "completed"
         and rows[-1].get("pair_sha256") == report["pair_sha256"], "journal bookends")
    events = [r["event"] for r in rows if r["kind"] == "pair_event"]
    attempts = [e for e in events if e["kind"] == "shared_encode_attempt"]
    returns = [e for e in events if e["kind"] == "shared_encode_return"]
    roles = ["control5", "control10", "control15", "goal"]
    pixels = [r["pixel_sha256"] for r in bundle["frames"]] + [bundle["goal"]["pixel_sha256"]]
    latents = [*pair["context"]["latent_history"], pair["goal"]]
    latent_hashes = [wire.hashlib.sha256(np.asarray(z, dtype="<f4").tobytes()).hexdigest() for z in latents]
    need([e["role"] for e in attempts] == roles == [e["role"] for e in returns]
         and [e["pixel_sha256"] for e in attempts] == pixels == [e["pixel_sha256"] for e in returns]
         and [e["latent_float32_sha256"] for e in returns] == latent_hashes, "shared encode receipts")
    shared = [e for e in events if e["kind"] == "shared_context"]
    complete = [e for e in events if e["kind"] == "pair_complete"]
    need(len(shared) == len(complete) == 1
         and shared[0]["context_sha256"] == complete[0]["context_sha256"] == wire.kernel.old.digest(pair["context"])
         and shared[0]["goal_sha256"] == wire.kernel.old.digest(pair["goal"])
         and shared[0]["raw_history_float32_sha256"] == wire.hashlib.sha256(actions[-10:].astype("<f4").tobytes()).hexdigest(),
         "shared context receipt")
    metadata = wire.checkpoint.native_metadata()
    expected_history = ((actions[-10:] - np.asarray(metadata["mean"], np.float32))
                        / np.asarray(metadata["std"], np.float32)).reshape(2, 10)
    need(np.array_equal(expected_history, np.asarray(pair["context"]["action_history"], np.float32)),
         "normalized actual actions")
    verify_branch_events(pair, events, bundle["base"])
    winners = {b["arm"]: wire.hashlib.sha256(bytes.fromhex(wire.reuse.winner_action_hex(b))).hexdigest()
               for b in pair["branches"]}
    need(complete[0]["winner_action_sha256"] == winners, "winner receipt")
    return {"status": "verified_saved_pair_known_root_not_admitted", "capture_sha256": CAPTURE_SHA,
            "report_sha256": report_sha, "source_scope": source["status"],
            "physical_execution": False, "independent_confirmation": False,
            "P1_admission": False, "P2_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    check = sub.add_parser("check")
    check.add_argument("--contract", required=True)
    check.add_argument("--contract-sha256", required=True)
    run = sub.add_parser("run")
    run.add_argument("--contract", required=True)
    run.add_argument("--contract-sha256", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--authorize-host-cuda", action="store_true", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--output", required=True)
    audit_parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        out = Path(args.output).absolute()
        need(out.parent == RESULTS and out.name.startswith(NAME + "_CONTRACT_"), "exclusive contract path")
        result = {"contract": str(out), "sha256": write_new(out, build())}
    elif args.command == "check":
        verify_contract(args.contract, args.contract_sha256)
        result = {"status": "verified_cpu_no_cuda", "contract_sha256": args.contract_sha256}
    elif args.command == "audit":
        result = audit(args.output, args.report_sha256)
    else:
        output = Path(args.output).absolute()
        need(output.parent == RESULTS and output.name.startswith(wire.NAME + "_RUN_")
             and not os.path.lexists(output), "exclusive CUDA output")
        verify_contract(args.contract, args.contract_sha256)
        with wire.store_api.locking.gpu_lock(wire.LOCK):
            verify_contract(args.contract, args.contract_sha256)
            report, digest = wire.execute_cuda(CAPTURE / "capture.json", CAPTURE_SHA, output)
            verify_contract(args.contract, args.contract_sha256)
        result = {"status": report["status"], "report_sha256": digest,
                  "output": str(output), "contract_sha256": args.contract_sha256}
        print(json.dumps(result, sort_keys=True))
        if report["status"] != "completed":
            raise SystemExit(1)
        return
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
