"""Presealed, CUDA-only, model-only two-arm dev303 V13-kernel replan.

This is a development-only variant. Both arms use the same zero initial mean,
seed bank, standard deviation, and random seed; V13 RGB-derived per-arm contact
feasibility remains. No physical action is authorized by this module.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_cp5_rgb_action_export_20261102_exclusive as export
from . import p1_dev303_v12_adapter_20261101_exclusive as task_source
from . import p1_dev302_v13_audited_tail_replan_20261101_codex as v13
from . import p1_dev302_v5_contact_tail_replan_20261101_codex as engine

NAME = "p1_dev303_cp5_equal_v13_replan_20261102_exclusive"
ROOT = export.ROOT
EXPORT_DIR = ROOT / "results/p1_dev303_cp5_rgb_action_export_20261102_exclusive_EXPORT_DEV01"
MANIFEST_SHA = "61d35aa6530af39af60087407fd655919ecee3bbda79a3055261ea8b14c124e7"
ARCHIVE_SHA = "29a1ec3a7fecacb19076b16947ce8753f17a7237861cae8ef1c1d3503aa0eea1"
TASK_PREP = ROOT / "results/p1_dev303_v12_adapter_20261101_exclusive_PREP_MAIN01.json"
TASK_PREP_SHA = "f21eb1abe6517bf272f58e3a96af2291f829a4ebf6379add6c02d0fa09e2f1e0"
ARMS = ("rgb_chunk", "zero_prefix")
SEED = engine.SEED


def require(value, message):
    if not value:
        raise ValueError(message)


def read_fixed(path, expected):
    raw = export.read_regular(path)
    require(export.sha(raw) == expected, f"fixed input SHA: {path}")
    return raw


def input_arrays():
    manifest = export.decode(read_fixed(EXPORT_DIR / "manifest.json", MANIFEST_SHA))
    archive = read_fixed(EXPORT_DIR / "rgb_action_history.npz", ARCHIVE_SHA)
    require(manifest["schema"] == export.NAME + ".manifest.v1"
            and manifest["archive_sha256"] == ARCHIVE_SHA
            and manifest["source_report_sha256"] == export.REPORT_SHA
            and manifest["source_receipt_sha256"] == task_source.RECEIPT_SHA
            and manifest["P1_admission"] is False
            and manifest["independent_confirmation"] is False
            and manifest["no_arm_selection"] is True
            and [r["arm"] for r in manifest["roles"]] == list(ARMS),
            "sealed two-arm export")
    with np.load(io.BytesIO(archive), allow_pickle=False) as contents:
        require(set(contents.files) == {"goal_rgb", *(arm + "_rgb_history" for arm in ARMS),
                                      *(arm + "_raw_action_history" for arm in ARMS)},
                "exact archive roster")
        arrays = {key: contents[key].copy() for key in contents.files}
    source_arrays, receipt, audit = task_source.source.source()
    require(receipt == task_source.RECEIPT_SHA and audit["seed"] == export.run.SEED
            and audit["P1_admission"] is False, "development root source")
    require(arrays["goal_rgb"].dtype == np.uint8
            and np.array_equal(arrays["goal_rgb"], source_arrays["goal_rgb"])
            and export.sha(arrays["goal_rgb"].tobytes()) == manifest["source_goal_rgb_pixel_sha256"],
            "goal RGB source binding")
    for arm, role in zip(ARMS, manifest["roles"]):
        images, actions = arrays[arm + "_rgb_history"], arrays[arm + "_raw_action_history"]
        require(images.shape == (3, 224, 224, 3) and images.dtype == np.uint8
                and np.array_equal(images[:2], source_arrays["rgb_history"][-2:])
                and export.sha(images[-1].tobytes()) == role["cp5_pixel_sha256"],
                f"{arm}: RGB history")
        require(actions.shape == (10, 2) and actions.dtype == np.dtype("<f4")
                and np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
                and not actions[:5].any()
                and export.sha(actions.tobytes()) == role["action_history_sha256"],
                f"{arm}: actual action history")
        cp5 = export.RUN_DIR / role["cp5_file"]
        require(Path(role["cp5_file"]).name == role["cp5_file"]
                and export.sha(export.read_regular(cp5)) == role["cp5_file_sha256"],
                f"{arm}: cp5 image source")
        require(np.array_equal(export._frame(export.RUN_DIR, {
            "file": role["cp5_file"], "file_sha256": role["cp5_file_sha256"],
            "pixel_sha256": role["cp5_pixel_sha256"],
        }), images[-1]), f"{arm}: exported cp5 image")
    return manifest, arrays


def protocol():
    manifest, arrays = input_arrays()
    raw_task = read_fixed(task_source.TASK, task_source.TASK_SHA)
    task_export = export.decode(gzip.decompress(raw_task))
    prep = export.decode(read_fixed(TASK_PREP, TASK_PREP_SHA))
    require(task_export["development_seed"] == export.run.SEED
            and task_export["P1_admission"] is False
            and task_export["new_independent_confirmation"] is False
            and task_export["blind_source_sha256"] == task_source.BLIND_SHA
            and prep["task"] == task_export["task"]
            and prep["source_receipt_sha256"] == task_source.RECEIPT_SHA
            and prep["pixel_sha256"] == [
                export.sha(arrays["rgb_chunk_rgb_history"][1].tobytes()),
                export.sha(arrays["goal_rgb"].tobytes())]
            and prep["model_sha256"] == engine.base.MODEL_SHA256
            and prep["P1_admission"] is False, "frozen task/model/goal binding")
    task = prep["task"]
    require(task["root_raw_action_history"] == np.zeros((10, 2), np.float32).tolist()
            and len(task["root_context"]["latent_history"]) >= 2
            and len(task["root_context"]["action_history"]) >= 1
            and np.asarray(task["action_mean"]).shape == (2,)
            and np.asarray(task["action_std"]).shape == (2,)
            and np.isfinite(np.asarray(task["action_std"])).all()
            and (np.asarray(task["action_std"]) > 0).all(), "frozen task history/normalization")
    action_mean = np.asarray(task["action_mean"], dtype=np.float64)
    action_std = np.asarray(task["action_std"], dtype=np.float64)
    previous_packet = np.asarray(task["root_context"]["action_history"][-1], dtype=np.float64)
    latents = np.asarray(task["root_context"]["latent_history"], dtype=np.float64)
    goal = np.asarray(task["goal"], dtype=np.float64)
    require(np.isfinite(action_mean).all()
            and previous_packet.shape == (10,)
            and np.allclose(previous_packet, np.tile(-action_mean / action_std, 5),
                            atol=1e-6, rtol=0.0)
            and latents.ndim == 2 and latents.shape[1] == task["latent_dim"]
            and goal.shape == (task["latent_dim"],)
            and np.isfinite(latents).all() and np.isfinite(goal).all(),
            "frozen latent goal and actual warmup packet")
    zero_tail = np.zeros((2, 5, 2), dtype=np.float32).tolist()
    roles = []
    for arm, source_row in zip(ARMS, manifest["roles"]):
        history = arrays[arm + "_raw_action_history"]
        cp5 = export.RUN_DIR / source_row["cp5_file"]
        geometry = engine._geometry(arrays[arm + "_rgb_history"][-1])
        roles.append({
            "role": arm, "candidate_index": list(ARMS).index(arm),
            "candidate_label": arm, "actual_prefix": history[5:].tolist(),
            "actual_prefix_sha256": engine.base.action_sha(history[5:]),
            "cp5_png": str(cp5), "cp5_pixel_sha256": source_row["cp5_pixel_sha256"],
            "rgb_contact_geometry": geometry, "contact_seed_tail": zero_tail,
        })
    files = [Path(__file__), Path(export.__file__), Path(v13.__file__), Path(engine.__file__),
             Path(engine.base.__file__), Path(engine.base.common.__file__),
             Path(engine.base.p0.__file__), Path(engine.contact.__file__),
             Path(task_source.__file__), TASK_PREP, task_source.TASK,
             EXPORT_DIR / "manifest.json", EXPORT_DIR / "rgb_action_history.npz"]
    files.extend(export.RUN_DIR / row["cp5_file"] for row in manifest["roles"])
    bindings = {str(path.resolve()): export.sha(export.read_regular(path)) for path in files}
    return {
        "schema": NAME + ".protocol.v1", "roles": roles, "task": task,
        "policy": prep["policy"], "model_sha256": engine.base.MODEL_SHA256,
        "task_sha256": task_source.TASK_SHA, "source_manifest_sha256": MANIFEST_SHA,
        "source_archive_sha256": ARCHIVE_SHA, "source_report_sha256": export.REPORT_SHA,
        "seed_tails": [zero_tail], "initial_mean": zero_tail, "random_seed": SEED,
        "algorithm": {
            "kernel": "V13 V5 contact-tail CEM; common zero initial mean and seed",
            "iterations": engine.ITERATIONS, "samples": engine.SAMPLES,
            "elites": engine.ELITES, "variants": engine.VARIANTS,
            "horizon_packets": engine.HORIZON, "initial_std": engine.INITIAL_STD,
            "minimum_projection": engine.MIN_PROJECTION,
            "common_random_initialization": True,
            "per_arm_rgb_feasibility_filter": True,
        },
        "per_arm_budget": {"rgb_encode_calls": 1, "action_encoder_calls": 10,
                           "predict_calls": 10, "predict_samples": 3200},
        "post_cp5_physical_outcomes_used_for_selection": False,
        "simulator_state_used_for_selection": False,
        "physical_execution": False, "physical_actions_authorized": False,
        "development_only": True, "independent_confirmation": False,
        "P1_admission": False, "bindings": bindings,
    }


@contextmanager
def _kernel_protocol():
    original_name, original_protocol = engine.NAME, engine.protocol
    engine.NAME, engine.protocol = NAME, protocol
    try:
        yield
    finally:
        engine.NAME, engine.protocol = original_name, original_protocol


def run_cuda(prepared, prepared_sha, output):
    """Run the existing V13 execution kernel; it requires CUDA and frozen model."""
    with _kernel_protocol():
        report_sha = engine.run(Path(prepared), prepared_sha, Path(output))
    report_path = Path(output) / "report.json"
    report = export.decode(read_fixed(report_path, report_sha))
    expected = protocol()
    require(report["schema"] == NAME + ".report.v1"
            and report["protocol_sha256"] == prepared_sha
            and report["status"] == "model_only_complete_not_admitted"
            and report["P1_admission"] is False and report["physical_execution"] is False
            and report["post_cp5_physical_outcomes_used"] is False
            and report["model_before"] == report["model_after"]
            and report["model_before"]["original_state_dict"] == expected["model_sha256"]
            and report["attempts"] == {"rgb_encode": 2, "action_encoder": 20, "predict": 20}
            and [row["role"] for row in report["rows"]] == list(ARMS),
            "complete equal-budget frozen-model report")
    for actual, bound in zip(report["rows"], expected["roles"]):
        require(actual["status"] == "replanned"
                and actual["actual_prefix_sha256"] == bound["actual_prefix_sha256"]
                and actual["cp5_pixel_sha256"] == bound["cp5_pixel_sha256"]
                and len(actual["iterations"]) == engine.ITERATIONS,
                "role output identity")
    audit = {"schema": NAME + ".audit.v1", "report_sha256": report_sha,
             "prepared_sha256": prepared_sha, "physical_actions_authorized": False,
             "P1_admission": False, "independent_confirmation": False}
    engine.write_new(Path(output) / "audit.json", audit)
    return report_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prepare-sha256")
    parser.add_argument("--authorize-model-cuda", action="store_true")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "ASCII alphanumeric tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.prepare_sha256 and not args.authorize_model_cuda
                and not os.path.lexists(output), "prepare only")
        digest = engine.write_new(prepared, protocol())
        print(json.dumps({"prepared": str(prepared), "sha256": digest}, sort_keys=True))
    else:
        require(args.authorize_model_cuda and args.prepare_sha256, "CUDA authorization and SHA")
        with engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            digest = run_cuda(prepared, args.prepare_sha256, output)
        print(json.dumps({"output": str(output), "report_sha256": digest,
                          "physical_actions_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
