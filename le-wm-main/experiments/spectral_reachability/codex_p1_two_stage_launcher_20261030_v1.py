"""Known-root two-stage feedback exploration, fixed 24 rows, no training.

Prepare fixes the roster and code hashes before execution. The GPU lock is
advisory; source hashes and a planner input whitelist are not OS isolation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from . import codex_p1_cp5_physical_24_20261030_v1 as physical
from . import codex_p1_two_stage_physical_episode_20261030_v1 as episode
from . import codex_p1_two_stage_metered_plan_20261030_v1 as stage
from . import codex_p1_two_stage_journal_feedback_20261030_v1 as feedback_journal

h10, base, need, sha = physical.h10, stage.base, physical.need, physical.sha
ROOT = h10.ROOT
NAME = "codex_p1_two_stage_launcher_20261030_v1"
WALL = 7200
GPU_BYTES = 6 * 1024**3


def build():
    p = h10.design.verify(h10.INPUT, h10.INPUT_SHA)
    need(base.sha(base.BUNDLE) == base.BUNDLE_SHA, "fixed blind bundle")
    rows = json.loads(base.BUNDLE.read_bytes())["rows"]
    anchors = p["anchors"]
    need(len(rows) == len(anchors) == 24, "full fixed roster")
    roster = []
    for i, (anchor, row) in enumerate(zip(anchors, rows, strict=True)):
        episode.feedback.common.blind.validate_row(row)
        need(anchor["index"] == row["index"] == i and anchor["role"] == row["role"]
             and anchor["logical_pair"] == row["logical_pair"]
             and anchor["prefix5_float32_hex"] == row["prefix_raw_f32_hex"]
             and anchor["checkpoint"]["pixel_sha256"] == row["cp5_pixel_sha256"], "roster binding")
        roster.append({k: anchor[k] for k in
                      ("index", "logical_pair", "root_slot", "condition", "arm", "role", "task_id",
                       "prefix5_float32_hex", "physical_root_sha256", "checkpoint")})
    sources = dict(p["bindings"])
    sources.update(base.bindings())
    for path in (Path(__file__), Path(episode.__file__), Path(stage.__file__),
                 Path(feedback_journal.__file__), Path(episode.feedback.__file__),
                 Path(__file__).with_name("codex_p1_two_stage_saved_20261030_v1.py"),
                 ROOT/"tests/test_codex_p1_two_stage_launcher_20261030_v1.py",
                 ROOT/"tests/test_codex_p1_two_stage_feedback_20261030_v1.py",
                 ROOT/"tests/test_codex_p1_two_stage_metered_plan_20261030_v1.py",
                 ROOT/"tests/test_codex_p1_two_stage_journal_feedback_20261030_v1.py", h10.INPUT, base.BUNDLE):
        sources[str(path.resolve())] = sha(path)
    need(all(sha(path) == digest for path, digest in sources.items()), "source drift")
    return {"schema": NAME+".prepare", "roster": roster, "sources": sources, "rows": 24,
            "optimizer": episode.feedback.CONFIG, "stage_caps": stage.CAPS, "encoding_caps": stage.ENC_CAPS,
            "physical_caps_per_row": physical.CAPS, "wall_seconds": WALL, "cuda_reserved_cap": GPU_BYTES,
            "lock": str(base.LOCK), "model": base.MODEL_HASH, "known_roots": True,
            "no_training": True, "os_isolation_certified": False, "dependency_closure_complete": False,
            "cooperative_lock_not_global": True, "equal_budget_certified": False, "P1_admission": False}


def prepare(output):
    path = Path(output)
    need(path.is_absolute() and path.parent == ROOT/"results" and path.name.startswith(NAME+"_prepare_")
         and not os.path.lexists(path), "exclusive prepare")
    return h10._write_new(path, build())


def verify(path, expected):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path) == expected, "prepare SHA")
    p = h10.native.read(path, cap=64*1024**2)
    need(p == build() and sha(path) == expected, "prepare/source drift")
    return p


def execute_row(anchor, blind, data, folder, model, transform, device, sources, factory, started):
    folder.mkdir(mode=0o700)
    store = h10.ObservedStore(h10.Store(folder/"physical"))
    row = h10.runner.row_for(anchor, {"index": 0, "name": "two_stage_independent_feedback"})
    meter, errors, env = physical.Meter(store), [], None
    meter.where = dict(branch=0, anchor_index=anchor["index"], condition=row["condition"],
                       arm=row["arm"], candidate_role=row["candidate_role"], probe=row["probe"])
    store.emit("run_started", anchor=anchor["index"], fixed_rows=1, caps=physical.CAPS)
    previous_latent = None
    def planner(number, png, executed_packet):
        nonlocal previous_latent
        import torch
        need(time.monotonic()-started < WALL and torch.cuda.memory_reserved(device) <= GPU_BYTES,
             "global wall/CUDA cap")
        latent, result = stage.run_stage(folder/f"stage_{number}", blind, number, png, previous_latent,
                                         executed_packet, model, transform, device, sources)
        if number == 1:
            previous_latent = latent
        return {**result, "stage_report_sha256": sha(folder/f"stage_{number}/report.json")}
    try:
        env = meter.call("constructor", factory)
        h10.base.validate_env(env.unwrapped)
        need(env.unwrapped.render_action is False and env.unwrapped.render_size == 224
             and env.unwrapped.render_mode == "rgb_array", "renderer")
        with h10.base.native_hooks(env, meter, row["secondary_errors"]):
            episode.episode(env, data["roots"][row["root_slot"]], data["source_plan"],
                            anchor, row, meter, store, planner)
        need(not row["secondary_errors"], "hook cleanup")
    except BaseException as exc:
        row.update(status="engineering_failed", error=h10.base.error("branch", exc))
        errors.append(row["error"])
    finally:
        if env is not None:
            try:
                meter.call("close", env.close, cleanup=True)
            except BaseException as exc:
                errors.append(h10.base.error("close", exc))
    store.emit("branch_closed", branch=0, status=row["status"])
    valid = not errors and row["status"] in ("probe_completed", "native_stopped")
    status = "recorded_not_consumed" if valid else "partial_or_failed"
    store.emit("run_finished", status=status, counts=meter.counts, errors=errors)
    report = {"schema": NAME+".row", "status": status, "row": row, "counts": meter.counts,
              "receipts": meter.receipts, "errors": errors, "journal_chain_tail": store.previous,
              "journal_sha256": sha(store.path/"journal.jsonl"), "P1_admission": False}
    store.put("report.json", report, final=True)
    return report


def run(prepared, expected, output):
    p = verify(prepared, expected)
    output = Path(output)
    need(output.is_absolute() and output.parent == ROOT/"results" and output.name.startswith(NAME+"_run_")
         and not os.path.lexists(output) and shutil.disk_usage(output.parent).free >= 2*1024**3,
         "exclusive run and disk")
    source_report = h10.native.read(h10.old.prep.DIRECTORY/"report.json", cap=64*1024**2)
    _, data = h10.runner.physical.verify(source_report["execution_binding"]["prepared"], h10.old.prep.PREP_SHA)
    h10.native.constructor_contract()
    anchors = h10.native.read(h10.INPUT, cap=64*1024**2)["anchors"]
    rows = json.loads(base.BUNDLE.read_bytes())["rows"]
    with base.locking.gpu_lock(base.LOCK):
        output.mkdir(mode=0o700)
        base.write(output/"binding.json", {"prepare_sha256": expected, "rows": 24})
        started, shards, failure, before, after, host = time.monotonic(), [], None, None, None, None
        try:
            os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
            import torch
            from torch.nn.attention import SDPBackend, sdpa_kernel
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from . import preflight_pusht_d2p_p0 as p0
            host = require_cuda_allocation("cuda:0")
            device = torch.device("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
            before = base.model_hash(model)
            need(before == base.MODEL_HASH, "frozen checkpoint")
            transform = p0.image_transform()
            def factory():
                import gymnasium as gym
                import stable_worldmodel  # noqa: F401
                return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
            with torch.autocast(device_type="cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
                for i, (anchor, blind) in enumerate(zip(anchors, rows, strict=True)):
                    need(anchor["index"] == blind["index"] == i and time.monotonic()-started < WALL,
                         "row order/global wall")
                    folder = output/f"row_{i:02d}"
                    report = execute_row(anchor, blind, data, folder, model, transform, device, p["sources"],
                                         factory, started)
                    shards.append({"index": i, "status": report["status"],
                                   "report_sha256": sha(folder/"physical/report.json")})
                    if report["status"] != "recorded_not_consumed":
                        raise ValueError(f"row {i} failed; retained report")
            after = base.model_hash(model)
            need(before == after and all(sha(k) == v for k, v in p["sources"].items()), "model/source drift")
        except BaseException as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)[:1500]}
        report = {"schema": NAME+".run", "status": "recorded_not_consumed" if len(shards) == 24 and not failure
                  else "partial_or_failed", "prepare_sha256": expected, "shards": shards,
                  "fixed_rows": 24, "not_started_indices": list(range(len(shards), 24)), "failure": failure,
                  "model_before": before, "model_after": after, "host": host,
                  "wall_seconds": time.monotonic()-started, "P1_admission": False,
                  "independent_confirmation": False, "os_isolation_certified": False}
        digest = base.write(output/"run_report.json", report)
        print(json.dumps({"status": report["status"], "sha256": digest, "rows": len(shards),
                          "failure": failure}), flush=True)
        return 0 if report["status"] == "recorded_not_consumed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    a = parser.parse_args()
    if a.command == "prepare":
        print(json.dumps({"sha256": prepare(a.output)}))
        return 0
    need(a.prepared and a.sha256, "prepared contract required")
    return run(a.prepared, a.sha256, a.output)


if __name__ == "__main__":
    raise SystemExit(main())
