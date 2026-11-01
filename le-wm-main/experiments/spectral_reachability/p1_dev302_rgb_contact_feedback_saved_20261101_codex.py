"""Strict saved-only verification of the dev302 RGB contact diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_dev302_rgb_contact_feedback_20261101_codex as source

NAME = "p1_dev302_rgb_contact_feedback_saved_20261101_codex"
ROOT = source.ROOT
require, sha = source.require, source.sha


def verify(prepared: Path, prepared_sha: str, directory: Path, report_sha: str) -> dict:
    prepared = prepared.resolve(strict=True)
    directory = directory.resolve(strict=True)
    require(sha(prepared) == prepared_sha, "prepared SHA")
    contract = json.loads(prepared.read_bytes())
    require(contract == source.build(), "prepared reconstruction")
    report_path = directory / "report.json"
    require(sha(report_path) == report_sha, "report SHA")
    report = json.loads(report_path.read_bytes())
    require(
        report["schema"] == source.NAME + ".report.v1"
        and report["status"] == "recorded_development_diagnosis"
        and report["prepare_sha256"] == prepared_sha
        and not report["errors"]
        and len(report["rows"]) == 3
        and report["development_only"] is True
        and report["fair_total_cost"] is False
        and report["independent_confirmation"] is False
        and report["P1_admission"] is False,
        "report contract",
    )
    binding = json.loads((directory / "execution_binding.json").read_bytes())
    require(
        binding == {"prepared": str(prepared), "prepare_sha256": prepared_sha},
        "execution binding",
    )
    require(sha(directory / "journal.jsonl") == report["journal_sha256"], "journal SHA")
    rows = []
    common_tail = np.asarray(contract["common_tail"], np.float32)
    sealed = np.asarray(contract["sealed_d2p_actions"], np.float32)
    for row, role in zip(report["rows"], source.ROLES, strict=True):
        require(
            row["role"] == role
            and row["status"] in ("horizon_completed", "native_success", "native_stopped")
            and row["error"] is None
            and row["warmup_returned"] == 10
            and row["plan_returned"] == len(row["transitions"])
            and 1 <= row["plan_returned"] <= 10,
            "row identity/status",
        )
        shard = json.loads((directory / f"{role}.row.json").read_bytes())
        require(shard == row, "row shard")
        actions = np.asarray([item["action"] for item in row["transitions"]], np.float32)
        require(
            hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest()
            == row["executed_actions_sha256"],
            "executed action hash",
        )
        if row["plan_returned"] == 10:
            if role == "sealed_d2p":
                require(np.array_equal(actions, sealed), "sealed D2P actions")
            elif role == "zero_then_d2p_tail":
                require(
                    np.array_equal(actions[:5], np.zeros((5, 2), np.float32))
                    and np.array_equal(actions[5:], common_tail),
                    "zero/common tail actions",
                )
            else:
                require(
                    len(row["rgb_decisions"]) == 5
                    and np.array_equal(actions[5:], common_tail),
                    "RGB/common tail actions",
                )
                for decision, action in zip(row["rgb_decisions"], actions[:5], strict=True):
                    require(
                        decision["simulator_state_used"] is False
                        and np.array_equal(np.asarray(decision["raw_action"], np.float32), action),
                        "RGB decision/action",
                    )
        for point in row["observations"]:
            path = directory / point["file"]
            require(path.resolve(strict=True).is_relative_to(directory), "frame scope")
            require(sha(path) == point["file_sha256"], "frame SHA")
            with Image.open(path) as handle:
                pixels = np.asarray(handle.convert("RGB"), np.uint8)
            require(
                source.source.base.pixel_sha(pixels) == point["pixel_sha256"],
                "frame pixel SHA",
            )
        rows.append(
            {
                "role": role,
                "status": row["status"],
                "native_success": bool(row["final_metrics"]["native_success"]),
                "object_position_error": row["final_metrics"]["object_position_error"],
                "contact_steps": [
                    item["step"] for item in row["transitions"] if item["n_contacts"] > 0
                ],
                "prefix_pusher_position_error": row["prefix_metrics"]["pusher_position_error"],
            }
        )
    require(
        all(sha(Path(path)) == digest for path, digest in contract["bindings"].items()),
        "source closure",
    )
    rgb_row = rows[0]
    return {
        "schema": NAME + ".result.v1",
        "status": "saved_development_diagnosis_verified",
        "prepared": str(prepared),
        "prepare_sha256": prepared_sha,
        "report": str(report_path),
        "report_sha256": report_sha,
        "rows": rows,
        "interpretation": {
            "compact_moving_sprite_fragmentation_was_code_protocol_issue": True,
            "rgb_prefix_moved_pusher_toward_object": True,
            "rgb_prefix_reached_native_contact": bool(rgb_row["contact_steps"]),
            "same_sealed_tail_produced_object_response": (
                rgb_row["object_position_error"] != rows[1]["object_position_error"]
            ),
            "old_tail_moves_away_after_rgb_prefix": True,
            "requires_rgb_refreshed_tail_replanning": True,
            "development_only": True,
            "independent_confirmation": False,
            "P1_admission": False,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--prepare-sha256", required=True)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(
        args.output.is_absolute()
        and args.output.parent == ROOT / "results"
        and not os.path.lexists(args.output),
        "exclusive output",
    )
    result = verify(args.prepared, args.prepare_sha256, args.directory, args.report_sha256)
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with args.output.open("xb") as stream:
        require(stream.write(raw) == len(raw), "complete write")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha(args.output),
                "status": result["status"],
                **result["interpretation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
