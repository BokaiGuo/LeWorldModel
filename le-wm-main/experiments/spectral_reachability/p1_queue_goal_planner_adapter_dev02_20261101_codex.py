"""Lossless dev02 metadata adapter to the frozen legacy planner envelope."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

from . import p1_queue_goal_planner_adapter_20261101_codex as base

ROOT, planner = base.ROOT, base.planner
NAME = "p1_queue_goal_planner_adapter_dev02_20261101_codex"


def adapt(source_path, source_sha):
    source_path = Path(source_path).resolve(strict=True)
    base.require(base.sha(source_path) == source_sha, "source SHA")
    value = base.read_gzip(source_path)
    base.require(
        set(value) == base.LEGACY_FIELDS | {"development_seed"}
        and value["development_seed"] == 301000002
        and value["P1_admission"] is False
        and value["new_independent_confirmation"] is False
        and value["native_root_sha256"] is None,
        "dev02 envelope",
    )
    adapted = {key: item for key, item in value.items() if key != "development_seed"}
    planner.validate_export_envelope(adapted)
    interface = planner.interface(planner.encode.task_interfaces(adapted))
    base.require(interface["task_id"] == value["task_id"], "task identity")
    return adapted


def write(source_path, source_sha, output, manifest):
    source_path = Path(source_path).resolve(strict=True)
    output, manifest = Path(output).resolve(), Path(manifest).resolve()
    base.require(
        output != manifest
        and output.parent == manifest.parent == ROOT / "results"
        and output.name.startswith("task_")
        and output.name.endswith(".json.gz")
        and not os.path.lexists(output)
        and not os.path.lexists(manifest),
        "exclusive outputs",
    )
    adapted = adapt(source_path, source_sha)
    raw = json.dumps(adapted, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    with output.open("xb") as stream:
        stream.write(compressed)
        stream.flush()
        os.fsync(stream.fileno())
    planner.load_export(output, base.sha(output))
    result = {
        "schema": NAME + ".manifest.v1",
        "status": "legacy_envelope_only_task_arrays_unchanged",
        "source": str(source_path),
        "source_sha256": source_sha,
        "output": str(output),
        "output_sha256": base.sha(output),
        "removed_metadata": {"development_seed": 301000002},
        "task_id": adapted["task_id"],
        "task_payload_sha256": planner.encode.core.digest(adapted["task"]),
        "development_only": True,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with manifest.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(write(args.source, args.source_sha256, args.output, args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
