"""Lossless metadata adapter from the queue task export to the legacy planner envelope."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

from . import p1_constructive_blind_planners_cuda_20261027_v1 as planner

ROOT = planner.ROOT
NAME = "p1_queue_goal_planner_adapter_20261101_codex"
LEGACY_FIELDS = {
    "schema", "task", "task_id", "task_contract", "blind_source_sha256",
    "native_root_sha256", "source_slot", "with_target", "blind_sha256",
    "history_semantics", "source_admission", "global_freshness", "P1_admission",
    "new_independent_confirmation",
}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_gzip(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 2 * 1024**2,
            "bounded regular source")
    raw = gzip.decompress(path.read_bytes())
    require(len(raw) < 8 * 1024**2, "expanded source cap")
    return json.loads(raw)


def adapt(source_path, source_sha):
    source_path = Path(source_path).resolve(strict=True)
    require(sha(source_path) == source_sha, "external source SHA")
    value = read_gzip(source_path)
    require(set(value) == LEGACY_FIELDS | {"development_seed"}, "queue export fields")
    require(
        type(value["development_seed"]) is int
        and value["development_seed"] == 301000001
        and value["P1_admission"] is False
        and value["new_independent_confirmation"] is False
        and value["native_root_sha256"] is None,
        "development-only boundary",
    )
    adapted = {key: value[key] for key in value if key != "development_seed"}
    planner.validate_export_envelope(adapted)
    interface = planner.interface(planner.encode.task_interfaces(adapted))
    require(interface["task_id"] == value["task_id"], "planner interface identity")
    return adapted


def write(source_path, source_sha, output, manifest):
    source_path = Path(source_path).resolve(strict=True)
    output, manifest = Path(output).resolve(), Path(manifest).resolve()
    require(
        output != manifest
        and output.parent == manifest.parent == ROOT / "results"
        and output.name.startswith("task_")
        and output.name.endswith(".json.gz")
        and not os.path.lexists(output)
        and not os.path.lexists(manifest),
        "exclusive adapter outputs",
    )
    adapted = adapt(source_path, source_sha)
    raw = json.dumps(adapted, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    with output.open("xb") as stream:
        require(stream.write(compressed) == len(compressed), "complete task write")
        stream.flush()
        os.fsync(stream.fileno())
    planner.load_export(output, sha(output))
    result = {
        "schema": NAME + ".manifest.v1",
        "status": "legacy_envelope_only_task_arrays_unchanged",
        "source": str(source_path),
        "source_sha256": source_sha,
        "output": str(output),
        "output_sha256": sha(output),
        "removed_metadata": {"development_seed": 301000001},
        "task_id": adapted["task_id"],
        "task_payload_sha256": planner.encode.core.digest(adapted["task"]),
        "development_only": True,
        "independent_confirmation": False,
        "P1_admission": False,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with manifest.open("xb") as stream:
        require(stream.write(payload) == len(payload), "complete manifest write")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(write(args.source, args.source_sha256, args.output, args.manifest),
                     sort_keys=True))


if __name__ == "__main__":
    main()
