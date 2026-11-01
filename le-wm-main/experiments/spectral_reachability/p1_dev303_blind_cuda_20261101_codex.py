"""CUDA encode and task-export the sealed dev303 development root."""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_blind_adapter_20261101_codex as adapter
from . import p1_queue_goal_blind_cuda_20261101_codex as core


NAME = "p1_dev303_blind_cuda_20261101_codex"
ROOT = core.ROOT
MANIFEST = ROOT / "results/p1_dev303_blind_adapter_20261101_codex_MAIN02.json"
MANIFEST_SHA = "b7b5a1b2394acdad13981c1fb4258d97941e0f1863e816e9dc86d85b91b460dc"
BLIND_SHA = "092af72253c9b0463e5c7a5814052599843dc4b6a979bd040d7f318c93dcddd9"
b = core.b


def build():
    b.require(b.sha(MANIFEST) == MANIFEST_SHA, "dev303 adapter manifest")
    manifest = b.read(MANIFEST)
    b.require(
        manifest["blind_sha256"] == BLIND_SHA
        and manifest["seed"] == 301000003
        and manifest["independent_confirmation"] is False
        and manifest["P1_admission"] is False,
        "dev303 development identity",
    )
    arrays, receipt_sha, audit = adapter.source()
    b.require(
        receipt_sha == manifest["source_receipt_sha256"]
        and audit["seed"] == manifest["seed"],
        "receipt identity",
    )
    packet = b.load_blind(manifest["blind_path"], BLIND_SHA)
    b.require(
        all(np.array_equal(packet[key], arrays[key]) for key in arrays),
        "receipt/blind exact equality",
    )
    pixels = [
        b.digest(value.tobytes())
        for value in [*packet["rgb_history"], packet["goal_rgb"]]
    ]
    action = b.digest(packet["raw_action_history"].tobytes())
    b.require(
        pixels == manifest["pixel_sha256"]
        and action == manifest["action_sha256"],
        "manifest identity",
    )
    slot = {
        "order": 0,
        "source_slot": 0,
        "blind_npz": manifest["blind_path"],
        "blind_sha256": BLIND_SHA,
        "pixel_sha256": pixels,
        "raw_action_sha256": action,
    }
    files = set(b.source_files()) | {
        Path(__file__),
        Path(adapter.__file__),
        Path(core.__file__),
        Path(core.locking.__file__),
    }
    bindings = {str(path.resolve()): b.sha(path) for path in files}
    bindings.update(b.MODELS)
    bindings[str(MANIFEST)] = MANIFEST_SHA
    bindings[manifest["blind_path"]] = BLIND_SHA
    return {
        "schema": NAME + ".prepare.v1",
        "slots": [slot],
        "normalization": b.native_metadata(),
        "model_bindings": b.MODELS,
        "policy": str(b.POLICY),
        "expected_model_hash": b.MODEL_TENSOR_SHA,
        "arguments": {"whitelist_sha": MANIFEST_SHA},
        "bindings_sha256": bindings,
        "source_receipt_sha256": receipt_sha,
        "seed": manifest["seed"],
        "contract": b.contract(1),
        "classification": "dev303_development_only",
        "historical_coverage": "unknown",
        "training_membership": "unknown",
        "independent_confirmation": False,
        "P1_admission": False,
    }


@contextmanager
def _patched():
    values = {
        "NAME": NAME,
        "MANIFEST": MANIFEST,
        "MANIFEST_SHA": MANIFEST_SHA,
        "BLIND_SHA": BLIND_SHA,
        "adapter": adapter,
        "build": build,
    }
    previous = {key: getattr(core, key) for key in values}
    for key, value in values.items():
        setattr(core, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(core, key, value)


def prepare(output):
    with _patched():
        return core.prepare(output)


def run(prepared, expected, output):
    with _patched():
        return core.run(prepared, expected, output)


def verify(prepared, expected, output):
    with _patched():
        return core.verify_saved(prepared, expected, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--prepared")
    parser.add_argument("--prepare-sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "run":
        result = run(args.prepared, args.prepare_sha256, args.output)
    else:
        result = verify(args.prepared, args.prepare_sha256, args.output)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
