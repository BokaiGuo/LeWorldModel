"""Development holdout02 receipt -> original zero-warmup blind NPZ contract."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_constructive_blind_encode_cuda_20261027_v1 as blind
from . import p1_queue_goal_saved_20261101_codex as saved

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV02.json"
QUEUE_SHA = "467975c9c23046e430aae0163fef0450cffc03c27fdf32f8294c616724d0067d"
REPORT_SHA = "aa88db0c4d5f8303ca24bd1ac52b1f6ffeaf6572ef81074589f4b1a934bbf919"
SLOT = "dev_holdout02"
ATTEMPT = 0


def image(directory, item):
    path = (directory / item["file"]).resolve(strict=True)
    blind.require(path.parent == directory.resolve() and blind.sha(path) == item["file_sha256"], "image")
    with Image.open(path) as handle:
        value = np.asarray(handle.convert("RGB"), np.uint8).copy()
    blind.require(
        value.shape == (224, 224, 3) and blind.digest(value.tobytes()) == item["pixel_sha256"],
        "pixels",
    )
    return value


def source():
    audit = saved.verify(QUEUE, QUEUE_SHA, SLOT, ATTEMPT, REPORT_SHA)
    directory = QUEUE.parent / (QUEUE.name + ".native_reset") / f"{SLOT}_{ATTEMPT:04d}.warmup"
    receipt = directory / "source_receipt.json"
    receipt_sha = blind.sha(receipt)
    data = blind.read(receipt)
    history = data["history"]
    frames = history["frames"]
    raw = np.asarray(history["actions"], np.float32)
    blind.require(
        audit["seed"] == 301000002
        and data["P1_admission"] is False
        and data["independent_confirmation"] is False
        and [row["step"] for row in frames] == [0, 5, 10]
        and raw.shape == (10, 2)
        and not raw.any(),
        "development holdout02 receipt",
    )
    packet = {
        "rgb_history": np.stack([image(directory, row) for row in frames]),
        "raw_action_history": raw,
        "goal_rgb": image(directory, history["goal_rgb"]),
    }
    blind.require(blind.sha(receipt) == receipt_sha, "receipt drift")
    return packet, receipt_sha, audit


def build(output):
    output = Path(output).absolute()
    blind.require(output.name.endswith("_blind_inputs.npz") and not output.exists(), "exclusive output")
    arrays, receipt_sha, audit = source()
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    blind.require(len(payload) <= 2 * 1024**2, "archive bound")
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
    expected = blind.digest(payload)
    readback = blind.load_blind(output, expected)
    blind.require(all(np.array_equal(readback[key], value) for key, value in arrays.items()), "readback")
    return {
        "schema": "p1_queue_goal_blind_adapter_dev02_20261101_codex.v1",
        "status": "blind_npz_verified_not_encoded",
        "blind_path": str(output),
        "blind_sha256": expected,
        "queue_sha256": QUEUE_SHA,
        "report_sha256": REPORT_SHA,
        "source_receipt_sha256": receipt_sha,
        "seed": audit["seed"],
        "pixel_sha256": [
            blind.digest(value.tobytes())
            for value in [*arrays["rgb_history"], arrays["goal_rgb"]]
        ],
        "action_sha256": blind.digest(arrays["raw_action_history"].tobytes()),
        "observed_steps": [0, 5, 10],
        "actual_zero_actions": 10,
        "historical_coverage": "unknown",
        "training_membership": "unknown",
        "independent_confirmation": False,
        "P1_admission": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).absolute()
    blind.require(not manifest.exists(), "exclusive manifest")
    value = build(args.output)
    with manifest.open("xb") as stream:
        stream.write(blind.canonical(value))
        stream.flush()
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
