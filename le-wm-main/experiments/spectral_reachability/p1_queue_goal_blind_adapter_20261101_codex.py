"""Development-only receipt to the original zero-warmup blind NPZ contract."""
import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_constructive_blind_encode_cuda_20261027_v1 as blind
from . import p1_queue_goal_saved_20261101_codex as saved

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "results/p1_queue_goal_history_20261101_codex_QUEUE_DEV01.json"
QUEUE_SHA = "6756ad9f3ccf058baa422520d3058dbbdc10e7ac86ff1091b53b719f5a1fe847"
REPORT_SHA = "05ce658129367dcad6d8bdf9a05e963758d1eec144ad760cbd05786dd7e7a354"
SLOT = "dev_reused"
ATTEMPT = 0


def image(directory, item):
    path = (directory / item["file"]).resolve(strict=True)
    blind.require(path.parent == directory.resolve() and blind.sha(path) == item["file_sha256"], "image source")
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    blind.require(rgb.shape == (224, 224, 3) and blind.digest(rgb.tobytes()) == item["pixel_sha256"], "image pixels")
    return rgb


def packet(history, directory):
    frames = history["frames"]
    blind.require([f["step"] for f in frames] == [0, 5, 10], "actual history steps")
    raw = np.asarray(history["actions"], dtype=np.float32)
    blind.require(raw.shape == (10, 2) and np.isfinite(raw).all() and not raw.any(), "actual zero action history")
    return dict(rgb_history=np.stack([image(directory, f) for f in frames]),
                raw_action_history=raw, goal_rgb=image(directory, history["goal_rgb"]))


def source():
    audit = saved.verify(QUEUE, QUEUE_SHA, SLOT, ATTEMPT, REPORT_SHA)
    directory = QUEUE.parent / (QUEUE.name + ".native_reset") / f"{SLOT}_{ATTEMPT:04d}.warmup"
    receipt = directory / "source_receipt.json"
    receipt_sha = blind.sha(receipt)
    data = blind.read(receipt)
    blind.require(data["schema"] == "p1_queue_goal_history_20261101_codex.source.v1"
                  and data["P1_admission"] is False and data["independent_confirmation"] is False,
                  "development receipt")
    result = packet(data["history"], directory)
    blind.require(blind.sha(receipt) == receipt_sha, "receipt drift")
    blind.require(saved.verify(QUEUE, QUEUE_SHA, SLOT, ATTEMPT, REPORT_SHA) == audit, "source drift")
    return result, receipt_sha, audit


def build(output):
    output = Path(output).absolute()
    blind.require(output.name.endswith("_blind_inputs.npz") and not output.exists(), "exclusive blind output")
    arrays, receipt_sha, audit = source()
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    blind.require(len(payload) <= 2 * 1024**2, "archive bound")
    with output.open("xb") as f:
        f.write(payload)
        f.flush()
    expected = blind.digest(payload)
    readback = blind.load_blind(output, expected)
    blind.require(all(np.array_equal(readback[k], v) for k, v in arrays.items()), "blind readback")
    arrays2, receipt_sha2, audit2 = source()
    blind.require(receipt_sha2 == receipt_sha and audit2 == audit and
                  all(np.array_equal(arrays2[k], v) for k, v in arrays.items()), "postwrite source drift")
    return dict(schema="p1_queue_goal_blind_adapter_20261101_codex.v1",
                status="blind_npz_verified_not_encoded", blind_path=str(output),
                blind_sha256=expected, queue_sha256=QUEUE_SHA, report_sha256=REPORT_SHA,
                source_receipt_sha256=receipt_sha, seed=audit["seed"],
                pixel_sha256=[blind.digest(x.tobytes()) for x in
                              [*arrays["rgb_history"], arrays["goal_rgb"]]],
                action_sha256=blind.digest(arrays["raw_action_history"].tobytes()),
                observed_steps=[0, 5, 10], actual_zero_actions=10,
                historical_coverage="unknown", training_membership="unknown",
                independent_confirmation=False, P1_admission=False,
                limitation="original runner fixed whitelist rejects this new NPZ; loader compatibility is not runner admission")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).absolute()
    blind.require(not manifest.exists() and manifest != Path(args.output).absolute(), "exclusive manifest")
    value = build(args.output)
    with manifest.open("xb") as f:
        f.write(blind.canonical(value))
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
