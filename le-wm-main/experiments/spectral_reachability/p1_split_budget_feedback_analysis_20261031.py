"""Independently verify midpoint binding and compare fixed physical controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_h10_paired_table_20261030_main as metric
from . import p1_split_budget_feedback_physical_20261031 as physical
from . import p1_uphill_packet_fidelity_20261031 as frames

ROOT = physical.ROOT
NAME = "p1_split_budget_feedback_analysis_20261031"
PREP = ROOT / "results/p1_split_budget_feedback_physical_20261031_prepare_RUN01.json"
RUN = ROOT / "results/p1_split_budget_feedback_physical_20261031_run_RUN01"
CONSUME = ROOT / "results/p1_split_budget_feedback_physical_20261031_consume_RUN01.json"
HASHES = {
    PREP: "0449e00529be43f81fb244fdade7ce06afdb640bb7965065786c6f7ff91f65be",
    RUN / "run_report.json": "b436bd183e3d27ba382f8ad92ae33e26adc4b8dddaac91ff599f062197ece5a3",
    CONSUME: "5e9fc8f802e7c0c4fb4adb6b2dfea556b7fdf474142d8b342a0a1ae0226e32d6",
}
need, sha = physical.need, physical.sha


def analyze():
    need(all(sha(path) == digest for path, digest in HASHES.items()), "source hash")
    prep, report, consumed = (json.loads(path.read_bytes()) for path in HASHES)
    anchor_path = physical.physical.bundle.PREP
    need(str(anchor_path) in prep["bindings"] and sha(anchor_path) == prep["bindings"][str(anchor_path)],
         "anchor source")
    anchors = json.loads(anchor_path.read_bytes())["anchors"]
    need(len(anchors) == len(prep["roster"]) == len(report["shards"]) == len(consumed["rows"]) == 24,
         "complete roster")
    rows = []
    for i, (anchor, item, shard, row) in enumerate(zip(
            anchors, prep["roster"], report["shards"], consumed["rows"], strict=True)):
        need(i == anchor["index"] == item["index"] == shard["index"] == row["index"]
             and anchor["role"] == item["role"] == row["role"], "identity")
        path = RUN / f"row_{i:02d}" / "report.json"
        need(sha(path) == shard["report_sha256"], "report binding")
        saved = json.loads(path.read_bytes())["row"]
        frame = frames.frame_at(saved, 5, item["condition"])
        need(all(frame[k] == item["midpoint"][k] for k in ("pixel_sha256", "state", "snapshot")),
             "midpoint receipt")
        png = path.parent / frame["png"]
        need(png.name == frame["png"] and sha(png) == frame["file_sha256"], "image file")
        with Image.open(png) as image:
            pixels = np.asarray(image).copy()
        need(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
             and hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"], "midpoint pixels")
        responses = saved["responses"]
        endpoint = metric.physical_metrics(responses[-1]["state"], anchor["goal"]) if len(responses) == 10 else None
        need((row["status"] != "native_stopped" or row["last_accepted_state_metrics"]["native_success"]),
             "native stop")
        rows.append({"index": i, "logical_pair": anchor["logical_pair"], "role": row["role"],
                     "root_slot": row["root_slot"], "condition": row["condition"], "status": row["status"],
                     "accepted_suffix_steps": len(responses),
                     "native_success": row["last_accepted_state_metrics"]["native_success"],
                     "h10_endpoint": endpoint, "last_accepted": row["last_accepted_state_metrics"]})
    need(all(sha(path) == digest for path, digest in HASHES.items()), "source drift")
    return {"schema": NAME, "status": "saved_midpoint_verified", "rows": rows,
            "sources": {str(k): v for k, v in HASHES.items()}, "independent_confirmation": False,
            "equal_total_budget_certified": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(args.output.is_absolute() and args.output.parent == ROOT / "results"
         and args.output.name.startswith(NAME + "_") and not args.output.exists(), "output")
    result = analyze()
    digest = physical.physical.design.run._write_new(args.output, result)
    print(json.dumps({"rows": len(result["rows"]), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
