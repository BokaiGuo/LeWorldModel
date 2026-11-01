"""Saved P5 panel decoded-pixel audit; no model execution or physical dispatch."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
NAME = "p5_crossmodel_decoded_pixels_20261102_exclusive"
SOURCE = ROOT / "results/p5_crossmodel_panel_binding_20261030_codex_v1_MAIN01.json"
SOURCE_SHA = "80ed10d3b8f623d361cc25e2764d390bc9cc758178c89e7467e99da3d785ef98"
PANEL = ROOT / "results/agent_observed_raw_coordinate_physical_20261028_v1_MAIN01"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_pinned(path, expected):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular input")
    raw = path.read_bytes()
    require(sha(raw) == expected, "input SHA mismatch")
    return raw


def decode_record(record, panel):
    path = Path(record["path"])
    require(path.parent == panel and path.name not in ("", ".", ".."), "panel path")
    raw = read_pinned(path, record["file_sha256"])
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == (224, 224),
                "RGB PNG shape and mode")
        pixels = image.tobytes()
    observed = sha(pixels)
    require(observed == record["pixel_sha256"], "decoded pixel SHA mismatch")
    return observed


def audit(binding, panel=PANEL, expected_rows=164):
    require(binding["schema"] == "p5_crossmodel_panel_binding_20261030_codex_v1"
            and binding["status"] == "saved_input_physical_identity_bound_not_scientific_admission"
            and binding["P1_admission"] is False and binding["P5_admission"] is False,
            "saved binding contract")
    rows = binding["rows"]
    require(len(rows) == expected_rows and [r["index"] for r in rows] == list(range(expected_rows)),
            "fixed ordered rows")
    checked = []
    for row in rows:
        records = row["history_rgb"] + [row["successor_rgb_identity_only"]]
        require(len(records) == 4, "three history frames and one successor")
        condition = row["condition"]
        overlay = {"off": 0, "on": 1}[condition]
        schedule = (("warmup", 5), ("plan", 0), ("plan", 5), ("plan", 10))
        for record, expected in zip(records, schedule):
            require((record["phase"], record["step"]) == expected
                    and record["condition"] == condition and record["overlay_index"] == overlay,
                    "frame order and condition")
        hashes = [decode_record(record, panel) for record in records]
        checked.append({"index": row["index"], "condition": condition,
                        "history_pixel_sha256": hashes[:3], "successor_pixel_sha256": hashes[3]})
    return {"schema": NAME, "status": "saved_decoded_pixel_identity_verified",
            "source_sha256": SOURCE_SHA, "rows": checked, "frame_references_checked": len(rows) * 4,
            "history_frame_references_checked": len(rows) * 3,
            "successor_frame_references_checked": len(rows),
            "pixel_hash_definition": "SHA256 of decoded 224x224 RGB uint8 bytes in row-major order",
            "known_roots_only": True, "fresh_roots_certified": 0,
            "P1_admission": False, "P5_admission": False,
            "new_neural_calls": 0, "new_physical_steps": 0, "training_calls": 0,
            "boundary": "Saved pixel identity only. Successors are post-action labels, not model inputs; "
                        "JEPA additionally uses agent position and velocity. No same-permission comparison, "
                        "prediction quality, task benefit or independent confirmation is established."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.name == NAME + ".json" and args.output.parent == ROOT / "results",
            "exclusive output path")
    raw = read_pinned(SOURCE, SOURCE_SHA)
    report = audit(json.loads(raw))
    require(read_pinned(SOURCE, SOURCE_SHA) == raw, "binding changed during audit")
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": report["status"], "sha256": sha(data), "output": str(args.output)}))


if __name__ == "__main__":
    main()
