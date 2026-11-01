"""Paired absolute and zero-suffix-relative physical intervention metrics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_coherent_intervention_20261101_main as experiment


def analyze(directory):
    directory = Path(directory)
    report_path, protocol_path = directory / "report.json", directory / "protocol.json"
    report, protocol = (json.loads(p.read_bytes()) for p in (report_path, protocol_path))
    sha = experiment.panel.saved.sha
    if sha(protocol_path) != report["protocol_sha256"] or protocol["sources"] != report["sources"]:
        raise ValueError("protocol binding")
    sources = dict(report["sources"])
    if any(sha(p) != h for p, h in sources.items()):
        raise ValueError("source drift")
    for path in (report_path, protocol_path, Path(__file__).resolve()):
        sources[str(path)] = sha(path)
    expected = [(prefix, suffix) for prefix in (*experiment.panel.ARMS, "zero")
                for suffix in ("common_structured", "zero")]
    if [(r["prefix"], r["suffix"]) for r in report["rows"]] != expected:
        raise ValueError("fixed10 roster")
    # Target identity comes from the original, unchanged centered target collector.
    goal, _, _ = experiment.target.collect()
    checked = {}
    for row, slot in zip(report["rows"], protocol["schedule"], strict=True):
        key = (row["prefix"], row["suffix"])
        if key != (slot["prefix"], slot["suffix"]) or row["status"] != "completed20" or row["steps"] != 20:
            raise ValueError("analysis requires complete20; do not impute censored endpoints")
        if not np.array_equal(np.asarray(row["executed_raw"], np.float32), np.asarray(slot["commands"], np.float32)):
            raise ValueError("executed actions")
        if [p["step"] for p in row["checkpoints"]] != [0, 5, 10, 15, 20]:
            raise ValueError("five checkpoints")
        for point in row["checkpoints"]:
            path = Path(point["png"])
            if path.parent != directory or sha(path) != point["file_sha256"]:
                raise ValueError("frame file")
            with Image.open(path) as im:
                pixels = np.asarray(im)
            if hashlib.sha256(pixels.tobytes()).hexdigest() != point["pixel_sha256"]:
                raise ValueError("frame pixels")
            native = experiment.metrics.qualification.native(goal["goal_state"], point["state"])
            expected_metrics = dict(**experiment.metrics.metrics(point["state"], goal["goal_state"]),
                                    joint_xy=native["joint_xy_error"],
                                    native_success=native["initial_success"])
            if expected_metrics != point["metrics"]:
                raise ValueError("physical metric recomputation")
            sources[str(path)] = sha(path)
        if row["checkpoints"][0]["state"] != goal["root_state"]:
            raise ValueError("shared root")
        checked[key] = row
    source_path = experiment.panel.ROOT / "results/p1_centered_panel_20261101_main_COHERENTMAIN01.json"
    source = json.loads(source_path.read_bytes())
    for item in source["rows"]:
        original = json.loads(Path(item["report"]).read_bytes())["rows"][0]
        comparison = checked[(item["arm"], "common_structured")]["checkpoints"]
        original_points = {p["step"]: p for p in original["checkpoints"]}
        for point in comparison:
            if point["step"] <= 5 or item["arm"] == "structured":
                prior = original_points[point["step"]]
                if point["state"] != prior["state"] or point["pixel_sha256"] != prior["pixel_sha256"]:
                    raise ValueError("source prefix/full structured replay")
    fields = ("object_xy", "object_angle", "pusher_xy", "joint_xy")
    rows = []
    for prefix in (*experiment.panel.ARMS, "zero"):
        common, zero = checked[(prefix, "common_structured")], checked[(prefix, "zero")]
        for a, b in zip(common["checkpoints"][:2], zero["checkpoints"][:2], strict=True):
            if a["state"] != b["state"] or a["pixel_sha256"] != b["pixel_sha256"]:
                raise ValueError("paired identical prefix")
        initial, cp5 = (p["metrics"] for p in common["checkpoints"][:2])
        a, b = common["checkpoints"][-1]["metrics"], zero["checkpoints"][-1]["metrics"]
        rows.append(dict(prefix=prefix, prefix_delta={k: cp5[k]-initial[k] for k in fields},
                         common_terminal=a, zero_terminal=b,
                         common_minus_zero={k: a[k]-b[k] for k in fields},
                         stops_or_censoring=False))
    return dict(rows=rows, sources=sources, P1_admission=False, independent_confirmation=False,
                capability_advantage_proven=False, known_development_roots=1,
                scope="Effect of same suffix only, not independently optimized suffix capacity.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["rows"]), flush=True)
