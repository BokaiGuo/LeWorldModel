"""Run fresh image-boundary feedback and compare against exposed legacy traces."""
import builtins
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_rgb_hold_retention_feedback_20261107_codex as original
from experiments.pc_input_permissions import file_hash
from experiments.pc_rgb_image_boundary_20261107_codex import RGBImageLeWM

ROOT = original.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_SHAPE_BOUNDARY_FEEDBACK_PARITY_PROTOCOL_20261107_CODEX.md"
NEW = ROOT / "results/pc_rgb_shape_boundary_feedback_parity_20261107_codex.json"
COMPARISON = ROOT / "results/pc_rgb_shape_boundary_feedback_parity_comparison_20261107_codex.json"
OLD = original.OUT
SCORE_KEYS = {"parent_cost", "probe_cost", "tail_cost", "effects", "cost"}
IGNORE_KEYS = {"planning_seconds", "wall_before_serialization_seconds", "input_audit_hash"}


def compare(a, b, path, differences):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key in IGNORE_KEYS:
                continue
            if key not in a or key not in b:
                differences.append(f"{path}.{key}: missing")
            else:
                compare(a[key], b[key], f"{path}.{key}", differences)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            differences.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{path}[{i}]", differences)
    elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
        field = path.rsplit(".", 1)[-1].split("[", 1)[0]
        atol, rtol = ((1e-4, 1e-5) if field in SCORE_KEYS else
                      (1e-6, 0) if field in {"distance", "terminal_distance"} else (0, 0))
        if not np.isclose(a, b, atol=atol, rtol=rtol):
            differences.append(f"{path}: {a} != {b}")
    elif a != b:
        differences.append(f"{path}: {a!r} != {b!r}")


def main():
    if NEW.exists() or COMPARISON.exists():
        raise FileExistsError("refusing to overwrite parity artifacts")
    original.OUT = NEW
    original.PROTOCOL = PROTOCOL
    original.RGBLeWM = RGBImageLeWM
    original.range = lambda *args: builtins.range(10, 14) if args == (10, 50) else builtins.range(*args)
    old_freeze = original.freeze_audit

    def freeze_with_boundary(track, model, files, **kwargs):
        return old_freeze(track, model, [*files, Path(__file__),
                                         Path(__import__(RGBImageLeWM.__module__, fromlist=["x"]).__file__)],
                          **kwargs)

    original.freeze_audit = freeze_with_boundary
    original.main()
    # The old result is opened only after the new execution has been saved.
    fresh = json.loads(NEW.read_text())
    historical = json.loads(OLD.read_text())
    differences = []
    for field in ("track", "input_sha256", "model_sha256", "normalizer_sha256",
                  "search_sha256", "source_plan_sha256"):
        compare(fresh[field], historical[field], field, differences)
    old_rows = {row["root"]: row for row in historical["rows"]}
    for row in fresh["rows"]:
        compare(row, old_rows[row["root"]], f"root[{row['root']}]", differences)
    report = {"protocol_sha256": file_hash(PROTOCOL), "wrapper_sha256": file_hash(__file__),
              "fresh_sha256": file_hash(NEW), "historical_sha256": file_hash(OLD),
              "roots": [row["root"] for row in fresh["rows"]],
              "difference_count": len(differences), "differences": differences}
    with COMPARISON.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"Parity comparison: {len(differences)} differences; {COMPARISON}", flush=True)


if __name__ == "__main__":
    main()
