"""Saved-only, root-stratified descriptive audit of the fixed P5 panel."""
import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_SHA = "6685430d507ddc808b500356e094622816fbc180bd2f3bca73c12cd3d56b51c8"
COORDS = ["agent_x", "agent_y", "block_x", "block_y", "sin_angle", "cos_angle"]


def summarize(data):
    if data.get("schema") != "p5_rgb_refresh_openloop_physical_compare_20261030_codex_v1.result" or data.get("P5_admission") is not False:
        raise ValueError("unexpected source contract")
    rows = data["rows"]
    if len(rows) != 164 or data["coordinates"] != COORDS:
        raise ValueError("unexpected panel")
    groups = {(root, cond): [] for root in (0, 1) for cond in ("off", "on")}
    indexes, selections, tasks = set(), set(), {}
    for row in rows:
        key = (row["root_slot"], row["condition"])
        if key not in groups or row["index"] in indexes or row["selection_digest"] in selections:
            raise ValueError("invalid identity")
        indexes.add(row["index"])
        selections.add(row["selection_digest"])
        tasks.setdefault(key, set()).add(row["task_id"])
        if len(row["steps"]) != 5:
            raise ValueError("invalid horizons")
        for h, step in enumerate(row["steps"], 1):
            if step["horizon"] != h or step["plan_step"] != h + 5:
                raise ValueError("invalid time alignment")
            for field in ("physical", "open_prediction", "refresh_prediction", "open_absolute_error", "refresh_absolute_error", "refresh_minus_open_absolute_error"):
                if len(step[field]) != 6 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in step[field]):
                    raise ValueError("invalid numeric vector")
            for i in range(6):
                op = abs(step["open_prediction"][i] - step["physical"][i])
                rf = abs(step["refresh_prediction"][i] - step["physical"][i])
                if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in ((op, step["open_absolute_error"][i]), (rf, step["refresh_absolute_error"][i]), (rf-op, step["refresh_minus_open_absolute_error"][i]))):
                    raise ValueError("inconsistent saved errors")
        groups[key].append(row)
    if indexes != set(range(164)) or any(len(v) != 41 for v in groups.values()) or any(len(v) != 1 for v in tasks.values()) or len(set.union(*tasks.values())) != 4:
        raise ValueError("unexpected root or group identity")
    output = []
    for (root, cond), group in sorted(groups.items()):
        for h in range(5):
            for i, coord in enumerate(COORDS):
                differences = [r["steps"][h]["refresh_minus_open_absolute_error"][i] for r in group]
                output.append({"root_slot": root, "condition": cond, "horizon": h + 1, "coordinate": coord,
                               "branches": len(group), "mean_open_absolute_error": math.fsum(r["steps"][h]["open_absolute_error"][i] for r in group)/len(group),
                               "mean_refresh_absolute_error": math.fsum(r["steps"][h]["refresh_absolute_error"][i] for r in group)/len(group),
                               "refresh_better": sum(x < 0 for x in differences), "tie": sum(x == 0 for x in differences),
                               "refresh_worse": sum(x > 0 for x in differences)})
    return {"status": "known_panel_descriptive_only", "P5_admission": False,
            "boundary": "Two known roots; correlated branches; refresh uses extra true observations. No independent inference, fair closed-loop comparison, or model ranking.",
            "root_count": 2, "branch_count": 164, "groups": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA:
        raise ValueError("source SHA mismatch")
    result = summarize(json.loads(raw))
    result["source_sha256"] = digest
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
