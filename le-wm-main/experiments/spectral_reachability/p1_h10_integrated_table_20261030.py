"""Combine sealed H10 measurements; descriptive only, no admission decision."""
import csv
import hashlib
import json
import sys
from pathlib import Path

TABLE_SHA = "4406169f802bf40912f9ff384d6d704b645d2d6a29ddf6f19936af32d4f3ef24"
JOIN_SHA = "bd55f5f2e07603de0a3b3d27587b942e58256e598ae7dcf5a1d8b97150ae52b5"
METRICS = ("object_position_error", "object_angle_error_wrapped",
           "pusher_position_error", "native_joint_xy_error", "native_angle_term")


def load(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("source hash mismatch: " + str(path))
    return json.loads(raw)


def build(table, join):
    if (len(table["rows"]) != 1008 or len(join["rows"]) != 24
            or join["input_sha256"]["common_table"] != TABLE_SHA):
        raise ValueError("source coverage or binding mismatch")
    anchors = {}
    for anchor in join["rows"]:
        identity = anchor["identity"]
        key = (identity["logical_pair"], identity["role"])
        if key in anchors:
            raise ValueError("duplicate anchor")
        anchors[key] = anchor
    output = []
    seen = set()
    for row in table["rows"]:
        key = (row["logical_pair"], row["role"])
        anchor = anchors[key]
        identity = anchor["identity"]
        if any(row[k] != identity[k] for k in ("root_slot", "condition", "arm", "role")):
            raise ValueError("anchor identity mismatch")
        matching = [p for p in anchor["common_suffixes"]
                    if p["probe"] == row["probe"] and p["raw10_sha256"] == row["probe_sha256"]]
        if (len(matching) != 1 or matching[0]["status"] != row["status"]
                or matching[0]["endpoint_metrics"] != row["common_suffix_endpoint_native_metrics"]):
            raise ValueError("probe identity mismatch")
        unique = key + (row["probe_sha256"], row["probe"])
        if unique in seen:
            raise ValueError("duplicate probe")
        seen.add(unique)
        ind = anchor["independent_suffix"]
        result = {k: row[k] for k in ("root_slot", "condition", "arm", "logical_pair",
                                     "role", "duplicate_group", "probe", "probe_sha256")}
        result["common_status"] = row["status"]
        result["common_observation"] = json.dumps(row["observation"], sort_keys=True)
        result["prefix_success_root"] = row["prefix_native_success_transition"]["root"]
        result["prefix_success_after"] = row["prefix_native_success_transition"]["prefix"]
        for metric in METRICS:
            root, prefix = row["root_metrics"], row["prefix_metrics"]
            result["prefix_delta_" + metric] = prefix[metric] - root[metric]
            for label, field in (("common_endpoint_", "common_suffix_endpoint_native_metrics"),
                                 ("common_vs_own_zero_", "same_prefix_endpoint_minus_own_zero_suffix")):
                values = row[field]
                result[label + metric] = None if values is None else values.get(metric)
            endpoint = ind["endpoint_metrics"]
            result["independent_endpoint_" + metric] = None if endpoint is None else endpoint.get(metric)
            # Use the actual zero-suffix endpoint, not this probe's endpoint.
            zero_probe = next(p for p in anchor["common_suffixes"] if p["probe"] == "zero")
            zero_metrics = zero_probe["endpoint_metrics"]
            result["independent_vs_own_zero_" + metric] = (
                None if endpoint is None or zero_metrics is None
                else endpoint[metric] - zero_metrics[metric])
        result["common_success"] = (None if row["common_suffix_endpoint_native_metrics"] is None
                                    else row["common_suffix_endpoint_native_metrics"]["native_success"])
        result["independent_success"] = (None if ind["endpoint_metrics"] is None
                                         else ind["endpoint_metrics"]["native_success"])
        result["independent_status"] = ind["status"]
        result["independent_last_observed"] = json.dumps(ind["last_accepted_state_metrics"], sort_keys=True)
        result["independent_action_sha256"] = ind["raw10_sha256"]
        result["independent_resource_receipt"] = json.dumps(ind["actual_planning_resources"], sort_keys=True)
        result["independent_physical_resources"] = json.dumps(ind["actual_physical_resources"], sort_keys=True)
        output.append(result)
    return output


def main():
    table = load(sys.argv[1], TABLE_SHA)
    join = load(sys.argv[2], JOIN_SHA)
    rows = build(table, join)
    with open(sys.argv[3], "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} descriptive rows; 24 independent suffixes, not 1008")


if __name__ == "__main__":
    main()
