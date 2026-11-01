"""Export the sealed crossover as paired records, preserving censored horizons."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

METRICS = ("object_position_error", "object_angle_error_wrapped",
           "pusher_position_error", "native_joint_xy_error")
ROLES = ("selected", "zero", "greedy")
PROBES = ("zero_suffix", "selected_feedback_tail", "zero_feedback_tail",
          "greedy_feedback_tail")
EXPECTED = "1533a282067f25ebb7f9777ffa0e81d8392ee200aba4dd39d80d917f83abc7e9"


def build(data):
    if data["status"] != "saved_checks_passed" or len(data["rows"]) != 96:
        raise ValueError("incomplete saved consumer")
    index = {}
    for row in data["rows"]:
        key = (row["logical_pair"], row["role"], row["probe"])
        if key in index:
            raise ValueError("duplicate branch")
        index[key] = row
    out = []
    for pair in range(8):
        for probe in PROBES:
            group = [index[(pair, role, probe)] for role in ROLES]
            if len({(r["root_slot"], r["condition"], r["arm"], r["probe_sha256"])
                    for r in group}) != 1 or any(r["root"] != group[0]["root"] for r in group):
                raise ValueError("paired identity mismatch")
            for r in group:
                zero = index[(pair, r["role"], "zero_suffix")]
                if r["prefix"] != zero["prefix"]:
                    raise ValueError("prefix mismatch")
                result = {k: r[k] for k in ("logical_pair", "root_slot", "condition",
                          "arm", "role", "probe", "probe_sha256", "status",
                          "accepted_suffix_steps", "native_success")}
                for metric in METRICS:
                    result["prefix_change_" + metric] = r["prefix"][metric] - r["root"][metric]
                    result["endpoint_" + metric] = None if r["endpoint"] is None else r["endpoint"][metric]
                    result["delta_vs_zero_" + metric] = (
                        None if r["endpoint"] is None or zero["endpoint"] is None
                        else r["endpoint"][metric] - zero["endpoint"][metric])
                out.append(result)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    content = args.input.read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED:
        raise ValueError("unexpected input hash")
    rows = build(json.loads(content))
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} paired records; missing horizons remain blank")


if __name__ == "__main__":
    main()
