"""Export one row per prefix and common suffix; independent outcome repeats per suffix."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/codex_p1_adam_paired_join_20261030_main_RUN01.json"
DEST = ROOT / "results/codex_p1_adam_paired_table_20261030_main_RUN01.csv"
EXPECTED = "7fd04ee76f6a5bfafdb9f6249db961bf6be6fb26829df19fc642438f88a9daa8"
FIELDS = ("object_position_error", "object_angle_error_wrapped", "pusher_position_error",
          "native_joint_xy_error", "native_success")


def add_metrics(out, name, metrics, fields=FIELDS):
    for field in fields:
        out[f"{name}_{field}"] = None if metrics is None else metrics.get(field)


def main():
    data = SOURCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError("source hash mismatch")
    view = json.loads(data)
    records = []
    for row in view["rows"]:
        ident = row["identity"]
        adam = row["adam_independent_suffix"]
        if len(row["common_suffixes"]) != 42:
            raise ValueError("common suffix count")
        for probe in row["common_suffixes"]:
            record = {k: ident[k] for k in ("index", "logical_pair", "root_slot", "condition", "role", "arm", "task_id")}
            record.update(prefix_hex=ident["prefix5_float32_hex"], common_probe=probe["probe"],
                          common_suffix_sha256=probe["probe_sha256"],
                          common_status=probe["status"],
                          common_steps=probe["observation"]["accepted_probe_steps"],
                          common_censor=probe["observation"]["censor"],
                          adam_status=adam["status"], adam_steps=adam["accepted_steps"],
                          adam_action_sha256=adam["raw10_sha256"],
                          adam_last_native_success=adam["last_observed_metrics"]["native_success"],
                          adam_predict_calls_code_path=adam["planning_calls_code_path"]["predict"],
                          adam_backward_calls_code_path=adam["planning_calls_code_path"]["backward"],
                          adam_planning_seconds=adam["planning_wall_seconds"])
            add_metrics(record, "root", row["root_metrics"])
            add_metrics(record, "prefix", row["prefix_metrics"])
            add_metrics(record, "prefix_delta", row["prefix_change"], FIELDS[:-1])
            add_metrics(record, "common_endpoint", probe["common_suffix_endpoint_native_metrics"])
            add_metrics(record, "common_delta_own_zero", probe["same_prefix_endpoint_minus_own_zero_suffix"], FIELDS[:-1])
            add_metrics(record, "adam_endpoint", adam["horizon_metrics"])
            add_metrics(record, "adam_delta_own_zero", adam["endpoint_minus_own_zero_suffix"], FIELDS[:-1])
            add_metrics(record, "adam_delta_zero_prefix", adam["endpoint_minus_zero_prefix_independent_endpoint"], FIELDS[:-1])
            records.append(record)
    with DEST.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"rows": len(records), "sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(), "path": str(DEST)}))


if __name__ == "__main__":
    main()
