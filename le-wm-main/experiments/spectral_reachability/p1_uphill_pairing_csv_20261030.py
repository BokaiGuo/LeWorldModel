"""Export a verified complete P1 pairing receipt as a descriptive CSV.

Each row is a common-suffix branch; independent planning results are repeated
for convenience and must be deduplicated by anchor_index for analysis.
This exporter performs no inference, physics, or admission decision.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from . import p1_uphill_complete_pairing_20261030 as pairing

ROLES = pairing.ROLES
FIELDS = pairing.FIELDS


def flatten(prefix, metrics, row):
    for field in FIELDS:
        row[f"{prefix}_{field}"] = None if metrics is None else metrics[field]
    row[f"{prefix}_native_success"] = (
        None if metrics is None else metrics.get("native_success")
    )


def export_rows(receipt):
    pairing.require(receipt["schema"] == "p1_uphill_complete_pairing_20261030"
                    and receipt["P1_admission"] is False
                    and receipt["independent_confirmation"] is False
                    and receipt["equal_budget_certified"] is False, "receipt contract")
    anchors = receipt["anchors"]
    pairs = receipt["common_suffix_pairs"]
    pairing.require(len(anchors) == 24 and len(pairs) == 336
                    and receipt["common_suffix_branches"] == 1008
                    and receipt["independent_plans_per_initialization"] == 24, "roster")
    by_index = {anchor["index"]: anchor for anchor in anchors}
    pairing.require(len(by_index) == 24 and set(by_index) == set(range(24)), "anchor indices")
    seen = set()
    rows = []
    for pair in pairs:
        key = (pair["logical_pair"], pair["probe"])
        pairing.require(key not in seen and set(pair["roles"]) == set(ROLES), "pair roster")
        seen.add(key)
        for role in ROLES:
            branch = pair["roles"][role]
            anchor = by_index[branch["anchor_index"]]
            identity = anchor["identity"]
            pairing.require(identity["role"] == role
                            and identity["logical_pair"] == pair["logical_pair"], "anchor binding")
            row = dict(identity)
            row.update(anchor_index=anchor["index"], probe=pair["probe"],
                       probe_sha256=pair["probe_sha256"],
                       prefix_action_sha256=anchor["prefix_action_sha256"],
                       prefix_cp5_pixel_sha256=anchor["prefix_cp5_pixel_sha256"],
                       common_status=branch["status"],
                       common_returned_suffix_steps=branch["returned_suffix_steps"],
                       independent_plan_repeated_per_probe=True,
                       equal_budget_certified=False, independent_confirmation=False,
                       p1_admission=False)
            for name, metrics in (
                ("root", anchor["root_metrics"]),
                ("prefix", anchor["prefix_metrics"]),
                ("prefix_minus_root", anchor["prefix_minus_root"]),
                ("common_endpoint", branch["endpoint"]),
                ("common_response_minus_own_zero", branch["endpoint_minus_own_zero_suffix"]),
                ("common_endpoint_minus_zero_prefix", branch["endpoint_minus_zero_prefix"]),
                ("common_response_minus_zero_prefix", branch["response_minus_zero_prefix"]),
            ):
                flatten(name, metrics, row)
            for kind in ("zero", "warm"):
                plan = anchor[f"{kind}_initialization"]
                row[f"{kind}_plan_status"] = plan["status"]
                # Preserve all saved plan fields rather than inferring stop/success
                # from a missing fixed-horizon endpoint.
                row[f"{kind}_plan_receipt_json"] = json.dumps(plan, sort_keys=True, allow_nan=False)
                row[f"{kind}_resource_receipt_json"] = json.dumps(
                    anchor[f"{kind}_resources"], sort_keys=True, allow_nan=False
                )
                flatten(f"{kind}_plan_endpoint", plan["endpoint"], row)
            rows.append(row)
    pairing.require(len(rows) == 1008 and len(seen) == 336, "row count")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt_path = Path(args.receipt)
    output = Path(args.output)
    pairing.require(output.is_absolute() and output.parent == pairing.ROOT / "results"
                    and output.name.startswith("p1_uphill_pairing_")
                    and output.suffix == ".csv" and not output.exists()
                    and not output.is_symlink(), "exclusive output")
    receipt = pairing.read_checked(receipt_path, args.receipt_sha256)
    rows = export_rows(receipt)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    raw = text.getvalue().encode("utf-8")
    pairing.require(hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                    == args.receipt_sha256, "receipt drift")
    with output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(raw).hexdigest(),
                      "rows": len(rows), "anchor_records": 24,
                      "anchor_records_are_independent_samples": False}))


if __name__ == "__main__":
    main()
