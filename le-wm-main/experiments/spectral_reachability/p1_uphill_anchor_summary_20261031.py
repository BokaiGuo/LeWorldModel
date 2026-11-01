"""Summarize each P1 prefix once, without treating shared probes as independent."""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from . import p1_uphill_complete_pairing_20261030 as pairing
from . import p1_uphill_pairing_csv_20261031 as long_table

NAME = "p1_uphill_anchor_summary_20261031"


def summarize(result):
    # flatten validates the complete roster and pair-to-anchor bindings.
    grouped = defaultdict(list)
    for row in long_table.flatten(result):
        grouped[row["anchor_index"]].append(row)
    pairing.require(len(grouped) == 24, "anchor count")
    output = []
    for anchor_index in sorted(grouped):
        rows = grouped[anchor_index]
        pairing.require(len(rows) == 42, "probe count")
        # A planning result must not silently change across repeated probe rows.
        prefix = {key: value for key, value in rows[0].items()
                  if not key.startswith("common_") and key not in ("probe", "probe_sha256")}
        for row in rows[1:]:
            pairing.require(all(row[key] == value for key, value in prefix.items()),
                            "repeated prefix or planning fields differ")
        statuses = Counter(row["common_status"] for row in rows)
        prefix["common_status_counts_json"] = json.dumps(dict(sorted(statuses.items())), sort_keys=True)
        prefix["common_probe_count"] = len(rows)
        for metric in pairing.FIELDS:
            for family in ("common_endpoint", "common_delta_own_zero",
                           "common_endpoint_delta_zero_prefix", "common_response_delta_zero_prefix"):
                values = [row[f"{family}_{metric}"] for row in rows
                          if row[f"{family}_{metric}"] is not None]
                pairing.require(all(math.isfinite(value) for value in values), "nonfinite metric")
                key = f"{family}_{metric}"
                prefix[key + "_observed_count"] = len(values)
                prefix[key + "_missing_count"] = len(rows) - len(values)
                prefix[key + "_min"] = min(values) if values else None
                prefix[key + "_median"] = statistics.median(values) if values else None
                prefix[key + "_max"] = max(values) if values else None
        output.append(prefix)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.pairing), Path(args.output)
    pairing.require(output.is_absolute() and output.parent == pairing.ROOT / "results"
                    and output.name.startswith(NAME + "_") and output.suffix == ".csv"
                    and not output.exists() and not output.is_symlink(), "exclusive output")
    rows = summarize(pairing.read_checked(source, args.sha256))
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    pairing.require(hashlib.sha256(source.read_bytes()).hexdigest() == args.sha256, "source drift")
    data = text.getvalue().encode()
    with output.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"output": str(output), "rows": len(rows), "source_sha256": args.sha256,
                      "sha256": hashlib.sha256(data).hexdigest()}))


if __name__ == "__main__":
    main()
