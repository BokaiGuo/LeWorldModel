"""Produce paired split-budget crossover CSV, without imputing stopped branches."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from . import p1_crossover_paired_table_20261031 as paired
from . import p1_split_budget_crossover_20261031 as source

NAME = "p1_split_budget_crossover_table_20261031"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source.need(args.output.is_absolute() and args.output.parent == source.ROOT / "results"
                and args.output.name.startswith(NAME + "_") and not args.output.exists()
                and not args.output.is_symlink(), "exclusive output")
    raw = args.input.read_bytes()
    source.need(hashlib.sha256(raw).hexdigest() == args.sha256, "input hash")
    value = json.loads(raw)
    source.need(value["schema"] == source.NAME + ".consume", "schema")
    rows = paired.build(value)
    with args.output.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "sha256": source.sha(args.output)}), flush=True)


if __name__ == "__main__":
    main()
