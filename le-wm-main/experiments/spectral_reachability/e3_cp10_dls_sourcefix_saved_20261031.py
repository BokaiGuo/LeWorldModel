"""Check goal-DLS action bindings and the initializer SHA in every node."""
import argparse
import hashlib
import json
from pathlib import Path

from . import e3_cp10_goal_dls_20261031 as dls
from . import e3_cp10_goal_dls_saved_20261031 as prior

NAME = "e3_cp10_dls_sourcefix_saved_20261031"


def check(root, directory, report_sha):
    base = prior.check(root, directory, report_sha)
    report = json.loads((Path(directory) / "report.json").read_bytes())
    source_sha = hashlib.sha256(Path(dls.__file__).read_bytes()).hexdigest()
    count = 0
    for row in report["rows"]:
        for node in report["nodes"][row["episode_id"]]:
            if node["status"] != "sealed":
                continue
            ledger = Path(node["receipt"]["ledger"]["journal"])
            with ledger.open("rb") as stream:
                header = json.loads(stream.readline())["payload"]
            actual = header["contract"]["arms"]["structured"]["initializer_sha256"]
            if actual != source_sha:
                raise ValueError("structured initializer is not bound to DLS source")
            count += 1
    if count != base["saved_ledger_nodes"]:
        raise ValueError("source contract count differs from verified nodes")
    return {**base, "schema": NAME, "structured_initializer_sha256": source_sha,
            "source_bound_nodes": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    root = Path(__file__).resolve().parents[2]
    if (not output.is_absolute() or output.parent != root / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("output must be an exclusive result path")
    result = check(args.root, args.directory, args.report_sha)
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with output.open("xb") as stream:
        if stream.write(encoded) != len(encoded):
            raise OSError("short write")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
