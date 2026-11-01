"""Full frozen P5 validator with a bounded reader for its larger actual journal.

No producer mutation, physics rerun, row filtering or scientific gate changes.
The inherited 8MiB reader predates the full callback meter; use the producer's
already sealed 96MiB payload partition as a saved-input bound instead.
"""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p5_contact_action_words_physical_20261028_v1 as producer

JOURNAL_CAP = 96 * 1024**2


def read_journal(path):
    path = Path(path)
    producer.sha(path)
    producer.require(path.stat().st_size <= JOURNAL_CAP, "sealed payload journal bound")
    rows, previous = [], None
    with path.open("rb") as stream:
        for sequence, line in enumerate(stream):
            producer.require(line.endswith(b"\n"), "complete journal line")
            row = json.loads(line)
            h = row.pop("entry_sha256")
            producer.require(row["sequence"] == sequence and row["previous_sha256"] == previous
                             and producer.digest(row) == h, "journal chain")
            previous = h
            rows.append(row["event"])
    return rows


class ReaderBridge:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    read_journal = staticmethod(read_journal)


def consume(manifest, manifest_sha, report, report_sha):
    value = producer.read_json(manifest, manifest_sha)
    producer.verify(value)
    contract, _, _, physical = producer.helpers()
    result = producer.validate_saved(value, manifest_sha, report, report_sha,
                                    ReaderBridge(physical), contract.load(contract.GEOMETRY))
    # Recheck closure after the unchanged full validator has inspected all rows.
    producer.verify(value)
    producer.require(producer.sha(report) == report_sha, "final report drift")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "manifest-sha", "report", "report-sha", "receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    target = Path(args.receipt).absolute()
    producer.require(not target.exists() and not any(p.is_symlink() for p in (target, *target.parents)), "new regular receipt")
    result = consume(args.manifest, args.manifest_sha, args.report, args.report_sha)
    producer.helpers()[1].save_json(target, result)
    print(json.dumps(dict(status=result["status"], receipt=str(target), sha256=producer.sha(target),
                          rows=len(result["private_suffix_frame_rows"] or []), denominator=result["denominator"],
                          scientific_admission=False), sort_keys=True))


if __name__ == "__main__":
    main()
