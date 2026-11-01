"""Persist the complete unchanged saved E1 physical validator result."""
import argparse
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_experiment1_common_controller_physical_consume_20261028_v1 as validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "report-sha", "receipt"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    target = Path(args.receipt).absolute()
    validator.require(not target.exists() and not any(p.is_symlink() for p in (target, *target.parents)), "exclusive regular receipt")
    result = validator.consume(args.output, args.report_sha)
    payload = validator.physical.raw(result)
    validator.require(len(payload) <= 8 * 1024**2, "full saved receipt bound")
    validator.require(shutil.disk_usage(target.parent).free >= validator.physical.FLOOR + len(payload), "disk floor")
    with target.open("xb") as stream:
        validator.require(stream.write(payload) == len(payload), "complete receipt write")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(validator.source.canonical(dict(status=result["status"], receipt=str(target),
        sha256=validator.sha(target), bytes=len(payload), rows=len(result["rows"]),
        checkpoints=len(result["checkpoints"]), learned_unknown_slots=result["learned_unknown_slots"],
        scientific_admission=False)))


if __name__ == "__main__":
    main()
