"""Persist the unchanged full common-controller strict receipt; compact stdout."""
import argparse
import os
from pathlib import Path
import shutil

from experiments.spectral_reachability import agent_experiment1_common_controller_20261028_v1 as source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    target = Path(args.receipt).absolute()
    source.require(not target.exists() and not target.is_symlink(), "new receipt only")
    result = source.consume(args.output, args.report_sha)
    raw = source.canonical(result)
    source.require(len(raw) <= source.FINAL, "bounded full strict receipt")
    source.require(shutil.disk_usage(target.parent).free >= source.FLOOR + len(raw), "receipt floor")
    with target.open("xb") as stream:
        source.require(stream.write(raw) == len(raw), "complete full receipt write")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(source.canonical(dict(status=result["status"], receipt=str(target),
        sha256=source.sha(target), plans=len(result["plans"]),
        scientific_slots=result["scientific_slots"], learned_unknown_slots=result["learned_unknown_slots"],
        predictor_samples=result["predictor_samples"], P1_admission=False)).decode())


if __name__ == "__main__":
    main()
