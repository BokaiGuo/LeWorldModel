"""Change the scripted task-generator contact direction, not eligibility."""
import argparse
import json
from pathlib import Path

from . import p1_reverse_witness_20261101_main as original

NAME = "p1_reverse_witness_v3_20261101_main"
DOC = original.ROOT.parent / "docs/D2P_P1_REVERSE_WITNESS_V3_PROTOCOL_20261101_MAIN.md"


def run(output):
    path = Path(__file__).resolve()
    before = original.physical.sha(path)
    original.NAME, original.DOC = NAME, DOC
    original.PHASES = ((130, -100, 4), (-100, -110, 6), (-70, -70, 3),
                       (0, 70, 5), (130, 80, 2))
    result = original.run(output)
    if original.physical.sha(path) != before:
        raise ValueError("driver source drift")
    output = Path(output)
    with (output / "driver_binding.json").open("x") as stream:
        json.dump(dict(driver=str(path), driver_sha256=before,
                       report_sha256=original.physical.sha(output / "report.json"),
                       protocol_sha256=original.physical.sha(output / "protocol.json"),
                       phases=original.PHASES, changed_after_v2=True, P1_admission=False),
                  stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
