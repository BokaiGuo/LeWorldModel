"""New development target, with unchanged physical criteria plus RGB margins."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_reverse_witness_20261101_main as original
from . import p1_rgb_visibility_20261101_main as visibility

NAME = "p1_reverse_witness_v4_20261101_main"
DOC = original.ROOT.parent / "docs/D2P_P1_VISIBLE_TARGET_PROTOCOL_20261101_MAIN.md"


def run(output):
    paths = [Path(__file__).resolve(), Path(visibility.__file__).resolve(), DOC]
    sources = {str(p): original.physical.sha(p) for p in paths}
    original.NAME, original.DOC = NAME, DOC
    original.PHASES = ((130, -100, 4), (-100, -110, 6), (-70, -70, 3),
                       (20, 55, 4), (130, 80, 3))
    original.run(output)
    output = Path(output)
    report = json.loads((output / "report.json").read_bytes())
    with Image.open(report["rows"][-1]["png"]) as image:
        screen = visibility.inspect(np.asarray(image))
    admitted = report["task_eligible"] and screen["eligible"]
    if any(original.physical.sha(p) != h for p, h in sources.items()):
        raise ValueError("driver source drift")
    record = dict(sources=sources, report_sha256=original.physical.sha(output / "report.json"),
                  protocol_sha256=original.physical.sha(output / "protocol.json"),
                  phases=original.PHASES, visual_screen=screen, task_admitted=admitted,
                  P1_admission=False, independent_confirmation=False)
    with (output / "visual_admission.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)
    return admitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
