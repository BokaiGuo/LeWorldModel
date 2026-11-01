"""All four frozen H4 outcomes, checked against external execution receipts."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from . import p1_reverse_analysis_20261101_main as analysis
from . import p1_saved_h4_parfit_20261101 as saved

PREFIX = "p1_reverse_h4_parfit_20261101"
HASHES = dict(
    structured="2ff3d42d93f2f5ecf34bc675a9f6e219e7a9ab2d4a1c0742c7ce3ed3f9b14bff",
    native="1c911c8f430a6753de6d4e6df4264c72a0850f97b436cd37188bec9ea66e3267",
    random="8fa32c87f2a87e0fd92f1411ecb6389dd479770c66e0a41756ee740ddd7d6300",
    immediate_progress="017188c71ffbe1995b697a0d596dafd1c0265d4410e08579ec13df52e7f393b4",
)


def check(directory):
    arm = next(a for a in HASHES if Path(directory).name == f"{PREFIX}_{a}_RUN_MAIN01")
    return saved.check(directory, HASHES[arm])


def run(output):
    sources = {str(Path(p).resolve()): saved.sha(p)
               for p in (__file__, saved.__file__, analysis.__file__)}
    analysis.PREFIX = PREFIX
    analysis.saved = SimpleNamespace(check=check)
    result = analysis.analyze()
    if any(saved.sha(p) != h for p, h in sources.items()):
        raise ValueError("analysis source drift")
    result["sources"].update(sources)
    result["external_report_sha256"] = HASHES
    result["variant"] = "H4_H3_H2_H1"
    with Path(output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps([dict(arm=r["arm"], status=r["status"], success=r["native_success"],
                           terminal=r["terminal_observed"]) for r in result["rows"]]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.output)
