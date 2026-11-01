"""Fixed all-eight-prefix intervention on the new opposite-side task."""
import argparse
import json
from pathlib import Path

from . import p1_candidate_intervention_20261101_main as engine
from . import p1_reverse_control_20261101_main as control
from . import p1_normalized_exchange_saved_20261101_main as saved

NAME = "p1_reverse_intervention_20261101_main"
DOC = control.ROOT.parent / "docs/D2P_P1_REVERSE_INTERVENTION_PROTOCOL_20261101_MAIN.md"


def run(output):
    path = Path(__file__).resolve()
    before = engine.engine.sha(path)
    engine.NAME, engine.DOC = NAME, DOC
    engine.DIRECTORY = control.ROOT / "results/p1_reverse_control_20261101_main_structured_RUN_MAIN01"
    engine.PREPARED = control.ROOT / "results/p1_reverse_control_20261101_main_structured_PREP_MAIN01.json"
    engine.source, engine.saved = control, saved
    result = engine.run(output)
    if engine.engine.sha(path) != before:
        raise ValueError("driver drift")
    with (Path(output) / "driver_binding.json").open("x") as stream:
        json.dump(dict(driver=str(path), driver_sha256=before,
                       report_sha256=engine.engine.sha(Path(output) / "report.json"),
                       prepared_sha256=engine.engine.sha(Path(output) / "prepared.json"),
                       P1_admission=False), stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output) else 1)
