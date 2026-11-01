"""Encode every fixed reverse-task intervention frame for same-time diagnosis."""
import argparse
import json
from pathlib import Path

from . import p1_candidate_rgb_cuda_20261101_main as encoder
from . import p1_reverse_control_20261101_main as control

NAME = "p1_reverse_intervention_rgb_20261101_main"
DIRECTORY = control.ROOT / "results/p1_reverse_intervention_20261101_main_run_MAIN01"
ORIGINAL_ANALYZE = encoder.analysis.analyze


def analyze(directory):
    directory = Path(directory)
    binding_path = directory / "driver_binding.json"
    binding = json.loads(binding_path.read_bytes())
    sha = encoder.source.base.sha
    if (sha(binding["driver"]) != binding["driver_sha256"]
            or sha(directory / "report.json") != binding["report_sha256"]
            or sha(directory / "prepared.json") != binding["prepared_sha256"]):
        raise ValueError("intervention driver binding")
    result = ORIGINAL_ANALYZE(directory)
    result["sources"][str(binding_path)] = sha(binding_path)
    result["sources"][binding["driver"]] = binding["driver_sha256"]
    return result


def run(output):
    encoder.NAME, encoder.DIRECTORY = NAME, DIRECTORY
    encoder.q.METRIC, encoder.q.METRIC_SHA = control.METRIC, control.METRIC_SHA
    encoder.analysis.analyze = analyze
    original = encoder.source.base.bindings

    def bindings():
        value = original()
        path = Path(__file__).resolve()
        value[str(path)] = encoder.source.base.sha(path)
        return value

    encoder.source.base.bindings = bindings
    encoder.run(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    run(args.output)
