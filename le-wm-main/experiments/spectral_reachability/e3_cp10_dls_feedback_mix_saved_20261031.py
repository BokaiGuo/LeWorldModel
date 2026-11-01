"""Validate fixed feedback-mix candidates and the physical action ledger."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from . import e3_cp10_dls_feedback_mix_20261031 as experiment
from . import e3_cp10_goal_dls_20261031 as dls

NAME = "e3_cp10_dls_feedback_mix_saved_20261031"


def check(root, directory, report_sha):
    original = experiment.setup(root)
    saved = importlib.import_module(f".e3_cp10_four_arm_root{root}_saved_20261031", __package__)
    base = saved.check(directory, report_sha)
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_bytes())
    _, target, _ = original.inputs()
    source_sha = original.runner.sha(Path(experiment.__file__).resolve())
    goal = np.asarray(target["latent"], dtype=np.float64)
    count = 0
    for row in report["rows"]:
        if row["arm"] != "structured":
            continue
        for node in report["nodes"][row["episode_id"]]:
            if node["status"] != "sealed":
                continue
            boundary = node["boundary"]
            with Path(node["receipt"]["ledger"]["journal"]).open("rb") as stream:
                header = json.loads(stream.readline())["payload"]
            if header["contract"]["arms"]["structured"]["initializer_sha256"] != source_sha:
                raise ValueError("wrong initializer source")
            events = [json.loads(line) for line in Path(node["receipt"]["trace"]).read_bytes().splitlines()]
            signal = next(e["payload"] for e in events if e["kind"] == "spectral_signal")
            initial = next(e["payload"] for e in events if e["kind"] == "initial_candidates")
            expected, _ = dls.goal_bank(signal["jacobian"], signal["projection"], goal,
                                        np.asarray(signal["nominal"]).reshape(192))
            if boundary:
                expected = experiment.mix_bank(expected, boundary)
            if not np.array_equal(expected, np.asarray(initial["raw"], dtype=np.float32)):
                raise ValueError("initializer does not match fixed policy")
            count += 1
    if not count:
        raise ValueError("missing structured nodes")
    return {"schema": NAME, "root": root, "report_sha256": report_sha,
            "nodes": base["nodes"], "checked_structured_nodes": count,
            "initializer_sha256": source_sha, "physical_event_chain_independently_checked": False,
            "fairness_proven": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    for key in ("directory", "report-sha", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    root = Path(__file__).resolve().parents[2]
    if (not output.is_absolute() or output.parent != root / "results"
            or not output.name.startswith(NAME + "_") or output.exists() or output.is_symlink()):
        raise ValueError("exclusive output")
    result = check(args.root, args.directory, args.report_sha)
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with output.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError("short write")
    print(json.dumps({"status": "verified", "nodes": result["nodes"],
                      "sha256": original_sha(output)}, sort_keys=True))


def original_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
