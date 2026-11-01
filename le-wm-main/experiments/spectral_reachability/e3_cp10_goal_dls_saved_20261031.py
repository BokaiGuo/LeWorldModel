"""Verify saved physical ledger and independently recompute goal-DLS banks."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from . import e3_cp10_goal_dls_20261031 as experiment
from . import p1_four_arm_extra32_budget_20261031 as control

NAME = "e3_cp10_goal_dls_saved_20261031"


def check(root, directory, report_sha):
    original = experiment.setup(root)
    saved = importlib.import_module(f".e3_cp10_four_arm_root{root}_saved_20261031", __package__)
    base = saved.check(directory, report_sha)
    report = json.loads((Path(directory) / "report.json").read_bytes())
    _, target, _ = original.inputs()
    goal = np.asarray(target["latent"], dtype=np.float64)
    checked = 0
    for row in report["rows"]:
        if row["arm"] != "structured":
            continue
        for node in report["nodes"][row["episode_id"]]:
            if node["status"] != "sealed":
                continue
            trace = Path(node["receipt"]["trace"])
            events = [json.loads(line) for line in trace.read_bytes().splitlines()]
            signal = next(x["payload"] for x in events if x["kind"] == "spectral_signal")
            initial = next(x["payload"] for x in events if x["kind"] == "initial_candidates")
            actual = np.asarray(initial["raw"], dtype=np.float32)
            expected, _ = experiment.goal_bank(signal["jacobian"], signal["projection"],
                                                goal, np.asarray(signal["nominal"]).reshape(192))
            if not np.array_equal(actual, expected):
                raise ValueError("saved goal-DLS initializer differs from recomputation")
            checked += 1
    if not checked or control.EXTRA_ARMS != ("structured", "native", "random"):
        raise ValueError("missing structured nodes or refinement policy")
    return {"schema": NAME, "root": root, "report_sha256": report_sha,
            "saved_ledger_nodes": base["nodes"], "goal_dls_nodes": checked,
            "physical_event_chain_independently_checked": False, "fairness_proven": False,
            "P1_admission": False}


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
