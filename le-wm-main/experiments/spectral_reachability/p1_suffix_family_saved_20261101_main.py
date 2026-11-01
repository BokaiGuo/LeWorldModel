"""Check all 42 physical interventions; report the whole fixed tail matrix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_coherent_suffix_family_20261101_leibniz as family
from . import p1_coherent_intervention_20261101_main as common


def run(directory, output):
    directory, output = Path(directory).resolve(), Path(output)
    report_path, protocol_path = directory / "report.json", directory / "protocol.json"
    report = json.loads(report_path.read_bytes())
    protocol = json.loads(protocol_path.read_bytes())
    family.require(family.sha(protocol_path) == report["protocol_sha256"], "protocol SHA")
    family.validate_schedule(protocol["schedule"])
    sources = protocol["sources"].copy()
    family.verify_sources(sources)
    family.require([(r["prefix"], r["tail"]) for r in report["rows"]] ==
                   [(s["prefix"], s["tail"]) for s in protocol["schedule"]], "all 42 branches")
    goal, _, extra = common.target.collect()
    family.merge_sources(sources, extra)
    records = []
    for row, slot in zip(report["rows"], protocol["schedule"], strict=True):
        family.require(row["status"] == "completed20" and not row.get("error")
                       and row["steps"] == row["steps_attempted"] == row["steps_returned"] == 20,
                       "full observed endpoint; never impute censored endpoints")
        family.require(np.array_equal(np.asarray(row["executed_raw"], np.float32),
                                      np.asarray(slot["commands"], np.float32)), "actual commands")
        family.require([f["step"] for f in row["checkpoints"]] == [0, 5, 10, 15, 20], "all frames")
        for frame in row["checkpoints"]:
            family.require(family.sha(frame["png"]) == frame["file_sha256"], "PNG file")
            with Image.open(frame["png"]) as im:
                pixels = np.asarray(im)
            family.require(hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"],
                           "RGB bytes")
            family.require(common.panel.saved.spec.digest(frame["state"]) == frame["state_sha256"],
                           "state bytes")
            native = common.metrics.qualification.native(goal["goal_state"], frame["state"])
            metrics = dict(**common.metrics.metrics(frame["state"], goal["goal_state"]),
                           joint_xy=native["joint_xy_error"], native_success=native["initial_success"])
            family.require(metrics == frame["metrics"], "physical metric recomputation")
            sources[frame["png"]] = frame["file_sha256"]
        family.require(np.array_equal(row["checkpoints"][0]["state"], goal["root_state"]), "root")
        cp5 = row["checkpoints"][1]
        family.require(cp5["pixel_sha256"] == slot["cp5"]["pixel_sha256"]
                       and cp5["state_sha256"] == slot["cp5"]["state_sha256"], "actual prefix identity")
        records.append(dict(prefix=row["prefix"], tail=row["tail"], status=row["status"],
                            terminal=row["checkpoints"][-1]["metrics"]))
    matrix = [[next(r["terminal"]["joint_xy"] for r in records
                    if r["prefix"] == p and r["tail"] == t) for t in family.TAILS]
              for p in family.PREFIXES]
    array = np.asarray(matrix)
    effect = array[:, 1:]-array[:, :1]
    comparison = {}
    for control in ("zero", "same_bank_greedy", "native"):
        i = family.PREFIXES.index(control)
        comparison[control] = dict(
            structured_minus_control_absolute=(array[0, 1:]-array[i, 1:]).tolist(),
            structured_minus_control_zero_adjusted=(effect[0]-effect[i]).tolist())
    for path in (report_path, protocol_path, Path(__file__).resolve()):
        sources[str(path)] = family.sha(path)
    family.verify_sources(sources)
    family.require(output.is_absolute() and output.parent == family.ROOT / "results"
                   and not output.exists(), "exclusive output")
    result = dict(prefixes=family.PREFIXES, tails=family.TAILS, joint_xy=matrix,
                  tail_minus_own_zero=effect.tolist(), comparisons=comparison, rows=records,
                  sources=sources, verified=True, P1_admission=False, independent_confirmation=False,
                  unique_root_count=1, neural_calls=0,
                  limitations="descriptive known-root finite-tail intervention, no independent samples, no oracle tail selection")
    family.write_once(output, result)
    print(json.dumps(dict(joint_xy=matrix, comparisons=comparison,
                          successes=sum(r["terminal"]["native_success"] for r in records),
                          sha256=family.sha(output))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.directory, args.output)
