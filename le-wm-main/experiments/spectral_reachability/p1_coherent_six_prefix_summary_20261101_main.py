"""Recompute matched physical endpoints for six known-root prefix controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_coherent_intervention_20261101_main as common

ROOT = common.panel.ROOT
PREFIXES = (*common.panel.ARMS, "zero", "same_bank_greedy")
FILES = (
    "p1_coherent_intervention_20261101_main_RUN_MAIN01",
    "p1_samebank_greedy_physical_20261101_main_RUN_MAIN01",
    "p1_coherent_own_physical_20261101_main_RUN_MAIN01",
    "p1_coherent_samebank_own_physical_20261101_codex_RUN_MAIN01",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, text):
    if not condition:
        raise ValueError(text)


def read_report(name, goal, root):
    path = ROOT / "results" / name / "report.json"
    value = json.loads(path.read_bytes())
    protocol = path.parent / "protocol.json"
    require(sha(protocol) == value["protocol_sha256"], "protocol identity")
    schedule = json.loads(protocol.read_bytes())["schedule"]
    require(len(schedule) == len(value["rows"]), "all scheduled rows")
    sources = dict(value["sources"])
    require(all(sha(p) == h for p, h in sources.items()), "source closure")
    sources.update({str(path): sha(path), str(protocol): sha(protocol)})
    for row, slot in zip(value["rows"], schedule, strict=True):
        require(row["prefix"] == slot["prefix"] and not row.get("error"), "valid scheduled row")
        actions = np.asarray(row["executed_raw"], np.float32)
        planned = np.asarray(slot["commands"], np.float32)
        require(actions.shape == (row["steps"], 2)
                and np.array_equal(actions, planned[:row["steps"]]), "actual scheduled actions")
        # This summary has a fixed-step endpoint; never pad an early stop.
        require(row["status"] == "completed20" and row["steps"] == 20, "observed full endpoint")
        require([c["step"] for c in row["checkpoints"]] == [0, 5, 10, 15, 20],
                "complete checkpoint roster")
        require(np.array_equal(row["checkpoints"][0]["state"], root), "same root")
        for frame in row["checkpoints"]:
            require(sha(frame["png"]) == frame["file_sha256"], "PNG file hash")
            with Image.open(frame["png"]) as image:
                pixels = np.asarray(image)
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"],
                    "actual pixels")
            native = common.metrics.qualification.native(goal, frame["state"])
            recalculated = dict(**common.metrics.metrics(frame["state"], goal),
                               joint_xy=native["joint_xy_error"],
                               native_success=native["initial_success"])
            require(recalculated == frame["metrics"], "independent physical metric recomputation")
            sources[frame["png"]] = frame["file_sha256"]
    return value["rows"], sources


def run(output):
    output = Path(output)
    require(output.is_absolute() and output.parent == ROOT / "results" and not output.exists(),
            "exclusive output")
    goal, _, sources = common.target.collect()
    reports = []
    for name in FILES:
        rows, bindings = read_report(name, goal["goal_state"], goal["root_state"])
        reports.append(rows)
        sources.update(bindings)
    interventions = reports[0] + reports[1]
    own = reports[2] + reports[3]
    require([r["prefix"] for r in own] == list(PREFIXES), "six own plans")
    rows = []
    for name, planned in zip(PREFIXES, own, strict=True):
        arms = [r for r in interventions if r["prefix"] == name]
        require(len(arms) == 2, "two interventions per prefix")
        zero = [r for r in arms if r["suffix"] == "zero"]
        shared = [r for r in arms if r["suffix"] in ("common", "common_structured")]
        require(len(zero) == len(shared) == 1, "common/zero identities")
        zero, shared = zero[0], shared[0]
        start, cp5 = planned["checkpoints"][:2]
        for peer in (zero, shared):
            require(peer["executed_raw"][:5] == planned["executed_raw"][:5]
                    and peer["checkpoints"][1]["state"] == cp5["state"]
                    and peer["checkpoints"][1]["pixel_sha256"] == cp5["pixel_sha256"],
                    "same actual prefix across interventions and own planning")
        last = lambda row: row["checkpoints"][-1]["metrics"]
        a, z, o = last(shared), last(zero), last(planned)
        rows.append(dict(prefix=name,
            prefix_delta={k: cp5["metrics"][k]-start["metrics"][k]
                          for k in ("object_xy", "object_angle", "pusher_xy", "joint_xy")},
            cp5=cp5["metrics"], common_terminal=a, zero_terminal=z, own_terminal=o,
            common_minus_zero_joint_xy=a["joint_xy"]-z["joint_xy"],
            own_minus_zero_joint_xy=o["joint_xy"]-z["joint_xy"],
            status=planned["status"], steps=planned["steps"]))
    structured = rows[0]["own_terminal"]["joint_xy"]
    for row in rows:
        row["own_minus_structured_joint_xy"] = row["own_terminal"]["joint_xy"]-structured
    sources[str(Path(__file__).resolve())] = sha(__file__)
    require(all(sha(p) == h for p, h in sources.items()), "final source closure")
    result = dict(rows=rows, sources=sources, neural_calls=0,
        unique_development_root_count=1, independent_confirmation=False, P1_admission=False,
        limitations=[
            "one known translated development root; correlated branches are not independent samples",
            "equal conditional suffix node costs do not establish equal complete-controller cost",
            "common suffix is structured actual tail, not a symmetric suffix family",
            "own suffix is open-loop15 after one real cp5 RGB refresh",
            "model immediate greedy is not guaranteed physically greedy",
            "no certified strong-baseline or independent-root superiority"])
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(output), sha256=sha(output), rows=rows)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
