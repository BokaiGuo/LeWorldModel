"""Physical execution of the fixed multi-start CUDA winners; known-root exploration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from . import codex_p1_cp5_physical_24_20261030_v1 as old
from . import codex_p1_cp5_terminal_rgb_gap_20261030_main as gap
from . import codex_p1_multistart_cp5_20261030_main as planner

NAME = "codex_p1_multistart_physical_20261030_main"
PLAN = old.ROOT / "results/codex_p1_multistart_cp5_20261030_main_RUN01.json"
PLAN_SHA = "7e14227760f52eee3d7aad232b09c22fa43436dc90ae17fc6f9c58b4256e05f0"
OLD_PREP = old.ROOT / "results/codex_p1_cp5_physical_24_20261030_v1_prepare_CODEX01.json"
OLD_SHA = "367091192c51c3be2565ee8c7d811cde164c155a4720c0c19531836c8b3fb737"
need = old.need


def build():
    plan, _ = gap.read(PLAN, PLAN_SHA)
    prior, _ = gap.read(OLD_PREP, OLD_SHA)
    blind, _ = gap.read(planner.base.BUNDLE, planner.base.BUNDLE_SHA)
    original = old.h10.design.verify(old.h10.INPUT, old.h10.INPUT_SHA)
    need(plan["status"] == "complete" and len(plan["rows"]) == len(prior["roster"])
         == len(blind["rows"]) == len(original["anchors"]) == 24, "complete roster")
    roster = []
    for i, (row, prev, input_, anchor) in enumerate(zip(
            plan["rows"], prior["roster"], blind["rows"], original["anchors"], strict=True)):
        need(row["index"] == prev["index"] == input_["index"] == anchor["index"] == i
             and row["role"] == prev["role"] == input_["role"] == anchor["role"]
             and row["logical_pair"] == prev["logical_pair"] == input_["logical_pair"] == anchor["logical_pair"]
             and prev["prefix5_float32_hex"] == input_["prefix_raw_f32_hex"] == anchor["prefix5_float32_hex"]
             and prev["cp5_pixel_sha256"] == input_["cp5_pixel_sha256"] == anchor["checkpoint"]["pixel_sha256"],
             "anchor identity")
        actions = np.asarray(row["selected_actions"], dtype="<f4")
        need(actions.shape == (2, 5, 2) and np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
             and hashlib.sha256(actions.tobytes()).hexdigest() == row["selected_action_sha256"]
             and row["selected_index"] == planner.pick(row["candidate_scores"]), "action binding")
        item = dict(prev)
        item.update(raw10_float32_hex=actions.tobytes().hex(), raw10_sha256=row["selected_action_sha256"])
        roster.append(item)
    bindings = dict(prior["bindings"])
    for path in (PLAN, OLD_PREP, planner.base.BUNDLE, Path(__file__).resolve(), Path(planner.__file__).resolve()):
        bindings[str(path)] = old.sha(path)
    need(all(old.sha(path) == digest for path, digest in bindings.items()), "input drift")
    return {"schema": NAME+".prepare", "roster": roster, "fixed_rows": 24, "bindings": bindings,
            "plan_sha256": PLAN_SHA, "caps_per_row": old.CAPS, "known_roots": True,
            "P1_admission": False, "independent_confirmation": False, "fairness_proven": False}


def run(output):
    prepared = build()
    output = Path(output)
    need(output.is_absolute() and output.parent == old.ROOT/"results" and
         output.name.startswith(NAME+"_RUN") and not os.path.lexists(output) and
         shutil.disk_usage(output.parent).free >= 2*1024**3, "exclusive output/disk")
    h10 = old.h10
    source_report = h10.native.read(h10.old.prep.DIRECTORY/"report.json", cap=64*1024**2)
    _, data = h10.runner.physical.verify(source_report["execution_binding"]["prepared"], h10.old.prep.PREP_SHA)
    h10.native.constructor_contract()
    anchors = h10.native.read(h10.INPUT, cap=64*1024**2)["anchors"]
    output.mkdir(mode=0o700)
    prepare_sha = h10._write_new(output/"prepare.json", prepared)
    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)
    shards, failure = [], None
    try:
        for anchor, item in zip(anchors, prepared["roster"], strict=True):
            folder = output/f"row_{anchor['index']:02d}"
            report = old.execute(anchor, item, data, folder, factory)
            shards.append({"index": anchor["index"], "report_sha256": old.sha(folder/"report.json"),
                           "status": report["status"]})
            if report["status"] != "recorded_not_consumed":
                failure = {"index": anchor["index"], "errors": report["errors"]}
                break
        need(all(old.sha(path) == digest for path, digest in prepared["bindings"].items()), "source drift")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    result = {"schema": NAME+".run", "status": "recorded_not_consumed" if len(shards) == 24 and not failure
              else "partial_or_failed", "shards": shards, "failure": failure, "fixed_rows": 24,
              "prepare_sha256": prepare_sha, "P1_admission": False, "independent_confirmation": False}
    digest = h10._write_new(output/"run_report.json", result)
    print(json.dumps({"status": result["status"], "sha256": digest, "rows": len(shards),
                      "failure": failure}), flush=True)
    return 0 if failure is None and len(shards) == 24 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print("Execute all 24 fixed multi-start winners; no new inference.", flush=True)
    raise SystemExit(run(args.output))
