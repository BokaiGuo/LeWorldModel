"""Replay fixed hybrid prefixes and their independently planned CUDA suffixes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import p1_hybrid_h10_20261031 as h
from . import p1_hybrid_cp5_bundle_20261031 as bundle
from . import codex_p1_cp5_physical_24_20261030_v1 as engine
from . import codex_p1_multistart_cp5_20261030_main as common

ROOT, need, sha = h.ROOT, h.need, h.sha
NAME = "p1_hybrid_cp5_physical_20261031"
BUNDLE = ROOT / "results/p1_hybrid_cp5_bundle_20261031_RUN01.json"
BUNDLE_SHA = "5699cbf775af200597ab33677d4460bed56cd34623e34809a0280daa1146d93e"
PLAN = ROOT / "results/p1_hybrid_cp5_adam_20261031_RUN01.json"
PLAN_SHA = "2d4021106fcb5394d3b481268f3f193b918b47bb46ccee859b83ad1df133979d"


def build():
    need(sha(BUNDLE) == BUNDLE_SHA and sha(PLAN) == PLAN_SHA, "pinned inputs")
    inputs, plan = h.read(BUNDLE), h.read(PLAN)
    need(inputs == bundle.build() and plan["status"] == "complete" and
         plan["physical_execution"] is False and plan["planned_rows"] == 2, "plan contract")
    old = h.read(bundle.PREPARE)
    bindings = dict(plan["sources"])
    for path in (BUNDLE, PLAN, Path(__file__).resolve(), Path(engine.__file__).resolve()):
        bindings[str(path)] = sha(path)
    items = []
    for anchor, inp, row in zip(old["anchors"], inputs["rows"], plan["rows"], strict=True):
        need(row["status"] == "planned" and row["logical_pair"] == anchor["logical_pair"]
             and row["cp5_pixel_sha256"] == anchor["checkpoint"]["pixel_sha256"]
             and row["prefix_raw_f32_hex"] == anchor["prefix5_float32_hex"]
             and inp["blind"]["cp5_pixel_sha256"] == row["cp5_pixel_sha256"], "plan input identity")
        init = common.initial_actions(anchor["logical_pair"])
        need(hashlib.sha256(init.tobytes()).hexdigest() == row["initial_f32_sha256"]
             and common.pick(row["candidate_scores"]) == row["selected_index"], "optimizer identity")
        actions = np.asarray(row["actions"], dtype="<f4")
        need(actions.shape == (2, 5, 2) and np.isfinite(actions).all()
             and (np.abs(actions) <= 1).all() and
             hashlib.sha256(actions.tobytes()).hexdigest() == row["actions_sha256"], "action bytes")
        item = {k: anchor[k] for k in ("index", "logical_pair", "root_slot", "condition", "arm", "role",
                                      "task_id", "prefix5_float32_hex")}
        item.update(raw10_float32_hex=actions.tobytes().hex(), raw10_sha256=row["actions_sha256"])
        items.append(item)
    need(len(items) == 2 and all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME + ".prepare", "anchors": old["anchors"], "items": items,
            "bindings": bindings, "fixed_rows": 2, "P1_admission": False,
            "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--authorize-physics", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results" and
         output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    fixed = build()
    if args.command == "prepare":
        print(json.dumps({"sha256": h.engine._write_new(output, fixed), "fixed_rows": 2}))
        return
    need(args.authorize_physics and args.prepared and sha(args.prepared) == args.sha256
         and h.read(args.prepared) == fixed and shutil.disk_usage(ROOT / "results").free >= 2 * 1024**3,
         "prepare, physics authorization, disk")
    _, data = h.existing.physical.bundle.physical.pilot.material(h.existing.physical.bundle.physical.JOIN_SHA)
    h.existing.physical.bundle.physical.prior.native.constructor_contract()
    output.mkdir(mode=0o700)

    def factory():
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        return gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224, with_target=False)

    shards = []
    with patch.object(h.engine.prior, "compare_trace", h.compare_trace):
        for anchor, item in zip(fixed["anchors"], fixed["items"], strict=True):
            path = output / f"row_{anchor['index']:02d}"
            report = engine.execute(anchor, item, data, path, factory)
            shards.append({"index": anchor["index"], "status": report["status"], "sha256": sha(path / "report.json")})
            print(json.dumps(shards[-1]), flush=True)
            if report["status"] != "recorded_not_consumed":
                break
    need(all(sha(path) == digest for path, digest in fixed["bindings"].items()), "source drift")
    status = ("recorded_not_consumed" if len(shards) == 2 and
              all(s["status"] == "recorded_not_consumed" for s in shards) else "partial_or_failed")
    result = {"schema": NAME + ".run", "status": status, "shards": shards,
              "prepare_sha256": args.sha256, "fixed_rows": 2, "P1_admission": False}
    print(json.dumps({"status": status, "sha256": h.engine._write_new(output / "run_report.json", result)}),
          flush=True)


if __name__ == "__main__":
    main()
