"""Four fixed RGB-only control arms on the opposite-side development task."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_normalized_exchange_q_20261101_main as search
from . import p1_reverse_target_cuda_20261101_main as target_definition
from . import p1_reverse_metric_cuda_20261101_main as metric_definition

NAME = "p1_reverse_control_20261101_main"
ROOT = target_definition.ROOT
DOC = ROOT.parent / "docs/D2P_P1_REVERSE_CONTROL_PROTOCOL_20261101_MAIN.md"
Q = search.h.parent
ARMS = Q.parent.ARMS
METRIC = ROOT / "results/p1_reverse_metric_cuda_20261101_main_MAIN01.json"
METRIC_SHA = "f0def100418f9120cf2b6d7f409205a789c37b2f6628acc5a5b4da28f1bed715"


def task_setup(root, arm, target_path, target_sha):
    if root != 0 or arm not in ARMS:
        raise ValueError("fixed root/arm")
    experiment = Q.parent.protected.setup(root)
    runner = experiment.runner
    old_inputs, old_build, old_protocol = experiment.inputs, runner.build, runner.protocol
    old_roster = runner.rgb.roster
    runner.rgb.roster = lambda: [row for row in old_roster() if row["arm"] == arm]

    def inputs():
        entry, _, sources = old_inputs()
        runner.require(runner.sha(target_path) == target_sha, "opposite-side target receipt")
        receipt = json.loads(Path(target_path).read_bytes())
        old_target = json.loads(experiment.TARGET.read_bytes())
        runner.require(receipt["schema"] == target_definition.NAME
                       and receipt["status"] == "complete" and len(receipt["rows"]) == 1
                       and receipt["model_before"] == receipt["model_after"] == old_target["model_before"],
                       "same frozen encoder, explicit new task schema")
        row = receipt["rows"][0]
        runner.require(row["root"] == 0 and row["root_state"] == entry["root_state"]
                       and all(row["task_eligibility"].values()), "eligible same root")
        pixels = np.frombuffer(base64.b64decode(row["pixels_uint8_base64"], validate=True),
                               dtype=np.uint8).reshape(224, 224, 3).copy()
        runner.require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"],
                       "target RGB identity")
        entry.update(pixels=pixels, pixel_sha256=row["pixel_sha256"], goal_state=row["goal_state"])
        for key in ("witness10_float32_hex", "historical_first_success_step",
                    "initial_native_distance", "initial_joint_xy_error"):
            entry.pop(key, None)
        sources.update(receipt["sources"])
        sources[str(Path(target_path).resolve())] = target_sha
        for path in (DOC, Path(__file__).resolve()):
            sources[str(path)] = runner.sha(path)
        return entry, row, sources

    def build():
        value = old_build()
        value["sources"].update(inputs()[2])
        return value

    def protocol():
        value = old_protocol()
        value.update(arms=[arm], panel_roster=list(ARMS), scientific_panel_denominator=4,
                     goal="opposite_side_witness_final_RGB", target_receipt_sha256=target_sha,
                     privileged_task_generator=True, witness_actions_used_in_planner=False,
                     development_only=True, independent_confirmation=False, P1_admission=False)
        return value

    experiment.inputs = inputs
    runner.build, runner.protocol = build, protocol
    return experiment


def setup(arm):
    Q.METRIC, Q.METRIC_SHA = METRIC, METRIC_SHA
    Q.TARGET, Q.TARGET_SHA = metric_definition.TARGET, metric_definition.TARGET_SHA
    Q.parent.setup = task_setup
    experiment = search.setup(arm)
    experiment.runner.NAME = f"{NAME}_{arm}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    experiment = setup(args.arm)
    prefix = ROOT / "results" / f"{NAME}_{args.arm}"
    prepared = Path(str(prefix) + f"_PREP_{args.tag}.json")
    output = Path(str(prefix) + f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive output")
    receipt = experiment.runner.prepare(str(prepared))
    print(json.dumps(receipt), flush=True)
    with experiment.extra.gpu.locking.gpu_lock(experiment.extra.gpu.LOCK):
        result = experiment.runner.run(str(prepared), receipt["sha256"], str(output),
                                      "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "recorded_known_root_not_admitted" else 1)
