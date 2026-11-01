"""Execute fixed push-and-return tasks using RGB-only frozen-model planning."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from . import p1_prefix_protected_carry_20261101_main as protected
from . import p1_return_target_cuda_20261101_main as targets

NAME = "p1_return_goal_shard_20261101_main"
DOC = Path(__file__).resolve().parents[3] / "docs/D2P_P1_RETURN_GOAL_RUN_PROTOCOL_20261101_MAIN.md"
ARMS = ("structured", "native", "random", "immediate_progress")


def setup(root, arm, target_path, target_sha):
    experiment = protected.setup(root)
    runner = experiment.runner
    old_inputs, old_build, old_protocol = experiment.inputs, runner.build, runner.protocol
    owner = runner.rgb if root == 0 else runner
    old_roster = owner.roster
    owner.roster = lambda: [row for row in old_roster() if row["arm"] == arm]

    def inputs():
        entry, _, sources = old_inputs()
        runner.require(runner.sha(target_path) == target_sha, "rendered target receipt")
        receipt = json.loads(Path(target_path).read_bytes())
        old_target = json.loads(experiment.TARGET.read_bytes())
        runner.require(receipt["schema"] == targets.NAME and len(receipt["rows"]) == 2
                       and receipt["status"] == "complete"
                       and receipt["model_before"] == receipt["model_after"] == old_target["model_before"],
                       "same frozen target encoder")
        row = receipt["rows"][root]
        runner.require(row["root"] == root and row["root_state"] == entry["root_state"]
                       and not row["native_initial"]["initial_success"]
                       and row["generation"]["exact_original_pixels"] is True,
                       "root/target renderer provenance")
        pixels = np.frombuffer(base64.b64decode(row["pixels_uint8_base64"], validate=True),
                               dtype=np.uint8).reshape(224, 224, 3).copy()
        runner.require(targets.hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"],
                       "target pixels")
        entry.update(pixels=pixels, pixel_sha256=row["pixel_sha256"], goal_state=row["goal_state"])
        for key in ("witness10_float32_hex", "historical_first_success_step",
                    "initial_native_distance", "initial_joint_xy_error"):
            entry.pop(key, None)
        sources.update(receipt["sources"])
        sources[str(Path(target_path).resolve())] = target_sha
        for path in (Path(__file__).resolve(), DOC):
            sources[str(path)] = runner.sha(path)
        return entry, row, sources

    def build():
        value = old_build()
        _, _, sources = inputs()
        value["sources"].update(sources)
        return value

    def protocol():
        value = old_protocol()
        value.update(arms=[arm], panel_roster=list(ARMS), scientific_panel_denominator=8,
                     goal="object_to_cp10_pusher_returns_to_root", target_receipt_sha256=target_sha,
                     historical_reachability_witness_for_this_pair=False,
                     development_only=True, independent_confirmation=False)
        return value

    experiment.inputs = inputs
    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_root{root}_{arm}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    exp = setup(args.root, args.arm, args.target, args.target_sha)
    prefix = exp.runner.ROOT / "results" / f"{NAME}_root{args.root}_{args.arm}"
    prepared, output = Path(str(prefix) + f"_PREP_{args.tag}.json"), Path(str(prefix) + f"_RUN_{args.tag}")
    if prepared.exists() or output.exists():
        raise ValueError("exclusive output")
    receipt = exp.runner.prepare(str(prepared))
    print(json.dumps(receipt), flush=True)
    with exp.extra.gpu.locking.gpu_lock(exp.extra.gpu.LOCK):
        result = exp.runner.run(str(prepared), receipt["sha256"], str(output),
                               "MAIN_AUTHORIZES_COMMON_SUFFIX_RGB_CUDA_AND_PHYSICS")
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["status"] == "recorded_known_root_not_admitted" else 1)
