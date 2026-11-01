"""Fixed cross-pairing of authenticated development roots and RGB targets."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from . import p1_prefix_protected_carry_20261101_main as protected
from . import e3_cp10_target_cuda_20261031 as targets

NAME = "p1_cross_goal_carry_20261101_main"
DOC = Path(__file__).resolve().parents[3] / "docs/D2P_P1_CROSS_GOAL_PROTOCOL_20261101_MAIN.md"


def cross_entry(root_entry, goal_entry):
    """Preserve root identity; never copy another root's action witness."""
    result = deepcopy(root_entry)
    for key in ("pixels", "pixel_sha256", "goal_state"):
        result[key] = deepcopy(goal_entry[key])
    for key in ("witness10_float32_hex", "historical_first_success_step",
                "initial_native_distance", "initial_joint_xy_error"):
        result.pop(key, None)
    return result


def setup(root):
    experiment = protected.setup(root)
    runner = experiment.runner
    old_inputs, old_build, old_protocol = experiment.inputs, runner.build, runner.protocol

    def inputs():
        root_entry, _, sources = old_inputs()
        entries, target_sources = targets.collect()
        receipt = json.loads(experiment.TARGET.read_bytes())
        goal_entry, target = entries[1 - root], receipt["rows"][1 - root]
        runner.require(runner.sha(experiment.TARGET) == experiment.TARGET_SHA
                       and target["root"] == 1 - root
                       and target["goal_state"] == goal_entry["goal_state"]
                       and target["pixel_sha256"] == goal_entry["pixel_sha256"],
                       "cross goal source")
        latent = np.asarray(target["latent"], np.float32)
        runner.require(latent.shape == (192,) and np.isfinite(latent).all(), "target latent")
        entry = cross_entry(root_entry, goal_entry)
        initial = targets.qualification.native(entry["goal_state"], entry["root_state"])
        runner.require(not initial["initial_success"], "cross task initially successful; not an eligible test")
        sources.update(target_sources)
        for path in (DOC, Path(__file__).resolve()):
            sources[str(path)] = runner.sha(path)
        return entry, target, sources

    def build():
        value = old_build()
        _, _, sources = inputs()
        value["sources"].update(sources)
        return value

    def protocol():
        value = old_protocol()
        value.update(goal=f"other_root_{1-root}_fixed_observed_cp10", root_goal_pair=[root, 1-root],
                     historical_reachability_witness_for_this_pair=False,
                     development_only=True, independent_confirmation=False)
        return value

    experiment.inputs = inputs
    runner.build, runner.protocol = build, protocol
    runner.NAME = f"{NAME}_root{root}"
    return experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        raise ValueError("simple tag")
    exp = setup(args.root)
    prefix = exp.runner.ROOT / "results" / f"{NAME}_root{args.root}"
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
