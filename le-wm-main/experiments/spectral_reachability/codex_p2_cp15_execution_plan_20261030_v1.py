"""Rebuild physical execution plans from the saved cp15 CUDA winners.

This is a source bridge, not a physical runner or a physical saved consumer.
It performs no neural inference and does not dispatch environment actions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import codex_p2_cp15_physical_binding_20261030_v1 as binding

NAME = "codex_p2_cp15_execution_plan_20261030_v1"
ARMS = ("joint", "suffix")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def assemble(base, physical, saved_binding):
    """Join new raw winners to the original task, never an old model witness."""
    require(type(base) is int and base in range(4), "fixed base")
    require(physical["job_index"] == saved_binding["base"] == base, "base identity")
    require(saved_binding["schema"] == binding.NAME + ".prepare", "binding schema")
    require(saved_binding["physical_ready"] is False and saved_binding["scientific_pass"] is False,
            "binding evidence boundary")
    require(saved_binding["observed_context"] == physical["observed_context"], "actual feedback context")
    require(saved_binding["prefix_reference"] == physical["prefix_reference"] and
            len(saved_binding["prefix_reference"]) == 106, "actual native/RGB prefix")
    require(saved_binding["original_bindings"] == physical["bindings"], "parent source bindings")
    require((saved_binding["cuda_report_sha256"], saved_binding["cuda_prepare_sha256"],
             saved_binding["cuda_receipt_sha256"]) ==
            (binding.REPORT_SHA, binding.PREP_SHA, binding.RECEIPT_SHA), "CUDA source identities")
    require(len(saved_binding["plans"]) == 2, "fixed two arms")
    parent = physical["plans"][1]
    parent_raw = np.asarray(parent["commands"], dtype="<f4")
    require(parent_raw.shape == (30, 2) and np.isfinite(parent_raw).all(), "parent action shape")
    plans = []
    for index, (arm, source) in enumerate(zip(ARMS, saved_binding["plans"], strict=True)):
        require(source["base"] == base and source["arm"] == arm and
                source["task_id"] == parent["task_id"] and source["root"] == parent["root_slot"],
                "fixed arm and native task identity")
        raw = bytes.fromhex(source["full_raw_float32_hex"])
        suffix = bytes.fromhex(source["suffix_raw_float32_hex"])
        require(len(raw) == 240 and len(suffix) == 120 and raw[:120] == parent_raw[:15].tobytes()
                and raw[120:] == suffix, "literal prefix and suffix bytes")
        require(hashlib.sha256(raw[:120]).hexdigest() == source["prefix_sha256"] and
                hashlib.sha256(suffix).hexdigest() == source["suffix_raw_sha256"], "raw hashes")
        values = np.frombuffer(raw, dtype="<f4").reshape(30, 2)
        require(np.isfinite(values).all() and np.all(np.abs(values) <= 1), "raw action domain")
        witness = dict(schema=NAME + ".cuda_winner_reference", base=base, arm=arm,
                       input_context=deepcopy(saved_binding["observed_context"]),
                       feedback_context_sha256=source["feedback_context_sha256"],
                       cuda_report_sha256=binding.REPORT_SHA, cuda_prepare_sha256=binding.PREP_SHA,
                       cuda_receipt_sha256=binding.RECEIPT_SHA,
                       suffix_raw_sha256=source["suffix_raw_sha256"],
                       scope="saved CUDA winner reference; not the parent model witness")
        witness["new_witness_sha256"] = digest(witness)
        plan = deepcopy(parent)
        plan.update(schema=NAME + ".execution_plan", index=index, base_index=base, policy=arm,
                    commands=values.astype(float).tolist(), raw_float32_hex=raw.hex(),
                    raw_float32_sha256=hashlib.sha256(raw).hexdigest(), new_model_plan=witness)
        plan.pop("selection_digest", None)
        plan["selection_digest"] = digest(plan)
        plans.append(plan)
    return dict(schema=NAME + ".prepare", base=base, plans=plans,
                prefix_reference=deepcopy(saved_binding["prefix_reference"]),
                observed_context=deepcopy(saved_binding["observed_context"]),
                physical_ready=False, physical_execution=False, scientific_pass=False,
                limitations=["known roots", "no physical lifecycle or independent saved consumer",
                             "not equal total compute", "not full program composition"])


def rebuild(base):
    """Reconstruct both trusted inputs and compare the previously saved binding."""
    require(type(base) is int and base in range(4), "fixed base")
    path = binding.ROOT / "results" / (binding.NAME + f"_B{base}_PREP_01.json")
    require(path.is_file() and not path.is_symlink(), "saved binding file")
    stored = json.loads(path.read_bytes())
    reconstructed = binding.build(base)
    require(stored == reconstructed, "saved binding does not reconstruct")
    physical, _ = binding.old.material(base)
    return assemble(base, physical, reconstructed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    value = rebuild(args.base)
    print(json.dumps(dict(base=args.base, plan_sha256=digest(value), arms=list(ARMS),
                          physical_ready=False, scientific_pass=False), sort_keys=True))


if __name__ == "__main__":
    main()
