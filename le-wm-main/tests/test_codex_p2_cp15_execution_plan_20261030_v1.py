from copy import deepcopy
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p2_cp15_execution_plan_20261030_v1 as bridge


def example():
    prefix = np.linspace(-0.5, 0.5, 30, dtype="<f4").tobytes()
    physical = dict(job_index=0, observed_context={"latent": [1]}, bindings={"source": "hash"},
                    prefix_reference=[{"kind": "frame", "n": i} for i in range(106)],
                    plans=[{}, dict(commands=np.frombuffer(prefix + bytes(120), dtype="<f4").reshape(30, 2).tolist(),
                                    task_id="task", root_slot=1, planner_plan={"witness_sha256": "old"},
                                    native_goal=[1, 2], selection_digest="old")])
    plans = []
    for i, arm in enumerate(bridge.ARMS):
        suffix = np.full((15, 2), i / 4, dtype="<f4").tobytes()
        plans.append(dict(base=0, arm=arm, task_id="task", root=1,
                          full_raw_float32_hex=(prefix + suffix).hex(),
                          suffix_raw_float32_hex=suffix.hex(),
                          prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                          suffix_raw_sha256=hashlib.sha256(suffix).hexdigest(),
                          feedback_context_sha256="context"))
    binding = dict(schema=bridge.binding.NAME + ".prepare", base=0, plans=plans,
                   observed_context=deepcopy(physical["observed_context"]),
                   prefix_reference=deepcopy(physical["prefix_reference"]),
                   original_bindings=deepcopy(physical["bindings"]),
                   cuda_report_sha256=bridge.binding.REPORT_SHA,
                   cuda_prepare_sha256=bridge.binding.PREP_SHA,
                   cuda_receipt_sha256=bridge.binding.RECEIPT_SHA,
                   physical_ready=False, scientific_pass=False)
    return physical, binding


def test_raw_winner_and_task_identity():
    physical, binding = example()
    result = bridge.assemble(0, physical, binding)
    assert [x["policy"] for x in result["plans"]] == list(bridge.ARMS)
    for plan, source in zip(result["plans"], binding["plans"], strict=True):
        assert np.asarray(plan["commands"], dtype="<f4").tobytes().hex() == source["full_raw_float32_hex"]
        assert plan["planner_plan"] == physical["plans"][1]["planner_plan"]
        assert plan["new_model_plan"]["schema"].endswith(".cuda_winner_reference")
    assert result["physical_ready"] is False


@pytest.mark.parametrize("mutation", ["context", "prefix", "goal", "suffix", "task", "order", "source"])
def test_reject_mismatched_inputs(mutation):
    physical, binding = example()
    if mutation == "context":
        binding["observed_context"]["latent"] = [2]
    elif mutation == "prefix":
        binding["prefix_reference"][0]["n"] = 999
    elif mutation == "goal":
        # Goal identity is part of the native parent, not established by an
        # isolated synthetic binding; changing source bindings must be rejected.
        binding["original_bindings"]["source"] = "changed"
    elif mutation == "suffix":
        binding["plans"][0]["suffix_raw_float32_hex"] = (b"\x01" + bytes(119)).hex()
    elif mutation == "task":
        binding["plans"][0]["task_id"] = "other"
    elif mutation == "order":
        binding["plans"].reverse()
    else:
        binding["cuda_report_sha256"] = "wrong"
    with pytest.raises(ValueError):
        bridge.assemble(0, physical, binding)


def test_invalid_base_rejected_before_read():
    with pytest.raises(ValueError):
        bridge.rebuild(True)
