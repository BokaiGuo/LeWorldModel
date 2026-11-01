"""Bind authentic cp15 physical prefix to new captured CUDA suffixes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import codex_p2_cp15_cuda_20261030_v1 as driver
from experiments.spectral_reachability import codex_p2_cp15_cuda_saved_20261030_v1 as saved
from experiments.spectral_reachability import agent_p2_full30_control15_feedback_physical_20261029_v1 as old

NAME = "codex_p2_cp15_physical_binding_20261030_v1"
ROOT = driver.ROOT
RUN = ROOT / "results/codex_p2_cp15_cuda_20261030_v1_RUN_01"
REPORT_SHA = "8bc0d958ad79824294c0e5cadeb8a8dd15c0af7395dcc98a919b35d1baeb07f5"
PREP = ROOT / "results/codex_p2_cp15_cuda_20261030_v1_PREP_01.json"
PREP_SHA = "1fbec2bce79c14fb649e4bf0026a44e083004b82da81c289d37fa2bbf98866ef"
RECEIPT = ROOT / "results/codex_p2_cp15_cuda_saved_20261030_v1_RECEIPT_01.json"
RECEIPT_SHA = "c38b26fc467ee19126bacba65ff7006a62ebe0b4577070d5d1ea63d2e3434432"


def make_binding(base, physical, case, rows):
    need = driver.panel.require
    need(type(base) is int and 0 <= base < 4 and physical["job_index"] == case["base"] == base
         and physical["observed_context"] == case["context"], "feedback identity")
    prefix = np.asarray(case["prefix_raw"], dtype="<f4")
    need(prefix.shape == (15, 2) and hashlib.sha256(prefix.tobytes()).hexdigest() ==
         case["prefix_float32_sha256"], "saved prefix")
    need(len(physical["prefix_reference"]) == 106 and
         sum(x["kind"] == "fresh_observation" for x in physical["prefix_reference"]) == 81 and
         sum(x["kind"] == "returned_transition" for x in physical["prefix_reference"]) == 25,
         "complete native/RGB reference")
    plans = []
    for arm, row in zip(driver.panel.ARMS, rows, strict=True):
        need((row["base"], row["arm"], row["status"]) == (base, arm, "complete"), "fixed CUDA row")
        raw = bytes.fromhex(row["winner_raw_hex"])
        need(len(raw) == 120 and hashlib.sha256(raw).hexdigest() == row["winner_raw_sha256"],
             "CUDA suffix bytes")
        suffix = np.frombuffer(raw, dtype="<f4").reshape(15, 2)
        need(np.isfinite(suffix).all() and np.all(np.abs(suffix) <= 1), "suffix action domain")
        parent = physical["plans"][1]
        original = np.asarray(parent["commands"], dtype="<f4")
        need(original.shape == (30, 2) and original[:15].tobytes() == prefix.tobytes()
             and parent["task_id"] == case["task_id"] and parent["root_slot"] == case["root"],
             "native parent task/root/prefix")
        plans.append(dict(base=base, arm=arm, task_id=case["task_id"], root=case["root"],
                          prefix_sha256=case["prefix_float32_sha256"],
                          feedback_context_sha256=case["feedback_context_sha256"],
                          suffix_raw_float32_hex=raw.hex(), suffix_raw_sha256=row["winner_raw_sha256"],
                          full_raw_float32_hex=prefix.tobytes().hex()+raw.hex(),
                          source="new CUDA winner; not old planner witness"))
    return dict(schema=NAME+".prepare", base=base, plans=plans,
                prefix_reference=physical["prefix_reference"],
                observed_context=case["context"], original_bindings=physical["bindings"],
                cuda_report_sha256=REPORT_SHA, cuda_prepare_sha256=PREP_SHA,
                cuda_receipt_sha256=RECEIPT_SHA, physical_ready=False, scientific_pass=False)


def guarded_suffix_dispatch(actual_prefix, binding, arm, stopped, durable_emit, dispatch):
    """Never dispatch a suffix before full comparison and durable guard."""
    need = driver.panel.require
    need(stopped is False and actual_prefix == binding["prefix_reference"], "cp15 prefix or stop")
    plans = [p for p in binding["plans"] if p["arm"] == arm]
    need(len(plans) == 1 and callable(durable_emit) and callable(dispatch), "fixed arm and callbacks")
    plan = plans[0]
    durable_emit({"kind": "feedback_prefix_guard", "base": binding["base"], "arm": arm,
                  "feedback_context_sha256": plan["feedback_context_sha256"],
                  "suffix_raw_sha256": plan["suffix_raw_sha256"]})
    return dispatch(bytes.fromhex(plan["suffix_raw_float32_hex"]))


def build(base):
    need = driver.panel.require
    need(driver.sha(RECEIPT) == RECEIPT_SHA, "CUDA saved receipt SHA")
    rebuilt = saved.consume(RUN, REPORT_SHA, PREP, PREP_SHA)
    need(rebuilt == json.loads(RECEIPT.read_bytes()), "full CUDA receipt reconstruction")
    report = json.loads((RUN / "report.json").read_bytes())
    case = json.loads(driver.PREP.read_bytes())["cases"][base]
    physical, _ = old.material(base)
    value = make_binding(base, physical, case, report["rows"][2*base:2*base+2])
    files = [Path(__file__).resolve(), ROOT / "tests/test_codex_p2_cp15_physical_binding_20261030_v1.py",
             ROOT / "innerdocs/codex_p2_cp15_physical_binding_20261030_v1.md"]
    value["own_sources"] = {str(p): driver.sha(p) for p in files}
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, type=int, choices=range(4))
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    driver.panel.require(out.is_absolute() and out.parent == ROOT / "results" and
                         out.name.startswith(NAME + f"_B{a.base}_PREP_") and not os.path.lexists(out),
                         "exclusive path")
    print(driver.write_new(out, build(a.base)))


if __name__ == "__main__":
    main()
