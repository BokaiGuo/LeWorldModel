"""Bind actual saved CUDA program winners to original cp15 physical prefix.

This produces a saved physical handoff only; it does not dispatch an env.
The horizon is 15 prefix + 10 winner steps, with no padding to H30.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import codex_p2_program_host_saved_20261030_v1 as saved
from experiments.spectral_reachability import codex_p2_cp15_physical_binding_20261030_v1 as old

NAME="codex_p2_program_physical_binding_20261030_v1"
ROOT=old.ROOT
RUN=ROOT/"results/codex_p2_program_host_runner_20261030_v1_RUN_02"
REPORT_SHA="f2daa8b1702eefb0d3d32384394900cf24a5d3cbd85f9c838e487cc76cb42296"
SUP_SHA="5bbafb89dcd64837b4cc38a7569ad100e4e3daf2e8b084d94aa9487d251749a1"
PREP=ROOT/"results/codex_p2_program_host_runner_20261030_v1_PREP_01.json"
PREP_SHA="8e915ae1da4df255daa1c168c7fed24383a469ead64ce3dac44470297e7325b9"
RECEIPT=ROOT/"results/codex_p2_program_host_saved_20261030_v1_RECEIPT_01.json"
RECEIPT_SHA="cdc59e679beab31faae873e7bdb3a856ab323ed1ba6cd97a1dc30975506cd6af"


def assemble(base,physical,case,branches):
    need=old.driver.panel.require
    need(type(base) is int and base in range(4) and physical["job_index"]==case["base"]==base
         and physical["observed_context"]==case["context"] and len(physical["prefix_reference"])==106,
         "actual cp15 context and reference")
    parent=physical["plans"][1]
    raw=np.asarray(parent["commands"],dtype="<f4")
    prefix=np.asarray(case["prefix_raw"],dtype="<f4")
    need(raw.shape==(30,2) and prefix.shape==(15,2) and raw[:15].tobytes()==prefix.tobytes() and
         hashlib.sha256(prefix.tobytes()).hexdigest()==case["prefix_float32_sha256"] and
         parent["task_id"]==case["task_id"] and parent["root_slot"]==case["root"],
         "source task and literal original raw prefix")
    need(len(branches)==2,"fixed arms")
    plans=[]
    for arm,branch in zip(("joint","library_chain"),branches,strict=True):
        need((branch["base"],branch["arm"],branch["physical"],branch["P2_admission"])==
             (base,arm,False,False),"fixed winner identity")
        winner=branch["result"]["winner"]
        suffix=bytes.fromhex(winner["candidate"]["raw_float32_hex"])
        need(len(suffix)==80 and len(winner["edges"])==2 and
             "".join(e["raw_float32_hex"] for e in winner["edges"])==suffix.hex(),"two actual winner packets")
        a=np.frombuffer(suffix,dtype="<f4").reshape(10,2)
        need(np.isfinite(a).all() and np.all(np.abs(a)<=1),"native raw action box")
        plans.append(dict(base=base,arm=arm,task_id=case["task_id"],root=case["root"],
            prefix_sha256=case["prefix_float32_sha256"],feedback_context_sha256=case["feedback_context_sha256"],
            winner_index=branch["result"]["winner_index"],program_ids=winner["candidate"]["program_ids"],
            winner_suffix_float32_hex=suffix.hex(),winner_suffix_sha256=hashlib.sha256(suffix).hexdigest(),
            full25_raw_float32_hex=prefix.tobytes().hex()+suffix.hex(),horizon=25,
            source="actual saved CUDA winner; never reconstructed logits or padded"))
    return dict(schema=NAME+".binding",base=base,plans=plans,physical_ready=False,
        prefix_reference=physical["prefix_reference"],observed_context=case["context"],
        original_bindings=physical["bindings"],cuda_report_sha256=REPORT_SHA,
        cuda_receipt_sha256=RECEIPT_SHA,scientific_pass=False)


def build(base):
    need=old.driver.panel.require
    need(type(base) is int and base in range(4),"base before input reads")
    need(old.driver.sha(RECEIPT)==RECEIPT_SHA,"saved receipt identity")
    rebuilt=saved.consume(RUN,REPORT_SHA,SUP_SHA,PREP,PREP_SHA)
    need(rebuilt==json.loads(RECEIPT.read_bytes()),"strict complete saved reader reconstruction")
    case=json.loads(old.driver.PREP.read_bytes())["cases"][base]
    physical,_=old.old.material(base)
    branches=[json.loads((RUN/f"branch_{base*2+i}.json").read_bytes()) for i in range(2)]
    return assemble(base,physical,case,branches)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base",type=int,choices=range(4),required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args();path=Path(a.output)
    old.driver.panel.require(path.is_absolute() and path.parent==ROOT/"results" and
        path.name.startswith(NAME+f"_B{a.base}_PREP_") and not os.path.lexists(path),"exclusive handoff")
    print(json.dumps(dict(path=str(path),sha256=old.driver.write_new(path,build(a.base)),
                          physical_ready=False,scientific_pass=False)))


if __name__=="__main__":main()
