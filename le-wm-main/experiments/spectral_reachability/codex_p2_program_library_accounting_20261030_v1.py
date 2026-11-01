"""Source preparation and durable resource accounting for the P2 library pair.

No CUDA, model loading, physical execution, or production run entry here.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import os
from pathlib import Path
import time

from experiments.spectral_reachability import codex_p2_cp15_cuda_20261030_v1 as previous
from experiments.spectral_reachability import codex_p2_program_library_pair_20261030_v1 as kernel
from experiments.spectral_reachability import codex_p2_program_library_saved_20261030_v1 as saved
from experiments.spectral_reachability import codex_p2_program_library_cuda_adapter_20261030_v1 as adapter
from experiments.spectral_reachability import codex_p3_quota_source_identity_20261030_v1 as identity
from experiments.spectral_reachability import codex_p3_quota_journal_20261030_v1 as journal
from experiments.spectral_reachability import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as lock
from experiments.spectral_reachability import codex_p2_program_library_supervisor_20261030_v1 as supervisor

NAME="codex_p2_program_library_accounting_20261030_v1"
ROOT=previous.ROOT
LOCK=previous.LOCK
CAPS={"packet":70,"action_encoder":70,"predict":70,"predictor":70,"pred_proj":70,
      "jacobian":1,"svd":1,"score":34}
WALL=450.0


def material():
    # The existing material reconstructs the original pinned cp15 panel and
    # hashes the real model/config without loading the model.
    prior=previous.material()
    paths=[Path(__file__).resolve(),Path(kernel.__file__).resolve(),
           Path(saved.__file__).resolve(),Path(adapter.__file__).resolve(),
           Path(identity.__file__).resolve(),Path(journal.__file__).resolve(),
           Path(lock.__file__).resolve(),
           Path(supervisor.__file__).resolve(),
           ROOT/"tests/test_codex_p2_program_library_supervisor_20261030_v1.py",
           ROOT/"tests/test_codex_p2_program_library_accounting_20261030_v1.py",
           ROOT/"innerdocs/codex_p2_program_library_accounting_20261030_v1.md"]
    sources={row["path"]:row for row in prior["sources"]}
    for row in identity.inventory(paths):
        sources[row["path"]]=row
    panel=json.loads(previous.PREP.read_bytes())
    kernel.require(len(panel["cases"])==4,"four pinned known cases")
    return dict(schema=NAME+".prepare",protocol=kernel.contract(),caps=CAPS,
                wall_seconds=WALL,case_order=[0,1,2,3],arms=list(kernel.ARMS),
                panel_path=str(previous.PREP),panel_sha256=previous.PREP_SHA,
                cases=[dict(base=c["base"],task_id=c["task_id"],root=c["root"],
                            feedback_context_sha256=c["feedback_context_sha256"],
                            prefix_float32_sha256=c["prefix_float32_sha256"]) for c in panel["cases"]],
                policy=prior["policy"],expected_model_hash=prior["expected_model_hash"],
                gpu_uuid=prior["gpu_uuid"],sources=sorted(sources.values(),key=lambda x:x["path"]),
                lock_path=str(LOCK),lock_scope="cooperating processes only",
                loaded_dynamic_closure="unknown",full_compute_equal=False,
                P1_admission=False,P2_admission=False,physical=False)


def write_prepare(path):
    path=Path(path)
    kernel.require(path.is_absolute() and path.parent==ROOT/"results" and
                   path.name.startswith(NAME+"_PREP_") and not os.path.lexists(path),"exclusive prepare")
    return previous.write_new(path,material())


def verify_prepare(path,expected):
    path=Path(path)
    kernel.require(path.is_absolute() and not path.is_symlink() and
                   previous.sha(path)==expected,"external prepare identity")
    value=json.loads(path.read_bytes())
    identity.verify(value["sources"])
    kernel.require(value==material(),"source and case reconstruction")
    return value


class Meter:
    """Per-root nested requests with durable attempt/return/failure events."""
    def __init__(self,emit,clock=time.monotonic):
        kernel.require(callable(emit) and callable(clock),"meter callbacks")
        self.emit=emit;self.clock=clock;self.start=self.last=clock()
        self.attempted=dict.fromkeys(CAPS,0);self.returned=dict.fromkeys(CAPS,0)
        self.failed=False;self.depth=0

    def tick(self):
        now=self.clock()
        kernel.require(math.isfinite(now) and now>=self.last,"monotonic clock")
        self.last=now
        return now-self.start

    def call(self,kind,fn,details=None):
        elapsed=self.tick()
        kernel.require(not self.failed and kind in CAPS and elapsed<WALL and
                       self.attempted[kind]<CAPS[kind] and callable(fn),"resource predispatch")
        ordinal=self.attempted[kind]+1
        self.emit(dict(kind="resource_attempt",resource=kind,ordinal=ordinal,
                       depth=self.depth,elapsed_seconds=elapsed,details=details))
        self.attempted[kind]=ordinal
        self.depth+=1
        try:
            value=fn()
        except BaseException as exc:
            self.failed=True
            self.emit(dict(kind="resource_failure",resource=kind,ordinal=ordinal,
                           depth=self.depth-1,error_type=type(exc).__name__,message=str(exc)[:1000]))
            raise
        finally:
            self.depth-=1
        elapsed=self.tick()
        self.returned[kind]+=1
        self.emit(dict(kind="resource_return",resource=kind,ordinal=ordinal,
                       depth=self.depth,elapsed_seconds=elapsed))
        if elapsed>WALL:
            self.failed=True
            raise RuntimeError("wall exceeded during resource call")
        return value

    def call_extra(self,kind,fn):
        return self.call(kind,fn)

    def complete(self):
        kernel.require(not self.failed and self.depth==0 and
                       self.attempted==self.returned==CAPS,"full separate resource census")
        return dict(attempted=dict(self.attempted),returned=dict(self.returned),
                    wall_seconds=self.tick())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(dict(path=args.output,sha256=write_prepare(args.output),
                          status="prepared_no_cuda_or_physics"),sort_keys=True))


if __name__=="__main__":
    main()
