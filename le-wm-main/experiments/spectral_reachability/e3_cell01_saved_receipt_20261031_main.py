"""Persist a compact receipt from the existing cell01 saved-data consumers."""
import hashlib
import json
from pathlib import Path

from . import agent_e3_chunked_identity_host_20261029_v1 as host

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "e3_cell01_saved_receipt_20261031_main_RUN01.json"
RUN = RESULTS / "e3_cell01_cooperative_native_20261031_main_RUN01"
AUTHORITY = RESULTS / "e3_cell01_startup_authority_20261031_main_RUN01.json"
REPORT_SHA = "36132a1dd46a52d67e903b431839ed00663b16f1ae4d7f24bc78559699462c1a"
COMPLETION_SHA = "9bfceb0d29f58d573b1bdb8684c0f2a6f5ea13b4d81a245c2a0da71f3d4d141e"
AUTHORITY_SHA = "0e525436e311dc8ffcb4cf7ec04fd0ffcbccafedc3ea52391fe9e888bb5105f8"


def compact(result):
    base = result["base"]
    physical = base["saved"]["physical"]
    native = base["partial_native"]
    boundaries = base["encoding_ledger"]["boundaries"]
    host.need(base["status"] == "completed20" and result["binding_chunks_complete"] is True
              and result["full_saved_strict"] is False and result["scientific_admission"] is False,
              "cell01 completed exploratory contract")
    host.need(physical["control_steps"] == 20 and physical["warm_steps"] == 10
              and physical["frames"] == 32 and physical["stopped"] is False,
              "cell01 physical roster")
    host.need([b["boundary"] for b in boundaries] == [10, 15]
              and all(b["completed"] is True and b["pending"] is None
                      and all(b["counts"][k] == {"attempted": 1, "returned": 1}
                              for k in ("encoder", "projector")) for b in boundaries),
              "cell01 feedback boundaries")
    host.need(native["terminal_row_verified"] is True and native["unmatched_frame_count"] == 0,
              "cell01 frame and terminal binding")
    return {
        "schema": "e3_cell01_saved_receipt_20261031_main",
        "cell": 1, "status": base["status"], "physical": physical,
        "encoding_boundaries": boundaries, "native_verification": native,
        "binding_chunks_complete": True, "full_saved_strict": False,
        "scientific_admission": False, "independent_confirmation": False,
        "equal_budget_certified": False,
        "limitations": "Known root; no physical replay or complete model identity seal; "
                       "control5 cost excluded; no scientific efficacy conclusion.",
    }


def main():
    host.need(not OUTPUT.exists() and not OUTPUT.is_symlink(), "exclusive receipt")
    print("Verifying saved host CUDA feedback records; no model inference or physical execution.", flush=True)
    result = host.inspect(str(RUN), REPORT_SHA, COMPLETION_SHA, str(AUTHORITY), AUTHORITY_SHA)
    receipt = compact(result)
    receipt.update(report_sha256=REPORT_SHA, completion_sha256=COMPLETION_SHA,
                   authority_sha256=AUTHORITY_SHA,
                   consumer_sha256=hashlib.sha256(Path(host.__file__).read_bytes()).hexdigest(),
                   receipt_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    main()
