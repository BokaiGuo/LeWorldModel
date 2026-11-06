"""Independently recalculate saved v22 packet metrics on CPU."""
import json
from pathlib import Path

from experiments import pc_jepa_native_executed_packet_audit_20261104_codex as audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_jepa_v22_executed_packet_posthoc_20261105_codex"
SOURCE = ROOT / "results/pc_jepa_v22_contact_feedback_dev_20261104_codex"


def main():
    receipt = json.loads((OUT / "wrapper_receipt.json").read_text())
    if receipt["status"] != "completed" or receipt["result_sha256"] != audit.sha(OUT / "result.json"):
        raise ValueError("diagnostic receipt mismatch")
    for path, digest in receipt["bindings"].items():
        if audit.sha(Path(path)) != digest:
            raise ValueError("wrapper source changed")
    audit.SOURCE, audit.OUT = SOURCE, OUT
    audit.main()


if __name__ == "__main__":
    main()
