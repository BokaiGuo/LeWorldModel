"""Run existing fixed-bank diagnostic on sealed v22 records in three phases."""
import argparse
import json
from pathlib import Path

from experiments import pc_jepa_v12_contact_score_bank_posthoc_20261104_codex as physical
from experiments import pc_jepa_v12_contact_score_bank_neural_posthoc_20261104_codex as neural
from experiments import pc_jepa_v12_contact_score_bank_audit_20261104_codex as audit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_v22_contact_feedback_dev_20261104_codex"
OUT = ROOT / "results/pc_jepa_v22_fixed_bank_posthoc_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V22_FIXED_BANK_PHYSICAL_SCORE_POSTHOC_PROTOCOL_20261105_CODEX.md"


def preflight():
    sha = physical.feedback.sha
    saved = json.loads((SOURCE / "audit.json").read_text())
    replay = json.loads((SOURCE / "physical_replay.json").read_text())
    chrono = json.loads((SOURCE / "chronology_audit.json").read_text())
    audit_sha = sha(SOURCE / "audit.json")
    summary_sha = sha(SOURCE / "summary.json")
    if (saved["status"] != "saved_data_audit_passed" or saved["summary_sha256"] != summary_sha
            or replay["status"] != "exact" or replay["summary_sha256"] != summary_sha
            or replay["audit_sha256"] != audit_sha or chrono["status"] != "passed"
            or chrono["audit_sha256"] != audit_sha):
        raise ValueError("v22 source audits do not match")
    return sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("physical", "neural", "audit"))
    args = parser.parse_args()
    sha = preflight()
    physical.SOURCE, physical.OUT, physical.PROTOCOL = SOURCE, OUT, PROTOCOL
    neural.SOURCE, neural.OUT = SOURCE, OUT
    audit.SOURCE, audit.OUT = SOURCE, OUT
    if args.phase == "physical":
        print("Replay fixed v22 candidate bank and evaluation-only witness variants.", flush=True)
        physical.main()
        receipt = {"status": "completed", "protocol_sha256": sha(PROTOCOL),
                   "wrapper_sha256": sha(Path(__file__)), "source_audit_sha256": sha(SOURCE / "audit.json"),
                   "source_chronology_sha256": sha(SOURCE / "chronology_audit.json"),
                   "result_sha256": sha(OUT / "result.json")}
        with (OUT / "v22_receipt.json").open("x") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
    else:
        receipt = json.loads((OUT / "v22_receipt.json").read_text())
        if (receipt["status"] != "completed" or receipt["protocol_sha256"] != sha(PROTOCOL)
                or receipt["wrapper_sha256"] != sha(Path(__file__))
                or receipt["source_audit_sha256"] != sha(SOURCE / "audit.json")
                or receipt["source_chronology_sha256"] != sha(SOURCE / "chronology_audit.json")
                or receipt["result_sha256"] != sha(OUT / "result.json")):
            raise ValueError("v22 physical receipt changed")
        if args.phase == "neural":
            print("Score fixed v22 bank in isolated neural workers; no planning.", flush=True)
            neural.main()
        else:
            print("Audit bank, rank arithmetic and same-backend physical replay.", flush=True)
            audit.main()


if __name__ == "__main__":
    main()
