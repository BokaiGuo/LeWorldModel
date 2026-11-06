"""Apply the existing executed-packet diagnostic to sealed v22 records."""
import json
from pathlib import Path

from experiments import pc_jepa_native_executed_packet_diag_20261104_codex as diag

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_v22_contact_feedback_dev_20261104_codex"
OUT = ROOT / "results/pc_jepa_v22_executed_packet_posthoc_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V22_EXECUTED_PACKET_POSTHOC_PROTOCOL_20261105_CODEX.md"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    audit = json.loads((SOURCE / "audit.json").read_text())
    replay = json.loads((SOURCE / "physical_replay.json").read_text())
    chronology = json.loads((SOURCE / "chronology_audit.json").read_text())
    audit_hash = diag.sha(SOURCE / "audit.json")
    summary_hash = diag.sha(SOURCE / "summary.json")
    if (audit["status"] != "saved_data_audit_passed" or audit["summary_sha256"] != summary_hash
            or replay["status"] != "exact" or replay["summary_sha256"] != summary_hash
            or replay["audit_sha256"] != audit_hash or chronology["status"] != "passed"
            or chronology["audit_sha256"] != audit_hash):
        raise ValueError("original audits do not match")
    files = [Path(__file__), Path(diag.__file__), PROTOCOL, SOURCE / "manifest.json",
             SOURCE / "summary.json", SOURCE / "audit.json", SOURCE / "physical_replay.json",
             SOURCE / "chronology_audit.json"]
    metadata_hashes = set()
    for arm in ("response", "cem"):
        files.append(SOURCE / f"{arm}_result.json")
        for step in range(3):
            directory = SOURCE / f"{arm}_{step}"
            metadata = directory / "input/model_metadata.json"
            decision_path = directory / "output/decision.json"
            decision = json.loads(decision_path.read_text())
            metadata_hashes.add(diag.sha(metadata))
            if decision["metadata_sha256"] != diag.sha(metadata):
                raise ValueError("decision metadata mismatch")
            files.extend((metadata, directory / "input/context.npz", decision_path,
                          SOURCE / f"{arm}_physical/frame_{15 + 5 * step:02d}/rgb.npy"))
            files.extend(SOURCE / f"{arm}_physical/command_{i:02d}.json"
                         for i in range(11 + 5 * step, 16 + 5 * step))
    if len(metadata_hashes) != 1:
        raise ValueError("different model metadata across decisions")
    bindings = {str(path): diag.sha(path) for path in files}
    diag.SOURCE, diag.OUT, diag.PROTOCOL = SOURCE, OUT, PROTOCOL
    diag.main()
    if any(diag.sha(Path(path)) != value for path, value in bindings.items()):
        raise ValueError("source changed during diagnostic")
    result_path = OUT / "result.json"
    result = json.loads(result_path.read_text())
    if result["status"] != "completed" or len(result["rows"]) != 6:
        raise ValueError("missing diagnostic rows")
    with (OUT / "wrapper_receipt.json").open("x") as handle:
        json.dump({"status": "completed", "bindings": bindings,
                   "result_sha256": diag.sha(result_path),
                   "scope": "same model implementation and backend; post-hoc six packets"},
                  handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
