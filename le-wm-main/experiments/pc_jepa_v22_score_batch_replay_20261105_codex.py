"""Apply the sealed score batch-shape diagnostic to the v22 selected chains."""
import json
from pathlib import Path

from experiments import pc_jepa_score_replay_precision_20261104_codex as precision

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_v22_contact_feedback_dev_20261104_codex"
DIAG = ROOT / "results/pc_jepa_v22_executed_packet_posthoc_20261105_codex"
OUT = ROOT / "results/pc_jepa_v22_score_batch_replay_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V22_SCORE_BATCH_REPLAY_PROTOCOL_20261105_CODEX.md"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    receipt = json.loads((DIAG / "wrapper_receipt.json").read_text())
    audit = json.loads((DIAG / "audit.json").read_text())
    result = DIAG / "result.json"
    if (receipt["status"] != "completed" or audit["status"] != "saved_array_metrics_verified"
            or receipt["result_sha256"] != precision.sha(result)
            or audit["result_sha256"] != precision.sha(result)):
        raise ValueError("prior diagnostic not audited")
    for path, digest in receipt["bindings"].items():
        if precision.sha(Path(path)) != digest:
            raise ValueError("prior diagnostic input changed")
    files = [Path(__file__), Path(precision.__file__), PROTOCOL, result, DIAG / "audit.json",
             DIAG / "wrapper_receipt.json"]
    bindings = {str(path): precision.sha(path) for path in files}
    bindings.update(receipt["bindings"])
    precision.SOURCE, precision.DIAG, precision.OUT, precision.PROTOCOL = SOURCE, DIAG, OUT, PROTOCOL
    precision.main()
    if any(precision.sha(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError("source changed during replay")
    result_path = OUT / "result.json"
    result_data = json.loads(result_path.read_text())
    if result_data["status"] != "completed" or len(result_data["rows"]) != 6:
        raise ValueError("missing results")
    with (OUT / "wrapper_receipt.json").open("x") as handle:
        json.dump({"status": "completed", "bindings": bindings,
                   "result_sha256": precision.sha(result_path)}, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
