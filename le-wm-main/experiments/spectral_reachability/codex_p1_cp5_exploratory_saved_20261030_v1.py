"""Reconsume saved 24-row exploratory CUDA planning without model or physics."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from . import codex_p1_cp5_exploratory_cuda_20261030_v1 as run


def read(path, expected=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("external file digest")
    return json.loads(raw), digest


def consume(directory, manifest_sha, report_sha):
    directory = Path(directory)
    manifest, _ = read(directory/"manifest.json", manifest_sha)
    report, _ = read(directory/"report.json", report_sha)
    if (manifest["schema"] != run.NAME+".manifest" or manifest["bundle_sha256"] != run.BUNDLE_SHA
            or manifest["rows"] != 24 or report["manifest_sha256"] != manifest_sha
            or report["status"] != "complete" or report["rows_completed"] != 24
            or report["model_before"] != run.MODEL_HASH or report["model_after"] != run.MODEL_HASH
            or report["physical_execution"] is not False):
        raise ValueError("saved manifest or report")
    bundle, _ = read(run.BUNDLE, run.BUNDLE_SHA)
    rows = []
    for index in range(24):
        event, event_sha = read(directory/f"row_{index:02d}_result.json")
        if event["kind"] != "row_complete" or event["index"] != index:
            raise ValueError("row identity")
        result = event["result"]
        if result["index"] != index or not isinstance(result["score"], (int, float)) or not math.isfinite(result["score"]):
            raise ValueError("score or index")
        action = result["actions"]
        if (len(action) != 1 or len(action[0]) != 2 or
                any(len(packet) != 5 or any(len(pair) != 2 or
                    any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1 for x in pair)
                    for pair in packet) for packet in action[0])):
            raise ValueError("saved raw actions")
        policy, contract = run.make_contract(bundle["rows"][index], manifest["sources"])
        digest = run.spec.digest(contract)
        saved = result["ledger_receipt"]
        folder = directory/f"row_{index:02d}"
        account = run.consumer.consume(folder/"encoder.jsonl", saved["encoding"]["sha256"],
                    manifest["encoder_caps"], manifest["wall_seconds"],
                    folder/"planning/events.jsonl", saved["planning"]["journal_sha256"],
                    digest, "structured", "fixed", f"row_{index:02d}")
        if account != saved or not account["complete"]:
            raise ValueError("saved ledger mismatch")
        rows.append({"index": index, "row_sha256": event_sha, "encoder_sha256": saved["encoding"]["sha256"],
                     "planning_sha256": saved["planning"]["journal_sha256"]})
    return {"schema": "codex_p1_cp5_exploratory_saved_20261030_v1", "status": "24_rows_consumed",
            "manifest_sha256": manifest_sha, "report_sha256": report_sha, "rows": rows,
            "physical_execution": False, "isolation_certified": False, "fairness_proven": False,
            "independent_confirmation": False, "P1_admission": False,
            "limitation": "Row and journal hashes are not independently anchored in original report."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = consume(args.directory, args.manifest_sha, args.report_sha)
    output = Path(args.output)
    if not output.is_absolute() or output.parent != run.ROOT/"results" or not output.name.startswith(
            "codex_p1_cp5_exploratory_saved_20261030_v1_"):
        raise ValueError("exclusive saved receipt path")
    print(json.dumps({"sha256": run.write(output, result), "status": result["status"]}))


if __name__ == "__main__":
    main()
