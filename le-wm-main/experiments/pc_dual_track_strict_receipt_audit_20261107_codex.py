"""Apply strict eligibility to existing actual dual-track smoke receipts."""
import json
from pathlib import Path

from experiments.pc_dual_track_receipt_audit_20261106_codex import FOLDER, PROTOCOL, TASKS
from experiments.pc_input_permissions import RGB, STATE, file_hash
from experiments.pc_track_registry_strict_20261107_codex import strict_eligible

OUTPUT = FOLDER / "strict_receipt_audit_20261107_codex.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = []
    for track in (RGB, STATE):
        path = FOLDER / track / "result.json"
        record = json.loads(path.read_text())
        args = dict(algorithm_version="input_boundary_smoke_v1", protocol_path=PROTOCOL,
                    task_manifest_path=TASKS[track],
                    required_operations={"score" if track == RGB else "predict"})
        strict_eligible(record, target_track=track, **args)
        try:
            strict_eligible(record, target_track=STATE if track == RGB else RGB, **args)
        except ValueError:
            rejected = True
        else:
            raise ValueError("cross-track evidence was accepted")
        rows.append({"track": track, "result_sha256": file_hash(path),
                     "protocol_sha256": file_hash(PROTOCOL), "task_sha256": file_hash(TASKS[track]),
                     "receipt_count": len(record["input_receipts"]), "cross_track_rejected": rejected})
        print(f"{track}: strict eligibility validated; cross-track rejected", flush=True)
    strict_code = Path(__file__).with_name("pc_track_registry_strict_20261107_codex.py")
    with OUTPUT.open("x") as stream:
        json.dump({"scope": "receipt schema and local file bindings only; not scientific gate evidence",
                   "runner_sha256": file_hash(__file__), "strict_code_sha256": file_hash(strict_code),
                   "rows": rows}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
