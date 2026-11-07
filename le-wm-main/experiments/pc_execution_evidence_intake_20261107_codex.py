"""Fail-closed intake for dual-track planning execution records.

Intake establishes record eligibility only; it never passes a scientific gate.
It is not an authenticity mechanism against adversarial Python or forged inputs.
"""
import argparse
import json
from pathlib import Path

from experiments.pc_input_permissions import RGB, STATE, SCOPES, file_hash
from experiments.pc_track_registry_strict_20261107_codex import strict_eligible

REQUIRED = {RGB: frozenset(("score",)), STATE: frozenset(("predict",))}
GATES = frozenset(f"PC{i}" for i in range(6))


def intake(record_path, *, track_id, algorithm_version, protocol_path, task_manifest_path, gate):
    """Validate a completed planning run and return a non-promoting receipt."""
    if track_id not in REQUIRED or gate not in GATES or not algorithm_version:
        raise ValueError("recognized track, gate and algorithm version required")
    path = Path(record_path)
    record = json.loads(path.read_text())
    strict_eligible(record, target_track=track_id, algorithm_version=algorithm_version,
                    protocol_path=protocol_path, task_manifest_path=task_manifest_path,
                    required_operations=REQUIRED[track_id])
    return {"schema": "pc.execution_intake.v1", "gate": gate, "track_id": track_id,
            "evidence_scope": SCOPES[track_id], "algorithm_version": algorithm_version,
            "record_path": str(path.resolve()), "record_sha256": file_hash(path),
            "protocol_path": str(Path(protocol_path).resolve()), "protocol_sha256": file_hash(protocol_path),
            "task_manifest_path": str(Path(task_manifest_path).resolve()),
            "task_manifest_sha256": file_hash(task_manifest_path),
            "required_operations": sorted(REQUIRED[track_id]),
            "execution_status": "completed", "scientific_status": "not_evaluated",
            "limitations": "receipt schema and file bindings only; not scientific support or proof of authenticity"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--track", required=True, choices=sorted(REQUIRED))
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--gate", required=True, choices=sorted(GATES))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = intake(args.record, track_id=args.track, algorithm_version=args.algorithm,
                    protocol_path=args.protocol, task_manifest_path=args.tasks, gate=args.gate)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Execution record accepted for review; scientific gate remains not evaluated.", flush=True)


if __name__ == "__main__":
    main()
