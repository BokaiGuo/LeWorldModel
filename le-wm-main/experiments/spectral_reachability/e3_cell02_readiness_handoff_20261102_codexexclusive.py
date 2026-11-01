"""CPU-only, non-authorizing cell02 handoff from existing source and gate checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from . import e3_cell02_unified_gate_codex_20261030_v1 as gate
from . import e3_non03_host_readiness_codex_20261030_v1 as readiness
from . import agent_e3_unified_transport_20261029_v1 as transport

ROOT = readiness.RESULTS
HISTORY = ROOT / "e3_multitask_history_audit_20261102_codexexclusive_RUN01.json"
HISTORY_SHA = "07dc387383eeafa5a7efcb9e852a8ca6e8d3100327e43f06d4f1486ee6998c40"
NAME = "e3_cell02_readiness_handoff_20261102_codexexclusive"


def capture(fn):
    start = time.monotonic()
    try:
        value = fn()
        return {"status": "returned", "value": value, "elapsed_seconds": time.monotonic()-start}
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__,
                "error": str(exc)[:2048], "elapsed_seconds": time.monotonic()-start}


def history_check():
    data = HISTORY.read_bytes()
    if hashlib.sha256(data).hexdigest() != HISTORY_SHA:
        raise ValueError("multitask audit hash mismatch")
    history = json.loads(data)
    if (history.get("complete") is not True or history.get("inspected") != [0, 1, 8, 9]
            or history.get("scientific_admission") is not False
            or [r.get("state") for r in history["rows"]] != ["saved_history_bound"] * 4):
        raise ValueError("multitask audit status mismatch")
    return {"sha256": HISTORY_SHA, "inspected": history["inspected"],
            "node_boundaries": [r["nodes"] for r in history["rows"]],
            "evidence": "saved bytes, not physical or neural replay"}


def build():
    checks = {"multitask_saved_history": capture(history_check),
              "cell02_saved_setup": capture(readiness.cell02),
              "cell02_existing_gate": capture(gate.audit)}
    gate_result = checks["cell02_existing_gate"].get("value", {})
    if gate_result.get("production_ready") is not False:
        checks["cell02_existing_gate"]["unexpected_gate_state"] = True
    return {
        "schema": NAME, "cell": 2, "task": "r0.g0.h3.dDxy",
        "checks": checks, "production_authorized": False, "dispatch_ready": False,
        "scientific_admission": False, "full_saved_strict": False,
        "resource_contract": {
            "transport_cap_bytes": transport.CAP, "reserve_bytes": transport.RESERVE,
            "disk_floor_bytes": transport.FLOOR, "wall_seconds": transport.WALL,
            "category_caps_bytes": dict(transport.LIMITS),
            "gpu_memory_required": "unknown; must be measured and coordinated",
            "prior_control5_and_failure_cost": "outside this CPU audit; retain source records",
        },
        "remaining": [
            "Resolve cross-entrypoint and cross-host attempt history and fence by independent authority.",
            "Review and bind approved gate, source closure, final environment and resource plan.",
            "Coordinate GPU separately; this record does not request or authorize execution.",
            "Preserve native stop, truncation, failed attempts and costs; accept only via saved consumer.",
        ],
        "no_model_load": True, "no_gpu": True, "no_physics": True,
        "scope": "CPU saved-input checks only; no reservation, claim, launcher or scientific gate change",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if not out.is_absolute() or out.parent != ROOT or not out.name.startswith(NAME + "_"):
        raise ValueError("exclusive output path")
    result = build()
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"sha256": hashlib.sha256(data).hexdigest(),
                      "check_statuses": {k: v["status"] for k, v in result["checks"].items()},
                      "dispatch_ready": False}))
    return 0 if all(v["status"] == "returned" for v in result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
