"""Development-only durable bridge for the existing common optimizer.

The fixed callback allowance is not a full resource cap. No initializer,
native objective, CUDA lock, or physical execution is authenticated here.
"""
import hashlib
import json
import os
from pathlib import Path

from . import p2_common_objective_ledger_20261102_exclusive as ledger_core
from . import p2_sra_common_optimizer_bridge_20261102_exclusive as optimizer

NAME = "p2_common_optimizer_durable_adapter_20261102_exclusive"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def execute_fixture(initializers, make_evaluate, output):
    """Run a callback fixture with fixed optimizer limits and durable events.

    This ABI is deliberately fixture-only and cannot authorize native execution.
    An interrupted process may leave only a journal; no completion is inferred.
    """
    if not callable(make_evaluate) or type(initializers) is not dict or tuple(initializers) != optimizer.ARMS:
        raise ValueError("fixed arms and callback factory required")
    initial = {arm: optimizer.raw_actions(initializers[arm]) for arm in optimizer.ARMS}
    directory = Path(output)
    directory.mkdir(mode=0o700)  # exclusive, including previously failed attempts
    ledger = ledger_core.Ledger(directory / "journal.jsonl")
    counts = {arm: {"attempted_scores": 0, "attempted_gradients": 0,
                    "returned_scores": 0, "returned_gradients": 0}
              for arm in optimizer.ARMS}
    status, result, failure = "failed", None, None
    try:
        ledger.record("fixture_contract", {
            "initial_action_sha256": {arm: sha(initial[arm].tobytes()) for arm in optimizer.ARMS},
            "score_cap_per_arm": optimizer.STEPS + 1,
            "gradient_cap_per_arm": optimizer.STEPS,
            "fixture_only": True,
        })
        evaluate = make_evaluate(ledger)
        if not callable(evaluate):
            raise ValueError("callback factory must return callable")

        def emit(event):
            ledger.record("optimizer_" + event["kind"], event)

        def guarded(arm, action, gradient_required):
            c = counts[arm]
            if (c["attempted_scores"] >= optimizer.STEPS + 1 or
                    (gradient_required and c["attempted_gradients"] >= optimizer.STEPS)):
                raise ValueError("optimizer callback cap exceeded")
            c["attempted_scores"] += 1
            c["attempted_gradients"] += int(gradient_required)
            value = evaluate(arm, action, gradient_required)
            # optimize_pair validates the return before its durable returned event.
            return value

        def record_return(event):
            emit(event)
            if event["kind"] == "returned":
                c = counts[event["arm"]]
                c["returned_scores"] += 1
                c["returned_gradients"] += int(event["gradient_required"])

        result = optimizer.optimize_pair(initial, guarded, record_return)
        expected = {"attempted_scores": 9, "attempted_gradients": 8,
                    "returned_scores": 9, "returned_gradients": 8}
        if any(c != expected for c in counts.values()):
            raise ValueError("incomplete callback census")
        status = "fixture_completed"
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    finally:
        ledger.close()

    verified = ledger_core.verify_ledger(directory / "journal.jsonl")
    if verified["outstanding"]:
        raise ValueError("incomplete native operation tickets")
    report = {
        "schema": NAME + ".v1", "status": status, "failure": failure,
        "callback_counts": counts, "optimizer_result": result,
        "ledger_tail_sha256": verified["tail_sha256"],
        "ledger_file_sha256": sha((directory / "journal.jsonl").read_bytes()),
        "fixture_only": True, "production_execution_authorized": False,
        "candidate_note": "Nine evaluated iterates per arm, not the same action candidate set.",
        "equal_total_compute": False, "independent_confirmation": False,
        "P1_admission": False, "P2_admission": False,
        "unmeasured": ["initializer cost", "native model operations", "FLOPs",
                       "energy", "IO", "physical execution"],
    }
    raw = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with (directory / "report.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return report
