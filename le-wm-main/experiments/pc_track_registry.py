"""Explicit, read-only eligibility checks before adding gate evidence."""
from experiments.pc_input_permissions import RGB, STATE, SCOPES, track, validate_evidence


def eligible(record, *, target_track, algorithm_version, protocol_hash, task_manifest_hash):
    if record.get("execution_status") != "completed":
        raise ValueError("partial/failed executions cannot be registered as completed evidence")
    if record.get("task_manifest_hash") != task_manifest_hash:
        raise ValueError("task binding mismatch")
    validate_evidence(record, target_track=target_track, algorithm_version=algorithm_version,
                      protocol_hash=protocol_hash)
    # A smoke manifest must never declare scientific passage on its own.
    if record.get("scientific_status") != "not_evaluated":
        raise ValueError("scientific decisions require a separate gate review")
    expected = {
        RGB: {"encode": {"pixels"}, "score": {"pixels", "goal", "action", "candidate_actions"}},
        STATE: {"predict": {"visual", "proprio", "candidate_actions"}},
    }
    for receipt in record["input_receipts"]:
        allowed = expected[target_track].get(receipt["operation"])
        if allowed is None or set(receipt["inputs"]) != allowed:
            raise ValueError("runtime receipt contains forbidden/missing input channels")
    return True


def empty_gate_ledger(track_id, algorithm_version):
    track(track_id)
    if not algorithm_version:
        raise ValueError("algorithm version required")
    return {"track_id": track_id, "algorithm_version": algorithm_version,
            "evidence_scope": SCOPES[track_id],
            "supports_primary_rgb_claim": track_id == RGB,
            "gates": {f"PC{i}": {"execution_status": "not_started",
                                "scientific_status": "not_evaluated", "evidence": []}
                      for i in range(6)}}
