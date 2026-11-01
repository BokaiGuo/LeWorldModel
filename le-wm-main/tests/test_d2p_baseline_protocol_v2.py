"""Metadata fixtures only: no model, device allocation, or efficacy evidence."""
import pytest

from experiments.spectral_reachability.d2p_baseline_protocol_v2 import (
    BINDINGS, RESOURCES, BudgetLedger, compare_protocols, fingerprint,
    method_identity, protocol,
)


def vector(n=0, **overrides):
    return dict.fromkeys(RESOURCES, n) | overrides


def frozen():
    bindings = dict.fromkeys(BINDINGS, "a" * 64)
    bindings["outcome_split"] = "b" * 64
    return protocol(bindings=bindings, feedback_steps=[4, 8],
                    resource_caps=vector(100), primary_resources=["predict_samples"])


def identity(label="proxy"):
    return method_identity(name="restart_adam_not_GRASP", label=label,
                           implementation_sha256="c" * 64)


def ledger(p=None):
    return BudgetLedger(p or frozen(), identity(), execution_backend="host_cuda")


def test_hash_canonical_copy_isolation():
    p = frozen()
    assert fingerprint(p) == fingerprint(dict(reversed(list(p.items()))))
    book = ledger(p)
    original = fingerprint(p)
    p["bindings"]["root"] = "d" * 64
    assert book.receipt()["protocol_sha256"] == original


@pytest.mark.parametrize("binding", BINDINGS)
def test_every_binding_mismatch_blocks_matching(binding):
    p, q = frozen(), frozen()
    q["bindings"][binding] = "d" * 64
    result = compare_protocols(p, q)
    assert not result["matched_protocol_and_primary_caps"]
    assert f"bindings.{binding}" in result["differences"]


@pytest.mark.parametrize("bad", [None, "unknown", "A" * 64, True, 0])
def test_unknown_is_not_known_matching(bad):
    p = frozen()
    p["bindings"]["goal_metric"] = bad
    with pytest.raises(ValueError):
        compare_protocols(p, p)


def test_feedback_and_primary_ceiling_not_equal_actual_compute():
    p, q = frozen(), frozen()
    result = compare_protocols(p, q)
    assert result["matched_protocol_and_primary_caps"]
    assert not result["equal_actual_compute"]
    assert not result["scientific_stage_admission"]
    q["feedback_steps"] = []
    assert not compare_protocols(p, q)["matched_protocol_and_primary_caps"]
    q = frozen()
    q["resource_caps"]["predict_samples"] += 1
    assert not compare_protocols(p, q)["matched_protocol_and_primary_caps"]
    q = frozen()
    q["resource_caps"]["jacobian_vjp_rows"] += 1
    result = compare_protocols(p, q)
    assert result["matched_protocol_and_primary_caps"]
    assert not result["nonprimary_caps_equal"]


@pytest.mark.parametrize("bad", [True, -1, 1.5, float("nan"), "3", None])
def test_bad_resource_counts(bad):
    p = frozen()
    p["resource_caps"]["predict_samples"] = bad
    with pytest.raises(ValueError):
        ledger(p)


def test_no_split_alias_missing_vector_or_implicit_schedule():
    p = frozen()
    p["bindings"]["outcome_split"] = p["bindings"]["calibration_split"]
    with pytest.raises(ValueError):
        ledger(p)
    for schedule in ([4, 4], [8, 4], [True], None):
        p = frozen()
        p["feedback_steps"] = schedule
        with pytest.raises(ValueError):
            ledger(p)
    p = frozen()
    del p["resource_caps"]["encoder_samples"]
    with pytest.raises(ValueError):
        ledger(p)


def test_faithfulness_claim_requires_audit_and_never_verifies_it():
    with pytest.raises(ValueError):
        identity("faithfulness_audited")
    audited = method_identity(name="GRASP", label="faithfulness_audited",
                              implementation_sha256="c" * 64, audit_sha256="d" * 64)
    assert not audited["official_reproduction_verified_here"]
    assert identity()["label"] == "proxy"
    with pytest.raises(ValueError):
        identity("official")


@pytest.mark.parametrize("backend", ["cpu", "cuda", "remote_cuda", None])
def test_backend_requires_explicit_host_cuda(backend):
    with pytest.raises(ValueError):
        BudgetLedger(frozen(), identity(), execution_backend=backend)


def test_successful_measured_full_vector_still_not_efficacy():
    book = ledger()
    for step in (4, 8):
        ticket = book.reserve("all proposals including rejected candidates", vector(10))
        book.complete(ticket, vector(3))
        book.feedback(step)
    receipt = book.receipt(finalize=True)
    assert receipt["completed"]
    assert receipt["charged_resources"] == vector(6)
    assert not receipt["physical_efficacy_verified"]
    assert not receipt["scientific_stage_admission"]
    receipt["events"].clear()
    assert len(book.receipt()["events"]) == 2
    with pytest.raises(RuntimeError):
        book.reserve("late work", vector())


def test_over_budget_rejected_before_dispatch_and_poisoned():
    book = ledger()
    with pytest.raises(ValueError, match="before dispatch"):
        book.reserve("oversized", vector(predict_samples=101))
    assert not book.receipt(finalize=True)["completed"]
    assert book.receipt()["failed"]


def test_failure_keeps_attempted_reservation_and_blocks_incumbent_recovery():
    book = ledger()
    t = book.reserve("failed solve", vector(9))
    book.complete(t, succeeded=False)
    receipt = book.receipt(finalize=True)
    assert receipt["charged_resources"] == vector(9)
    assert not receipt["completed"]
    with pytest.raises(RuntimeError):
        book.reserve("return best ever as success", vector())


def test_overrun_and_invalid_measurement_fail_closed():
    for actual in (vector(predict_samples=11), {"predict_samples": 1}):
        book = ledger()
        t = book.reserve("jacobian and line search", vector(10))
        with pytest.raises(ValueError):
            book.complete(t, actual)
        assert book.receipt()["failed"]
        assert not book.receipt(finalize=True)["completed"]


def test_pending_missing_feedback_empty_or_wrong_feedback_never_complete():
    book = ledger()
    assert not book.receipt(finalize=True)["completed"]
    book = ledger()
    book.reserve("pending", vector(1))
    assert not book.receipt(finalize=True)["completed"]
    book = ledger()
    t = book.reserve("done", vector(1))
    with pytest.raises(RuntimeError):
        book.reserve("overlap", vector(1))
    book.complete(t, vector(1))
    assert not book.receipt(finalize=True)["completed"]
    book = ledger()
    with pytest.raises(ValueError):
        book.feedback(8)
    assert book.receipt()["failed"]


def test_actual_not_reserved_accumulates_and_ticket_cannot_replay():
    book = ledger()
    t = book.reserve("first", vector(90))
    book.complete(t, vector(1))
    with pytest.raises(ValueError):
        book.complete(t, vector(1))
    t = book.reserve("second", vector(90))
    book.complete(t, vector(1))
    assert book.receipt()["charged_resources"] == vector(2)
