from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import (
    p2_conditional_tmpfix_finalization_consumer_20261102_exclusive as c,
)


@pytest.fixture(scope="module")
def inputs():
    report = c.prior.load_json(c.RESULT / "report.json")
    events = [json.loads(line) for line in (c.RESULT / "journal.jsonl").read_bytes().splitlines()]
    old = json.loads(c.executor.handoff.checked(c.executor.handoff.old.PREP, c.prior.OLD_PLAN_SHA))
    recipe = c.executor._checked_recipe()
    return report, events, recipe["plans"], c.executor._derived(old, recipe)


def rechain(events):
    tail = None
    for event in events:
        event["previous_sha256"] = tail
        event["event_sha256"] = c.prior.digest(
            {k: v for k, v in event.items() if k != "event_sha256"})
        tail = event["event_sha256"]


def test_actual_result_and_completion_scope():
    result = c.audit()
    assert result["status"] == "tmpfix_saved_record_verified_known_root_not_admitted"
    assert result["separate_completion_file_present"] is False
    assert result["png_bytes_and_pixels_checked"] == 222
    assert result["journal_events"] == 2410
    assert result["P2_admission"] is False


def test_finish_disagreement_rejected_even_with_rehashed_chain(inputs):
    report, original, plans, prepared = inputs
    events = deepcopy(original)
    events[-1]["errors"] = ["injected"]
    rechain(events)
    with pytest.raises(ValueError, match="final event/report"):
        c.check_lifecycle(report, events, plans, prepared)


def test_operation_return_mutation_rejected_even_with_rehashed_chain(inputs):
    report, original, plans, prepared = inputs
    events = deepcopy(original)
    op = next(x for x in events if x["kind"] == "operation_returned")
    op["receipt"]["resource"] = "close"
    rechain(events)
    with pytest.raises(ValueError, match="operation return"):
        c.check_lifecycle(report, events, plans, prepared)


def test_prefix_guard_mutation_rejected_even_with_rehashed_chain(inputs):
    report, original, plans, prepared = inputs
    events = deepcopy(original)
    next(x for x in events if x["kind"] == "feedback_prefix_guard")["guard"]["prefix_raw15_bits"] = "00"
    rechain(events)
    with pytest.raises(ValueError):
        c.check_lifecycle(report, events, plans, prepared)


def test_literal_action_mutation_rejected_even_with_rehashed_chain(inputs):
    report, original, plans, prepared = inputs
    events = deepcopy(original)
    next(x for x in events if x["kind"] == "returned_transition" and x["phase"] == "plan"
         and x["step"] == 20)["actual_bits"] = "00"
    rechain(events)
    with pytest.raises(ValueError, match="literal action"):
        c.check_lifecycle(report, events, plans, prepared)
