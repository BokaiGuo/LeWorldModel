import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import (
    p2_conditional_physical_saved_consumer_20261102_exclusive as consumer,
)


def test_real_saved_output_is_verified_but_completion_is_fail_closed():
    result = consumer.audit()
    assert result["journal_events"] == 2410
    assert result["png_verified"] == 222
    assert result["fixed_actions_verified"] == {"warmup_per_arm": 10, "plan_per_arm": 25}
    assert result["source_binding_verified"] is True
    assert result["completion_external_seal_present"] is False
    assert result["consumer_verified"] is False
    assert "completion.json absent" in result["failures"][0]


def test_chain_tamper_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    source = consumer.RESULT
    for name in ("binding.json", "report.json", "journal.jsonl"):
        (target / name).write_bytes((source / name).read_bytes())
    # The pinned-target restriction prevents accidentally consuming a copied
    # directory; exercise the chain primitive directly with a one-byte mutation.
    events = [json.loads(x) for x in (source / "journal.jsonl").read_text().splitlines()]
    events[1]["kind"] = "unknown"
    with pytest.raises(ValueError, match="journal event hash"):
        consumer._check_chain(events)


def test_completion_is_not_inferred_from_finished_event():
    assert not (consumer.RESULT / "completion.json").exists()
    assert consumer.audit()["status"] == "verified_saved_outputs_missing_external_completion"
