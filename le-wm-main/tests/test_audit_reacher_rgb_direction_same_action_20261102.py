import copy
import json

import pytest

from experiments.spectral_reachability import audit_reacher_rgb_direction_same_action_20261102 as audit


@pytest.fixture(scope="module")
def sources():
    root = audit.run.ROOT / "results"
    plan = json.loads((root / "reacher_rgb_direction_same_action_PLAN_20261102_v1.json").read_text())
    report = json.loads((root / "reacher_rgb_direction_same_action_RUN_20261102_v1.json").read_text())
    events = [audit.run.paired.audit.saved.verify_journal(p)
              for p in (audit.run.paired.prior.L2_JOURNAL, audit.run.JOURNAL)]
    return plan, report, events, "5e2971cc4f1ff1a49ab5e7e5a3327a3bc5e81c60ba37271c7b4b5b4e5836abdb"


def test_saved_bindings_and_rank_denominator(sources):
    result = audit.validate(*sources)
    assert [row["pairs"] for row in result["roots"]] == [13, 13]


@pytest.mark.parametrize("field", ["actions", "physical", "residual", "score"])
def test_tampering_rejected(sources, field):
    plan, report, events, digest = sources
    report = copy.deepcopy(report)
    row = report["rows"][0]
    if field == "actions":
        row["normalized_actions"][0][0][0] += 1
    elif field == "physical":
        row["physical_final"][0] += 1
    elif field == "residual":
        row["normalized_raw_max_abs"][0] = 1e-7
    else:
        row["l2_model_cost"][0] = float("nan")
    with pytest.raises(Exception):
        audit.validate(plan, report, events, digest)
