import copy

import pytest

from experiments.spectral_reachability import p3_all30_stop_crosscheck_20261102_codexexclusive as audit


@pytest.fixture(scope="module")
def records():
    read = audit.all30.moving.read
    ar, br = read(audit.ALL / "report.json"), read(audit.FOUR / "report.json")
    return (read(audit.ALL_PREP), read(audit.FOUR_PREP),
            audit.all30.chain(audit.ALL / "journal.jsonl", ar["journal_chain_tail"]),
            audit.all30.moving.journal(audit.FOUR / "journal.jsonl", br["journal_chain_tail"]))


def test_real_crosscheck(records):
    result = audit.verify_records(*records, audit.ALL)
    assert result["same_candidate"]["identical_plan_frames"] == 11
    assert result["failed_branch"]["agent_xy"][1] > 512
    assert not result["failed_branch"]["accepted"]


@pytest.mark.parametrize("fault", ["candidate", "state", "guard", "post_stop"])
def test_reject_altered_evidence(records, fault):
    ap, bp, a, b = copy.deepcopy(records)
    if fault == "candidate":
        bp["plans"][2]["raw_float32_sha256"] = "wrong"
    elif fault == "state":
        next(e for e in b if e.get("branch") == 2 and e["kind"] == "fresh_observation"
             and e["receipt"]["phase"] == "plan")["receipt"]["state"][0] += 1
    elif fault == "guard":
        next(e for e in a if e.get("branch") == 7 and e["kind"] == "guard_receipt"
             and e["phase"] == "plan" and e["step"] == 8)["accepted"] = True
    else:
        a.append({"branch": 7, "kind": "returned_transition", "phase": "plan", "step": 9})
    with pytest.raises(ValueError):
        audit.verify_records(ap, bp, a, b, audit.ALL)
