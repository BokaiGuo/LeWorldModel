from copy import deepcopy
import pytest
from experiments.spectral_reachability import codex_p2_candidate_domain_censor_20261030_v1 as m


@pytest.fixture(scope="module")
def saved():
    path=m.m.ROOT/"results/codex_p2_feedback_candidate_physical_20261030_v1_RUN_JOB0"
    files=m.m.saved_artifacts(path)
    return m.m.native.decode(files["report.json"]),[m.m.native.decode(x) for x in files["journal.jsonl"].splitlines()],files


def test_actual_complete_adapted_saved_review():
    result=m.build()
    assert result["slot_count"]==8 and result["known_roots"]==2
    assert [x["censor"]["step"] for x in result["jobs"][:2]]==[29,21]
    assert [x["censor"]["agent_color_pixels"] for x in result["jobs"][:2]]==[2,9]
    assert result["original_failures_preserved"] and not result["P2_admission"]
    assert result["counts"]["public_step"]=={"attempted":310,"completed":310}


@pytest.mark.parametrize("change",["error","actual_bits","state","terminal","image","next_step"])
def test_reject_unproven_censor(saved,change):
    report,events,files=deepcopy(saved)
    if change=="error":
        report["rows"][1]["first_error"]["message"]="other"
    elif change=="image":
        files["b1_plan_29_0.png"]=b"not a png"
    elif change=="next_step":
        events.append({"kind":"operation_dispatch_intent","branch":1,"index":len(events),
                       "receipt":{"resource":"public_step"}})
    else:
        transition=[x for x in events if x["kind"]=="returned_transition" and x["branch"]==1 and x["phase"]=="plan"][-1]
        if change=="actual_bits":transition["actual_bits"]="0000000000000000"
        elif change=="terminal":transition["terminated"]=True
        else:transition["state"][0]=500.
    with pytest.raises((ValueError, KeyError)):
        m.incident(report,events,files)
