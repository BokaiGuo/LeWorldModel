from copy import deepcopy
import pytest

from experiments.spectral_reachability import codex_p2_feedback_candidate_physical_20261030_v1 as m


@pytest.fixture(scope="module")
def actual():
    return m.material(0)[0]


def test_actual_material_shared_feedback_and_raw(actual):
    m.validate_execution(actual)
    assert [p["policy"] for p in actual["plans"]] == ["selected", "fixed_index_1"]
    assert actual["model_jobs"][0] == actual["model_jobs"][1]
    assert actual["plans"][0]["commands"][:15] == actual["plans"][1]["commands"][:15]
    assert actual["plans"][1]["candidate_index"] == 1
    assert not m.contract()["equal_total_compute"]


@pytest.mark.parametrize("mutation", ["candidate", "action", "context", "prefix", "guard"])
def test_tampering_refused(actual, mutation):
    p = deepcopy(actual)
    if mutation == "candidate":
        p["plans"][1]["candidate_index"] = 2
    elif mutation == "action":
        p["plans"][1]["commands"][15][0] += .1
    elif mutation == "context":
        p["observed_context"]["latent_history"][0][0] += 1
    elif mutation == "prefix":
        p["prefix_reference"].pop()
    else:
        p["planner_authority"]["candidate_prepare_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        m.validate_execution(p)


def test_no_suffix_dispatch_without_prefix_guard(actual):
    row = {"policy":"selected", "base_index":0, "feedback_guard":None}
    plan = actual["plans"][0]
    events = [{"kind":"operation_dispatch_intent", "receipt":{"resource":"public_step", "phase":"plan", "step":16}}]
    with pytest.raises(ValueError):
        m.audit_feedback(row, plan, events, actual)


def test_auth_before_any_input_or_factory(monkeypatch):
    monkeypatch.setattr(m, "verify", lambda *a: pytest.fail("verify called"))
    with pytest.raises(ValueError):
        m.run("x", "y", "z", lambda: pytest.fail("factory called"), authorization="wrong")
