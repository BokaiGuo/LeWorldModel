from copy import deepcopy

import pytest

from experiments.spectral_reachability import codex_p2_joined_native_physical_20261030_v1 as m


@pytest.fixture(scope="module")
def prepared():
    return m.prepare()


def test_actual_eight_plan_binding(prepared):
    plans = prepared["plans"]
    assert len(plans) == 8
    assert sum(p["planned_commands"] for p in plans) == 80
    assert [p["root_index"] for p in plans] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert all(len(p["commands"]) == 10 for p in plans)
    assert all(p["selected_predicted_context"] != p["predictor_context"] for p in plans)


def test_reject_existing_output_without_factory(tmp_path, monkeypatch, prepared):
    monkeypatch.setattr(m, "prepare", lambda: prepared)
    output = tmp_path / (m.NAME + "_existing")
    output.mkdir()
    with pytest.raises(ValueError, match="exclusive output"):
        m.execute(output)
    assert not list(output.iterdir())


def test_mismatch_saved_join_rejected_before_physics(tmp_path, monkeypatch):
    fake = tmp_path / "join.json"
    fake.write_text("{}")
    monkeypatch.setattr(m, "JOIN", fake)
    with pytest.raises(ValueError, match="join SHA"):
        m.prepare()


def test_action_or_identity_tampering_rejected(prepared):
    plan = deepcopy(prepared["plans"][0])
    plan["commands"][0][0] = 2.0
    with pytest.raises(ValueError):
        m.native.validate_plan(plan)
    plan = deepcopy(prepared["plans"][0])
    plan["root_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="plan digest"):
        m.native.validate_plan(plan)


def test_public_fixture_production_entry_stays_blocked(tmp_path):
    from experiments.spectral_reachability import codex_p2_knownroot_physical_bridge_20261030_v2 as old
    with pytest.raises(RuntimeError, match="BLOCKED"):
        old.run_physical(packet={}, output=tmp_path, factory=lambda _: None, authorization=old.AUTH)


def test_constructor_failure_saves_full_blocked_census(tmp_path, monkeypatch, prepared):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    monkeypatch.setattr(m, "prepare", lambda: prepared)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic constructor failure")
    monkeypatch.setattr(base.gym, "make", fail)
    output = tmp_path / (m.NAME + "_failed")
    report = m.execute(output)
    assert report["status"] == "partial_or_failed"
    assert len(report["rows"]) == 8 and all(r["status"] == "blocked" for r in report["rows"])
    assert "synthetic constructor failure" in str(report["failures"])
    assert m.consume(output, m.joiner.host.frontier.file_sha(output / "report.json"))["status"] == "partial_or_failed"
