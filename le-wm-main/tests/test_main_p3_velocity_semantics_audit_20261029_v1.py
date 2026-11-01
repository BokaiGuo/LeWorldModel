"""Read-only actual evidence and mutation tests; no NN or simulator calls."""
from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability import main_p3_velocity_semantics_audit_20261029_v1 as m


@pytest.fixture(scope="module")
def actual():
    assert m.sha(m.SUMMARY) == m.SUMMARY_SHA
    return json.loads(m.SUMMARY.read_bytes())


def test_native_source_and_all_ten_actual_snapshots(actual):
    contract = m.source_contract(m.ENV.read_text())
    result = m.audit(actual)
    assert contract["state7_last_two"] == "agent_velocity"
    assert contract["source_reward_uses_all_seven"] and not contract["success_uses_velocity"]
    assert not contract["renderer_directly_reads_velocity"]
    assert result["total"] == 10 and len(result["pairs"]) == 5
    assert all(row["goal_velocity_norm"] > 100 for row in result["branches"])
    assert all(row["zero_goal_velocity_is_new_gate"] is False for row in result["pairs"])


@pytest.mark.parametrize("mutation", ["state_bits", "agent_velocity", "goal", "order", "snapshot", "goal_rgb"])
def test_coherent_or_incoherent_mismatches_fail_closed(actual, mutation):
    data = deepcopy(actual)
    if mutation == "state_bits":
        data["original_rows"][0]["endpoint15"]["snapshot"]["state_float64_bits"] = "00" * 56
    elif mutation == "agent_velocity":
        data["original_rows"][0]["endpoint15"]["snapshot"]["bodies"]["agent"]["velocity"][0] += 1
    elif mutation == "goal":
        data["arithmetic"]["branches"][0]["metrics"]["goal7"][5] += 1
    elif mutation == "order":
        data["arithmetic"]["branches"].reverse()
    elif mutation == "snapshot":
        data["original_rows"][0]["observations"][-1]["post"]["snapshot"]["state"][0] += 1
    else:
        data["original_rows"][0]["observations"][-1]["post"]["goal_rgb_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        m.audit(data)


def test_renderer_velocity_read_cannot_be_silently_assumed_absent():
    source = m.ENV.read_text().replace("def _render_frame(self, mode):",
                                        "def _render_frame(self, mode):\n        x = self.agent.velocity")
    assert source != m.ENV.read_text()
    with pytest.raises(ValueError, match="render method"):
        m.source_contract(source)


def test_actual_build_and_whole_verify(tmp_path, monkeypatch, capsys):
    output = tmp_path / (m.NAME + "_fixture.json")
    monkeypatch.setattr("sys.argv", ["audit", "build", "--output", str(output)])
    m.main()
    publication = json.loads(capsys.readouterr().out)
    assert publication["sha256"] == m.sha(output)
    monkeypatch.setattr("sys.argv", ["audit", "verify", "--summary", str(output), "--sha", m.sha(output)])
    m.main()
    assert json.loads(capsys.readouterr().out)["verified"] is True
    value = json.loads(output.read_bytes())
    value["audit"]["pairs"][0]["original_G"] = 0
    output.write_bytes(m.encoded(value))
    monkeypatch.setattr("sys.argv", ["audit", "verify", "--summary", str(output), "--sha", m.sha(output)])
    with pytest.raises(ValueError, match="whole deterministic"):
        m.main()
