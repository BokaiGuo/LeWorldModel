from copy import deepcopy
import pytest

from experiments.spectral_reachability import codex_e1_arbitrary_reconstruction_20261101_exclusive as audit


@pytest.fixture(scope="module")
def inputs():
    return audit.load()


def test_real_saved_sources_reconstruct(inputs):
    result = audit.audit(*inputs)
    assert len(result["rows"]) == 2
    assert all(row["candidate_count"] == 20 for row in result["rows"])
    assert result["historical_receipt"] is None
    assert not result["equal_budget_certified"]


@pytest.mark.parametrize("mutation", ["goal", "target", "digest", "identity"])
def test_tampering_rejected(inputs, mutation):
    old, new, tasks = deepcopy(inputs)
    if mutation == "goal":
        tasks[0]["task"]["goal"][0] += 1
    elif mutation == "target":
        new["rows"][0]["target_data_sha256"] = "0" * 64
    elif mutation == "digest":
        old["targets"][0]["arbitrary_sha256"] = "0" * 64
    else:
        tasks[0]["task_id"] = "wrong"
    with pytest.raises(ValueError):
        audit.audit(old, new, tasks)


def test_exclusive_output_refuses_overwrite(inputs, tmp_path, monkeypatch):
    output = tmp_path / "occupied.json"
    output.write_text("preserve")
    monkeypatch.setattr(audit, "OUTPUT", output)
    monkeypatch.setattr(audit, "load", lambda: inputs)
    with pytest.raises(FileExistsError):
        audit.main()
    assert output.read_text() == "preserve"
