import pytest

from experiments.spectral_reachability import p1_spare_budget_pair_audit_20261030 as audit

OLD = audit.ROOT / "results/p1_four_arm_hooked_cuda_20261030_main_RUN01"
NEW = audit.ROOT / "results/p1_four_arm_spare_budget_cuda_20261030_RUN01"


@pytest.fixture
def pair():
    if not (OLD / "report.json").is_file() or not (NEW / "report.json").is_file():
        pytest.skip("known-root saved reports unavailable")
    return OLD, NEW


def test_pair_evaluates_fixed_candidate_changes(pair):
    result = audit.analyze(*pair)
    assert len(result["rows"]) == 8
    assert all(row["old_winner"] == row["new_winner"] for row in result["rows"])
    assert all(delta < 0 for row in result["rows"] for delta in row["extra_candidate_score_delta"])
    assert all(not row["whole_task_hash_equal"] for row in result["rows"])
    assert not result["P1_admission"]


def test_mismatched_history_fails_closed(pair, monkeypatch):
    original = audit.identity

    def wrong(path):
        value = dict(original(path))
        if "_spare" in str(path):
            value["history"] = "0" * 64
        return value

    monkeypatch.setattr(audit, "identity", wrong)
    with pytest.raises(ValueError, match="planning identity mismatch"):
        audit.analyze(*pair)
