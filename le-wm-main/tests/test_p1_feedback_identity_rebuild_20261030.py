import pytest

from experiments.spectral_reachability import p1_feedback_identity_rebuild_20261030 as module

OLD = module.ROOT / "results/p1_four_arm_hooked_cuda_20261030_main_RUN01"
NEW = module.ROOT / "results/p1_four_arm_spare_budget_cuda_20261030_RUN01"


@pytest.fixture
def pair():
    if not (OLD / "report.json").is_file() or not (NEW / "report.json").is_file():
        pytest.skip("saved run artifacts unavailable")
    return OLD, NEW


def test_rebuild_pairs_all_feedback_nodes(pair):
    result = module.analyze(*pair)
    assert len(result["rows"]) == 8
    assert all(row["task_hashes_reconstructed"] for row in result["rows"])
    assert not result["P1_admission"] and not result["fairness_proven"]


def test_changed_feedback_history_fails(pair, monkeypatch):
    original = module.load

    def altered(path):
        value = original(path)
        if path == NEW / "s0_random_b05_seal.json":
            value["identity"]["raw10"][0][0] += .1
        return value

    monkeypatch.setattr(module, "load", altered)
    with pytest.raises(ValueError, match="raw history/action continuity"):
        module.analyze(*pair)
