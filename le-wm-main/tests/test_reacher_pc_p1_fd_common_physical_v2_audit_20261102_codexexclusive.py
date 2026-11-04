import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import audit_reacher_pc_p1_fd_common_physical_v2_20261102_codexexclusive as audit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sources():
    paths = (
        "results/reacher_pc_fd_common_physical_20261102_v2/report.json",
        "results/reacher_pc_fd_common_physical_20261102_PLAN_v1.json",
        "results/reacher_pc_fd_suffix_cuda_20261102_v1/report.json",
        "results/reacher_pc_fd_suffix_saved_audit_20261102_v1.json",
    )
    return [json.loads((ROOT / path).read_text()) for path in paths]


def test_sealed_v2_counts_and_pairing(sources):
    result = audit.verify(*sources)
    assert result["expected"] == result["complete"] == 20
    assert [len(row["paired_final_deltas"]) for row in result["roots"]] == [5, 5]
    assert result["p1_admission"] is False


@pytest.mark.parametrize("change,match", [
    (lambda s: s[0].__setitem__("plan_sha256", "wrong"), "lineage"),
    (lambda s: s[0].__setitem__("complete_branches", 19), "denominator"),
    (lambda s: s[0]["rows"][0]["branches"][0]["raw_actions"][0].__setitem__(0, .12345), "action binding"),
    (lambda s: s[0]["rows"][0]["branches"][0]["trajectory"][1]["action"].__setitem__(0, .12345), "trajectory action"),
    (lambda s: s[0]["rows"][0]["branches"][0]["trajectory"][1].__setitem__("qpos_distance", 99), "distance"),
    (lambda s: s[0]["rows"][0]["branches"][0]["trajectory"][1]["rgb"].__setitem__("sha256", "wrong"), "RGB hash"),
    (lambda s: s[0]["rows"][0]["branches"][0].__setitem__("stop", {"step": 15}), "stop or length"),
    (lambda s: s[0]["rows"][0]["branches"][0]["identity"].__setitem__("reset_rgb_sha256", "wrong"), "identity"),
    (lambda s: s[2]["rows"][0]["compiler_receipt"].__setitem__("strict_selected_index", 4), "selection"),
])
def test_rejects_tampered_saved_output(sources, change, match):
    change(sources)
    with pytest.raises(ValueError, match=match):
        audit.verify(*sources)
