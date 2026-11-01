from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_structured_root1_cp10_saved_20261101_exclusive as m


def test_saved_seals_and_bad_report_hash():
    folder = Path(__file__).resolve().parents[1] / "results/codex_p1_structured_root1_cp10_20261101_exclusive_RUN_CODEXROOT1DEV01"
    if not folder.exists():
        pytest.skip("host run not present")
    sha = "762eb75e9338abf38d4e546a13ad04ae6a480255a0d5430cdab67407810ee6d6"
    result = m.audit(folder, sha)
    assert result["steps"] == 8 and result["sealed_boundaries"] == [0, 5]
    with pytest.raises(ValueError, match="report hash"):
        m.audit(folder, "0" * 64)
