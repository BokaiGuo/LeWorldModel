from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_blind_uphill_bank_saved_20261101_exclusive as m


def test_saved_report_and_tampered_hash():
    folder = Path(__file__).resolve().parents[1] / "results/codex_p1_blind_uphill_bank_20261101_exclusive_RUN_CODEXBLIND01"
    if not folder.exists():
        pytest.skip("host output unavailable")
    result = m.verify(folder, "35935f3b45e755afac925c1c622e3896899029ee1b6490c57cd2a147b146e199")
    assert [r["selected_index"] for r in result["rows"]] == [3, 0]
    assert all(not r["terminal_improves_root"] for r in result["rows"])
    with pytest.raises(ValueError, match="report SHA"):
        m.verify(folder, "0" * 64)
