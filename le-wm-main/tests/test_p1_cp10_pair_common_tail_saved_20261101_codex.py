from pathlib import Path

import pytest

from experiments.spectral_reachability import p1_cp10_pair_common_tail_saved_20261101_codex as subject


def test_actual_saved_run_and_wrong_hash():
    path = Path(__file__).resolve().parents[1] / "results/p1_cp10_pair_common_tail_20261101_codex_run_CODEX01"
    if not path.is_dir():
        pytest.skip("actual physical result not present")
    with pytest.raises(ValueError, match="external report SHA"):
        subject.verify(path, "0" * 64)
    result = subject.verify(path, "f4a2225b715d62b960770edd2b084ab377d68be798365936e8efb1b55c20cd3d")
    assert result["images_verified"] == 64 and len(result["rows"]) == 4
    assert not result["P1_admission"] and not result["independent_confirmation"]
    assert result["effects"]["structured_minus_cem_interaction"] == pytest.approx(3.374366929295178)
