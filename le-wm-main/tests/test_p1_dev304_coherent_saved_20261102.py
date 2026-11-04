import copy

import pytest

from experiments.spectral_reachability import p1_dev304_coherent_saved_20261102 as subject


def test_sealed_dev304_records():
    result = subject.analyze()
    assert result["frames_verified"] == 20
    assert result["same_selected_actions"]
    assert result["score_max_absolute_difference_by_iteration"][0] > 0
    assert result["score_max_absolute_difference_by_iteration"][1:] == [0] * 4
    assert all(not row["success"] for row in result["rows"])
    assert not result["p1_admission"]


@pytest.mark.parametrize("corruption", ["action", "frame", "censoring"])
def test_tampered_execution_fails_closed(monkeypatch, corruption):
    original_read = subject.run.base.read

    def modified_read(path, *args, **kwargs):
        data = original_read(path, *args, **kwargs)
        if str(path).endswith("_EXEC_DEV01/report.json"):
            data = copy.deepcopy(data)
            row = data["rows"][0]
            if corruption == "action":
                row["transitions"][0]["action"][0] += 0.1
            elif corruption == "frame":
                row["observations"][0]["pixel_sha256"] = "0" * 64
            else:
                row["status"] = "native_early_stop"
        return data

    monkeypatch.setattr(subject.run.base, "read", modified_read)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        subject.analyze()
