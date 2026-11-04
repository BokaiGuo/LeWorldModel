import pytest

from experiments.spectral_reachability import p2_multi_joint_physical_saved_v2_20261102 as audit


def test_inventory_accepts_only_optional_empty_cache(tmp_path):
    (tmp_path / "session_report.json").touch()
    (tmp_path / "root").mkdir()
    expected = {"session_report.json", "root"}
    assert not audit.verify_inventory(tmp_path, expected)
    cache = tmp_path / "torchinductor_gbk"
    cache.mkdir()
    assert audit.verify_inventory(tmp_path, expected)
    (cache / "unexpected").touch()
    with pytest.raises(ValueError, match="empty"):
        audit.verify_inventory(tmp_path, expected)


def test_inventory_rejects_unrelated_entry_or_symlink(tmp_path):
    (tmp_path / "session_report.json").touch()
    expected = {"session_report.json"}
    (tmp_path / "unknown").touch()
    with pytest.raises(ValueError, match="inventory"):
        audit.verify_inventory(tmp_path, expected)
    (tmp_path / "unknown").unlink()
    (tmp_path / "torchinductor_gbk").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        audit.verify_inventory(tmp_path, expected)
