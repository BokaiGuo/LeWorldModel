import pytest

from experiments.spectral_reachability import codex_p3_quota_source_identity_20261030_v1 as s


def test_identity_drift_and_symlink(tmp_path):
    path = tmp_path / "weight"
    path.write_bytes(b"weight")
    result = s.inventory([path])
    s.verify(result)
    path.write_bytes(b"change")
    with pytest.raises(ValueError, match="drift"):
        s.verify(result)
    link = tmp_path / "alias"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="canonical"):
        s.file_identity(link)


def test_duplicate_files_rejected(tmp_path):
    path = tmp_path / "one"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="unique"):
        s.inventory([path, path])
