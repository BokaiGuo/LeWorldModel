import pytest

from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def test_detects_changed_extra_dependency(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"a": 1}')
    recorded = seal.capture([path])
    seal.verify(recorded)
    path.write_text('{"a": 2}')
    with pytest.raises(ValueError, match="dependency changed"):
        seal.verify(recorded)


def test_rejects_missing_required_module():
    recorded = seal.capture()
    recorded["files"].pop(next(iter(recorded["files"])))
    with pytest.raises(ValueError, match="missing required module"):
        seal.verify(recorded)
