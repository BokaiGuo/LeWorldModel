"""Installer safety logic only. No packages downloaded or environments run."""
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p5_jepa_native_bootstrap_main_20261027_v1 as install


def setup(monkeypatch, tmp_path, codes=(0, 0), exceed=False):
    stage = tmp_path / 'stage'
    overlay = tmp_path / 'overlay'
    manifest = tmp_path / 'results' / install.runtime.NAME / 'manifest_frozen.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}')
    monkeypatch.setattr(install.runtime, 'ROOT', tmp_path)
    monkeypatch.setattr(install.runtime, 'paths', lambda root: (overlay, stage))
    monkeypatch.setattr(install.runtime, 'bootstrap', lambda *a: dict(environment={}, stage1_argv=['one'], stage2_argv=['two']))
    monkeypatch.setattr(install.runtime, 'verify', lambda *a: None)
    calls = []
    codes = iter(codes)
    class Process:
        pid = 12
        returncode = None
        def __init__(self, argv, **kwargs):
            calls.append(argv)
            self.code = next(codes)
        def poll(self):
            if exceed:
                return None
            self.returncode = self.code
            return self.code
        def terminate(self):
            calls.append('terminate')
        def wait(self, **kwargs):
            self.returncode = -15
            return -15
    monkeypatch.setattr(install.subprocess, 'Popen', Process)
    def budget(*args):
        if exceed:
            raise RuntimeError('over budget')
        return 39
    monkeypatch.setattr(install.runtime, 'budget', budget)
    return overlay, stage, calls


def test_success(monkeypatch, tmp_path):
    _, stage, calls = setup(monkeypatch, tmp_path)
    install.execute()
    result = json.loads((stage / 'installer_receipt.json').read_text())
    assert result['status'] == 'isolated_install_completed'
    assert calls == [['one'], ['two']]
    assert result['new_bytes'] == 39
    assert result['neural_calls'] == result['physics_calls'] == 0


def test_failed_stage_no_second_install(monkeypatch, tmp_path):
    _, stage, calls = setup(monkeypatch, tmp_path, codes=(1,))
    with pytest.raises(RuntimeError, match='stage 1 failed'):
        install.execute()
    assert calls == [['one']]
    assert json.loads((stage / 'installer_receipt.json').read_text())['status'] == 'isolated_install_failed'
    assert (stage / 'stage_1.log').exists()


def test_existing_overlay_rejected(monkeypatch, tmp_path):
    overlay, stage, calls = setup(monkeypatch, tmp_path)
    overlay.mkdir()
    with pytest.raises(FileExistsError):
        install.execute()
    assert calls == [] and not stage.exists()


def test_size_limit_terminates_and_preserves(monkeypatch, tmp_path):
    overlay, stage, calls = setup(monkeypatch, tmp_path, exceed=True)
    with pytest.raises(RuntimeError, match='over budget'):
        install.execute()
    assert calls == [['one'], 'terminate'] and overlay.exists()
    assert json.loads((stage / 'installer_receipt.json').read_text())['status'] == 'isolated_install_failed'
