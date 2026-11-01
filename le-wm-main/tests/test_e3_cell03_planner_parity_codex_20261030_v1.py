import json
import os

import pytest

from experiments.spectral_reachability import e3_cell03_planner_parity_codex_20261030_v1 as m


def test_confirmation_rejects_before_verification(monkeypatch):
    monkeypatch.setattr(m, 'verify', lambda _: pytest.fail('must reject before verify'))
    with pytest.raises(ValueError, match='authorization'):
        m.run('anything', 'wrong')


def test_manifest_drift_and_tamper(tmp_path, monkeypatch):
    path = tmp_path/'manifest'
    monkeypatch.setattr(m, 'MANIFEST', path)
    path.write_bytes(m.base.raw({'bound': 1}))
    digest = m.base.sha(path)
    monkeypatch.setattr(m, 'prepare', lambda: {'bound': 1})
    assert m.verify(digest) == {'bound': 1}
    monkeypatch.setattr(m, 'prepare', lambda: {'bound': 2})
    with pytest.raises(ValueError, match='drift'):
        m.verify(digest)
    with pytest.raises(ValueError, match='SHA'):
        m.verify('0'*64)


def test_failure_keeps_claim_and_error(tmp_path, monkeypatch):
    out = tmp_path/'run'
    out.mkdir()
    monkeypatch.setattr(m, 'OUTPUT', out)
    monkeypatch.setattr(m, 'verify', lambda _: (_ for _ in ()).throw(ValueError('source drift')))
    with pytest.raises(ValueError, match='source drift'):
        m.execute('fixed')
    assert json.loads((out/'failure.json').read_bytes())['message'] == 'source drift'
    assert (out/'child.claim').is_file()
    with pytest.raises(FileExistsError):
        m.execute('fixed')


def test_child_requires_lock():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-B', '-m', m.MODULE, 'child', '--sha256', 'x'],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1'),
                            capture_output=True, text=True)
    assert result.returncode != 0 and 'inherited lock required' in result.stderr
