import hashlib
import json
import subprocess
import sys

import pytest

from experiments.spectral_reachability import e3_cell03_recovery_manifest_codex_20261030_v2 as m


def test_exclusive_manifest_and_sha(tmp_path):
    path = tmp_path / 'manifest.json'
    body = {'attempt': 'test', 'approved': False}
    sha = m.write_exclusive(path, body)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_bytes()) == body
    with pytest.raises(FileExistsError):
        m.write_exclusive(path, {'attempt': 'changed'})
    assert json.loads(path.read_bytes()) == body


def test_dangling_symlink_rejected(tmp_path):
    path = tmp_path / 'manifest.json'
    path.symlink_to(tmp_path / 'missing')
    with pytest.raises(FileExistsError):
        m.write_exclusive(path, {})


def test_run_is_fail_closed():
    p = subprocess.run([sys.executable, '-B', '-m', m.__name__, 'run'],
                       capture_output=True, text=True)
    assert p.returncode != 0
    assert 'no P2 GPU coordination' in p.stderr


def test_canonical_manifest_rejects_nan():
    with pytest.raises(ValueError):
        m.raw({'x': float('nan')})
