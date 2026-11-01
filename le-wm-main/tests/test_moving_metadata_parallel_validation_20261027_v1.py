"""Mock subprocess tests; no actual suite/model/physics launched here."""
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import moving_metadata_parallel_validation_20261027_v1 as m


@pytest.mark.parametrize('code,expected', [(0, 'passed'), (1, 'failed')])
def test_subprocess_isolated_and_fixed(monkeypatch, code, expected):
    def fake(command, **kwargs):
        assert command[:7] == [m.sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider']
        assert kwargs['cwd'] == m.ROOT and kwargs['timeout'] == 180
        assert kwargs['env']['PYTEST_ADDOPTS'] == ''
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        return SimpleNamespace(returncode=code, stdout=b'mock test summary', stderr=b'')
    monkeypatch.setattr(m.subprocess, 'run', fake)
    row = m.run_suite(m.SUITES[0])
    assert row['status'] == expected and row['test_source_unchanged']


def test_timeout_not_success(monkeypatch):
    def fake(*args, **kwargs):
        raise m.subprocess.TimeoutExpired(args[0], 180, output=b'partial')
    monkeypatch.setattr(m.subprocess, 'run', fake)
    row = m.run_suite(m.SUITES[0])
    assert row['status'] == 'timeout' and row['returncode'] is None


def test_unknown_suite_cannot_launch(monkeypatch):
    monkeypatch.setattr(m.subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(ValueError):
        m.run_suite('../../experiment.py')


def test_existing_output_rejected_before_tests(monkeypatch):
    monkeypatch.setattr(m, 'run_suite', lambda *a: pytest.fail('must not launch'))
    with pytest.raises(FileExistsError):
        m.main(['--output', m.__file__])


def test_worker_cap():
    with pytest.raises(SystemExit):
        m.main(['--workers', '5', '--output', '/tmp/never-written.json'])
