import ast
import json
import multiprocessing
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.spectral_reachability import e3_cell02_launch_wiring_sim_codex_20261030_v1 as wiring


def good():
    return {'history_clear': True, 'fence_live': True, 'manifest_current': True}


def run_fixture(path, **kwargs):
    return wiring.simulate(path, wiring.FIXTURE_CANDIDATE, 'a' * 64,
                           history_clear=True, fence_live=True, manifest_current=True,
                           recheck=kwargs.pop('recheck', good), **kwargs)


def test_real_inert_child_and_no_production_authorization(tmp_path):
    result = run_fixture(tmp_path)
    assert result == {'fixture_complete': True, 'production_authorized': False,
                      'production_ready': False, 'scientific_admission': False,
                      'full_saved_strict': False}
    assert (tmp_path / wiring.gate.TEST_MARKER).is_file()


@pytest.mark.parametrize('key', ['history_clear', 'fence_live', 'manifest_current'])
def test_missing_prerequisites_never_claims(tmp_path, key):
    args = good()
    args[key] = False
    with pytest.raises(ValueError):
        wiring.simulate(tmp_path, wiring.FIXTURE_CANDIDATE, 'a' * 64, recheck=good, **args)
    assert not (tmp_path / wiring.gate.TEST_MARKER).exists()


def test_old_entrypoint_direct_choice_fails_before_claim(tmp_path):
    with pytest.raises(ValueError, match='only the candidate'):
        wiring.simulate(tmp_path, wiring.gate.ENTRYPOINTS[1], 'a' * 64,
                        recheck=good, **good())
    assert not (tmp_path / wiring.gate.TEST_MARKER).exists()


def test_prior_legacy_claim_or_output_blocks(tmp_path):
    old = tmp_path / 'arbitrary-legacy-output'
    old.write_text('existing', encoding='utf8')
    with pytest.raises(FileExistsError):
        run_fixture(tmp_path)
    assert old.read_text(encoding='utf8') == 'existing'
    assert not (tmp_path / wiring.gate.TEST_MARKER).exists()


@pytest.mark.parametrize('changed', ['history_clear', 'fence_live', 'manifest_current'])
def test_lost_gate_before_dispatch_preserves_claim(tmp_path, changed):
    check = good()
    check[changed] = False
    invoked = []
    with pytest.raises(ValueError, match='boundary recheck'):
        run_fixture(tmp_path, recheck=lambda: check,
                    worker=lambda *a, **kw: invoked.append(True))
    assert not invoked
    assert (tmp_path / wiring.gate.TEST_MARKER).exists()


def test_legacy_bypass_after_claim_before_dispatch_blocks(tmp_path):
    invoked = []
    def bypass():
        # Simulates a noncooperating launcher in another process, not a real
        # production entrypoint.
        subprocess.run([sys.executable, '-B', '-s', '-c',
                        "from pathlib import Path; Path('unknown-legacy-attempt').write_text('fake')"],
                       cwd=tmp_path, check=True, timeout=10)
        return good()
    with pytest.raises(ValueError, match='history changed'):
        run_fixture(tmp_path, recheck=bypass, worker=lambda *a, **kw: invoked.append(True))
    assert not invoked
    assert (tmp_path / wiring.gate.TEST_MARKER).exists()


def test_legacy_bypass_during_stub_is_detected_but_not_prevented(tmp_path):
    def bypass(*args, **kwargs):
        (tmp_path / 'unknown-legacy-attempt').write_text('fake', encoding='utf8')
        return SimpleNamespace(returncode=0, stdout='{"fixture_complete":true}')
    with pytest.raises(ValueError, match='post-dispatch'):
        run_fixture(tmp_path, worker=bypass)
    assert (tmp_path / 'unknown-legacy-attempt').exists()
    assert (tmp_path / wiring.gate.TEST_MARKER).exists()


@pytest.mark.parametrize('returncode,stdout', [(1, '{}'), (0, '{}'), (0, 'invalid'),
                                                (0, '{"fixture_complete":1}')])
def test_partial_and_failed_child_never_complete(tmp_path, returncode, stdout):
    def worker(*args, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout)
    with pytest.raises(RuntimeError):
        run_fixture(tmp_path, worker=worker)
    assert (tmp_path / wiring.gate.TEST_MARKER).exists()


def test_lost_fence_after_stub_is_not_success(tmp_path):
    checks = iter([good(), {**good(), 'fence_live': False}])
    with pytest.raises(ValueError, match='post-dispatch'):
        run_fixture(tmp_path, recheck=lambda: next(checks))
    assert (tmp_path / wiring.gate.TEST_MARKER).exists()


def contender(path, start, queue):
    start.wait()
    try:
        run_fixture(path)
        queue.put('completed')
    except FileExistsError:
        queue.put('blocked')


def test_competing_wiring_claims_only_one(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    start, queue = ctx.Event(), ctx.Queue()
    children = [ctx.Process(target=contender, args=(str(tmp_path), start, queue))
                for _ in range(4)]
    for child in children:
        child.start()
    start.set()
    for child in children:
        child.join(30)
        assert child.exitcode == 0
    results = [queue.get(timeout=5) for child in children]
    assert results.count('completed') == 1 and results.count('blocked') == 3


def test_real_production_dispatch_cannot_be_unblocked_by_forged_evidence():
    with pytest.raises(RuntimeError, match='not fenced'):
        wiring.production_dispatch(history_clear=True, fence_live=True, manifest_current=True,
                                   authorization=True, command=['anything'])


def test_production_cli_always_nonzero():
    proc = subprocess.run([sys.executable, '-B', '-s', '-m', wiring.__name__],
                          cwd=wiring.gate.candidate.ROOT, capture_output=True, text=True, timeout=30)
    assert proc.returncode != 0
    assert json.loads(proc.stderr)['production_authorized'] is False


def test_legacy_run_does_not_call_new_gate():
    """Inspect code only; never invoke the legacy entrypoint."""
    path = wiring.gate.HERE / 'agent_e3_continuous_host_launcher_20261029_v1.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    names = {node.id for node in ast.walk(run) if isinstance(node, ast.Name)}
    assert '_run_lifecycle' in names
    assert not {'production_dispatch', 'temporary_gate', 'simulate'} & names
