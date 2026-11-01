"""Explicit binding/receipt tests only; no neural or simulator invocation."""
import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import p5_prospective_c1_native_fd_20261027_v2 as m


def test_v2_uses_unchanged_math_and_core():
    assert m.base.policy()['epsilons'] == [.01, .005]
    assert m.base.policy()['completed']['predictor'] == 40
    assert sum(m.base.policy()['completed'].values()) == 40
    assert m.base.policy()['new_nominal_predictions'] == 0
    tree = ast.parse(Path(m.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert any(isinstance(n.func, ast.Attribute) and n.func.attr == 'cuda_core' for n in calls)
    assert not any(isinstance(n, ast.FunctionDef) and n.name in ('cuda_core', 'fd_loop', 'perturbations') for n in tree.body)


def test_merge_rejects_conflicting_source_or_input_hash():
    assert m.merge_bindings({'a': 'hash'}, {'a': 'hash', 'b': 'new'}) == {'a': 'hash', 'b': 'new'}
    with pytest.raises(ValueError, match='conflicting'):
        m.merge_bindings({'a': 'old'}, {'a': 'changed'})


def test_explicit_receipt_path_not_hardcoded(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    for name in ('compiler_v2_run', 'compiler_another_explicit_run'):
        receipt = tmp_path/'results'/name/'compiler_receipt.json'
        records = {'context': dict(path=str(receipt.parent/'context.npy'))}
        assert m.check_receipt_location(receipt, records) == receipt


@pytest.mark.parametrize('fault', ['outside', 'name', 'mixed_context', 'wrong_suffix'])
def test_receipt_and_context_directory_mismatch_rejected(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    receipt = tmp_path/'results'/'compiler_v2'/'compiler_receipt.json'
    records = {'context': dict(path=str(receipt.parent/'context.npy'))}
    if fault == 'outside': receipt = tmp_path/'elsewhere'/'compiler_receipt.json'
    if fault == 'name': receipt = receipt.with_name('unrelated.json')
    if fault == 'mixed_context': records['context']['path'] = str(tmp_path/'results'/'compiler_v1'/'context.npy')
    if fault == 'wrong_suffix': records['context']['path'] = str(receipt.parent/'context.pkl')
    with pytest.raises(ValueError): m.check_receipt_location(receipt, records)


def journal(tmp_path):
    receipt = dict(status='selected_packet_replayed')
    ledger = m.compiler.Ledger(tmp_path, receipt)
    ledger.event('begin')
    ledger.event('terminal', status=receipt['status'])
    path = tmp_path/'journal.jsonl'
    receipt['journal_sha256'] = m.file_sha(path)
    return path, receipt


def test_actual_ledger_journal_chain_without_neural_work(tmp_path):
    m.verify_journal(*journal(tmp_path))


@pytest.mark.parametrize('fault', ['file_hash', 'sequence', 'previous', 'entry_hash', 'terminal', 'status', 'empty'])
def test_journal_tamper_rejected(tmp_path, fault):
    path, receipt = journal(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == 'file_hash': receipt['journal_sha256'] = 'wrong'
    else:
        if fault == 'sequence': rows[1]['sequence'] = 12
        if fault == 'previous': rows[1]['previous_sha256'] = 'wrong'
        if fault == 'entry_hash': rows[1]['entry_sha256'] = 'wrong'
        if fault == 'terminal': rows[1]['event']['kind'] = 'not_terminal'
        if fault == 'status': receipt['status'] = 'compiler_failed'
        if fault == 'empty': rows = []
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        receipt['journal_sha256'] = m.file_sha(path)
    with pytest.raises(ValueError): m.verify_journal(path, receipt)


def test_old_receipt_cannot_bind_new_compiler_manifest(tmp_path):
    old, new = tmp_path/'old_manifest.json', tmp_path/'new_manifest.json'
    old.write_text('{"bridge":"old_executor"}')
    new.write_text('{"bridge":"executor_v2"}')
    receipt = dict(status='selected_packet_replayed', runner_schema=m.compiler.TAG,
                   manifest_file_sha256=m.file_sha(old))
    with pytest.raises(ValueError, match='same compiler/bridge'):
        m.base.reference_arrays(receipt, dict(path=new, manifest={}), {})


def test_source_prepare_cannot_start_cuda(monkeypatch):
    monkeypatch.setattr(m, 'verify', lambda *args: (dict(reference=None, status='source_prepared_awaiting_explicit_compiler_manifest'), None))
    with pytest.raises(ValueError, match='binding required'):
        m.run('not_read', 'not_read', 'must_not_create')


def test_module_no_torch_no_old_run_default_no_archived_reader():
    text = Path(m.__file__).read_text(); tree = ast.parse(text)
    assert 'cuda_main_v1' not in text and 'cuda_main_v2' not in text
    for n in ast.walk(tree):
        if isinstance(n, ast.Import): assert all(x.name != 'torch' for x in n.names)
        if isinstance(n, ast.ImportFrom): assert n.module != 'torch'
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            assert n.func.attr not in {'read_arrays', 'row_inputs', 'validate_cuda', 'choose', 'frozen_branches', 'step', 'reset', 'render'}


def test_preparation_has_no_cuda_core_calls():
    tree = ast.parse(Path(m.__file__).read_text())
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name in ('freeze_source', 'prepare', 'bind_reference'):
            assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                           and n.func.attr in ('cuda_core', 'load_cuda', 'predict', 'encode') for n in ast.walk(fn))


def test_existing_actual_v1_receipt_journal_readonly():
    # Optional historical compatibility witness, never chooses it as new-v2 input.
    path = m.ROOT/'results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v1'
    if not (path/'compiler_receipt.json').is_file(): pytest.skip('saved compiler v1 receipt unavailable')
    m.verify_journal(path/'journal.jsonl', m.read_json(path/'compiler_receipt.json'))
