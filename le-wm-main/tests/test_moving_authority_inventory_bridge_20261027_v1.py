"""Metadata/RAM fixtures only; no resolver execution or scientific evidence."""
import hashlib
import json
from pathlib import Path
import pytest
from experiments.spectral_reachability import moving_authority_inventory_bridge_20261027_v1 as b


def row(i=0):
    return dict(path=str(b.ROOT / f'results/missing_{i}.json'),
                sha256=f'{i:064x}', family_hint='original_four_candidate_scope_unverified')


def raw(rows=None):
    return json.dumps(dict(schema=b.INPUT_SCHEMA, evidence_files=rows if rows is not None else [row()])).encode()


def convert(data):
    return b.convert(data, hashlib.sha256(data).hexdigest())


def test_all_618_preserved_in_order_without_opening_evidence(monkeypatch):
    rows = [dict(row(i), success=bool(i % 2), status='failed' if i % 3 else 'selected') for i in range(618)]
    monkeypatch.setattr(Path, 'open', lambda *a, **k: pytest.fail('evidence opened'))
    result = convert(raw(rows))
    assert result == dict(schema=b.OUTPUT_SCHEMA, files=[dict(path=r['path'], sha256=r['sha256'], family=r['family_hint']) for r in rows])


def test_opaque_outcome_never_decoded():
    data = raw().replace(b'"evidence_files":', b'"outcome": {"cost": NaN}, "evidence_files":')
    assert len(convert(data)['files']) == 1


@pytest.mark.parametrize('field,value', [('path','relative.json'), ('path',str(b.ROOT / 'results/../a.json')),
    ('path','/etc/passwd'), ('sha256','A'*64), ('sha256',None), ('family_hint',None), ('family_hint','')])
def test_invalid_descriptor(field, value):
    item = row(); item[field] = value
    with pytest.raises(ValueError): convert(raw([item]))


@pytest.mark.parametrize('field', ['path','sha256','family_hint'])
def test_missing_not_defaulted(field):
    item = row(); del item[field]
    with pytest.raises(ValueError): convert(raw([item]))


@pytest.mark.parametrize('changed', [{}, {'sha256':'f'*64}, {'family_hint':'moving'}])
def test_duplicate_path_failclosed(changed):
    with pytest.raises(ValueError, match='duplicate|conflict'): convert(raw([row(),dict(row(),**changed)]))


def test_same_content_different_paths_preserved():
    assert len(convert(raw([row(),dict(row(1),sha256=row()['sha256'])]))['files']) == 2


def test_sha_required_and_checked():
    for sha in (None, 'x', 'f'*64):
        with pytest.raises(ValueError): b.convert(raw(), sha)


def test_duplicate_keys_and_trailing_data():
    for data in (raw().replace(b'"family_hint":',b'"sha256":"' + b'f'*64 + b'", "family_hint":'), raw()+b'{}'):
        with pytest.raises(ValueError): convert(data)


def test_schema_and_container_types():
    for data in (raw().replace(b.INPUT_SCHEMA.encode(),b'wrong'), raw([]),
                 raw().replace(b'"evidence_files": [', b'"evidence_files": {"x":').replace(b']}', b'}}')):
        with pytest.raises(ValueError): convert(data)


def test_budgets(monkeypatch):
    monkeypatch.setattr(b,'INPUT_CAP',10)
    with pytest.raises(ValueError): convert(raw())
    monkeypatch.setattr(b,'INPUT_CAP',2*1024**2)
    monkeypatch.setattr(b,'ENTRY_CAP',1)
    with pytest.raises(ValueError): convert(raw([row(),row(1)]))


def test_cli_new_output_and_hash_mismatch(tmp_path,capsys):
    source=tmp_path/'catalog.json'; source.write_bytes(raw())
    target=tmp_path/'inventory.json'
    args=['--input',str(source),'--input-sha256',hashlib.sha256(raw()).hexdigest(),'--output',str(target)]
    b.main(args)
    summary=json.loads(capsys.readouterr().out)
    assert summary['sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
    assert summary['files']==1
    before=target.read_bytes()
    with pytest.raises(FileExistsError): b.main(args)
    assert target.read_bytes()==before
    args[3]='f'*64; args[-1]=str(tmp_path/'not-created.json')
    with pytest.raises(ValueError): b.main(args)
    assert not (tmp_path/'not-created.json').exists()


def test_new_output_refuses_dangling_symlink(tmp_path):
    target=tmp_path/'link'; target.symlink_to(tmp_path/'missing')
    with pytest.raises(FileExistsError): b.write_new(target,convert(raw()))
    assert not (tmp_path/'missing').exists()
