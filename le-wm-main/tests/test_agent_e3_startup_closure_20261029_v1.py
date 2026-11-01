import hashlib
from types import SimpleNamespace
import pytest
from experiments.spectral_reachability import agent_e3_startup_closure_20261029_v1 as c


def test_inventory_hash_new_and_changed_files(tmp_path):
    p = tmp_path/'module.py'; p.write_text('a=1')
    modules = {'one': SimpleNamespace(__file__=str(p))}
    entry = c.inventory(modules, root=tmp_path)
    assert entry[0]['sha256'] == hashlib.sha256(b'a=1').hexdigest()
    p.write_text('a=2')
    assert c.inventory(modules, root=tmp_path)[0]['sha256'] != entry[0]['sha256']
    q = tmp_path/'new.so'; q.write_bytes(b'extension')
    modules['two'] = SimpleNamespace(__file__=str(q))
    assert len(c.inventory(modules, root=tmp_path)) == 2


def test_precheck_fails_on_new_or_changed_dependency(monkeypatch):
    monkeypatch.setattr(c, 'verify', lambda *args: None)
    monkeypatch.setattr(c.host, 'authority', lambda *args: {})
    monkeypatch.setattr(c, 'read_closure', lambda *args: ({'entries':[dict(module='m',path='/module',sha256='old')]}, {'/module': 'old'}))
    for item in (dict(module='m',path='/new',sha256='x'), dict(module='m',path='/module',sha256='changed'), dict(module='new',path='/module',sha256='old')):
        monkeypatch.setattr(c, 'probe', lambda binding: [item])
        with pytest.raises(ValueError, match='new or changed'):
            c.precheck('s','h','c','h','a','h')
    monkeypatch.setattr(c, 'probe', lambda binding: [dict(module='m',path='/module',sha256='old')])
    assert c.precheck('s','h','c','h','a','h')['verified']


def test_exclusive_write_and_authorization(tmp_path):
    p = tmp_path/'out.json'
    c.write_new(str(p), {'a': 1})
    with pytest.raises((ValueError, FileExistsError)): c.write_new(str(p), {'a': 2})
    with pytest.raises(ValueError): c.run('s','h','c','h','a','h',str(tmp_path/'job'),'no')
    assert not (tmp_path/'job').exists()
