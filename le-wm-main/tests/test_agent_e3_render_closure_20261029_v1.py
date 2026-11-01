from types import SimpleNamespace
import pytest
from experiments.spectral_reachability import agent_e3_render_closure_20261029_v1 as r
from experiments.spectral_reachability import agent_e3_loaded_module_census_20261029_v1 as census


def test_real_import_and_png_fixture_no_physics():
    entries=r.probe()
    assert any(x['module']=='PIL.BmpImagePlugin' for x in entries)
    assert any(x['module']=='stable_worldmodel.envs.pusht.env' for x in entries)


def test_precheck_rejects_new_and_changed(monkeypatch):
    sealed={'entries':[dict(module='m',path='/old',sha256='a')]}
    monkeypatch.setattr(r,'verify',lambda *a:None)
    monkeypatch.setattr(r,'read_closure',lambda *a:(sealed,{'/old':'a'}))
    for e in (dict(module='new',path='/old',sha256='a'),dict(module='m',path='/new',sha256='a'),dict(module='m',path='/old',sha256='b')):
        monkeypatch.setattr(r,'probe',lambda:[e])
        with pytest.raises(ValueError,match='new or changed'):r.precheck('s','s','c','c','a','a')
    monkeypatch.setattr(r,'probe',lambda:sealed['entries'])
    assert r.precheck('s','s','c','c','a','a')['verified']


def test_census_still_rejects_unsealed_and_mutation(tmp_path):
    p=tmp_path/'late.py';p.write_text('original')
    modules={'late':SimpleNamespace(__file__=str(p))}
    with pytest.raises(ValueError,match='presealed'):census.snapshot({},modules,root=tmp_path)
    pins={str(p):r.old.file_hash(p)}
    census.snapshot(pins,modules,root=tmp_path)
    p.write_text('modified')
    with pytest.raises(ValueError,match='changed'):census.snapshot(pins,modules,root=tmp_path)


def test_exclusive_output(tmp_path):
    path=tmp_path/'seal.json';r.old.write_new(str(path),dict(v=1))
    with pytest.raises((ValueError,FileExistsError)):r.old.write_new(str(path),dict(v=2))
