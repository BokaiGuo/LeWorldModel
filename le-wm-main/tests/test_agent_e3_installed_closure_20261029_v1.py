import pytest
from experiments.spectral_reachability import agent_e3_installed_closure_20261029_v1 as c


def test_two_scanners_cover_extensions_and_reject_links(tmp_path):
    (tmp_path/'pkg').mkdir();(tmp_path/'pkg'/'a.py').write_text('x')
    (tmp_path/'pkg'/'b.so').write_bytes(b'elf');(tmp_path/'pkg'/'c.pyc').write_bytes(b'bytecode')
    (tmp_path/'pkg'/'metadata.txt').write_text('ignored')
    expected=c.enumerate_walk(tmp_path)
    assert len(expected)==3 and c.enumerate_independent(tmp_path)==expected
    (tmp_path/'pkg'/'link.py').symlink_to(tmp_path/'pkg'/'a.py')
    with pytest.raises(ValueError):c.enumerate_walk(tmp_path)
    with pytest.raises(ValueError):c.enumerate_independent(tmp_path)


def test_verifier_detects_set_and_byte_drift(tmp_path,monkeypatch):
    root=tmp_path/'site';root.mkdir();p=root/'x.so';p.write_bytes(b'first')
    source=tmp_path/'source';source.write_bytes(b'authority');srcsha=c.old.file_hash(source)
    closure=tmp_path/'closure';obj=dict(schema=c.NAME+'.closure',source=dict(path=str(source),sha256=srcsha),root=str(root),
      python=str(c.Path(c.sys.executable).resolve()),extensions=list(c.EXTENSIONS),files={str(p):c.old.file_hash(p)},
      scope='all_regular_importable_files_under_site_packages',model_executed=False,physics_executed=False)
    sha=c.old.write_new(str(closure),obj);monkeypatch.setattr(c,'site_root',lambda:root)
    assert c.verify_closure(str(closure),sha,str(source),srcsha)[str(p)]==c.old.file_hash(p)
    p.write_bytes(b'other')
    with pytest.raises(ValueError,match='drift'):c.verify_closure(str(closure),sha,str(source),srcsha)
    p.write_bytes(b'first');(root/'new.py').write_text('x')
    with pytest.raises(ValueError,match='set changed'):c.verify_closure(str(closure),sha,str(source),srcsha)


def test_conflicting_pin_rejected(tmp_path,monkeypatch):
    source=tmp_path/'s';source.write_bytes(b's');sha=c.old.file_hash(source)
    monkeypatch.setattr(c.host,'authority',lambda *a:{'prepared':{'bindings_sha256':{'/a':'1'}}})
    monkeypatch.setattr(c,'verify_closure',lambda *a:{'/a':'2'})
    monkeypatch.setattr(c.launch.host,'read',lambda *a:{'additional_sources':{}})
    with pytest.raises(ValueError,match='collision'):c.build(str(source),sha,'/c','z')
